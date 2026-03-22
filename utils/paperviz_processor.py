# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Processing pipeline of PaperVizAgent
"""

import asyncio
import inspect
from typing import List, Dict, Any, AsyncGenerator

import numpy as np
from tqdm.asyncio import tqdm

from agents.vanilla_agent import VanillaAgent
from agents.planner_agent import PlannerAgent
from agents.visualizer_agent import VisualizerAgent
from agents.stylist_agent import StylistAgent
from agents.critic_agent import CriticAgent
from agents.retriever_agent import RetrieverAgent
from agents.polish_agent import PolishAgent

from .config import ExpConfig
from .eval_toolkits import get_score_for_image_referenced


class PaperVizProcessor:
    """Main class for multimodal document processor"""

    def __init__(
        self,
        exp_config: ExpConfig,
        vanilla_agent: VanillaAgent,
        planner_agent: PlannerAgent,
        visualizer_agent: VisualizerAgent,
        stylist_agent: StylistAgent,
        critic_agent: CriticAgent,
        retriever_agent: RetrieverAgent,
        polish_agent: PolishAgent,
        event_callback=None,
    ):
        self.exp_config = exp_config
        self.vanilla_agent = vanilla_agent
        self.planner_agent = planner_agent
        self.visualizer_agent = visualizer_agent
        self.stylist_agent = stylist_agent
        self.critic_agent = critic_agent
        self.retriever_agent = retriever_agent
        self.polish_agent = polish_agent
        self.event_callback = event_callback

    async def _emit_event(self, event: Dict[str, Any], data: Dict[str, Any] | None = None):
        """Emit a task-progress event if the caller provided a callback."""
        if self.event_callback is None:
            return
        maybe_awaitable = self.event_callback(event, data)
        if inspect.isawaitable(maybe_awaitable):
            await maybe_awaitable

    async def _emit_text_stage(
        self,
        data: Dict[str, Any],
        stage: str,
        label: str,
        desc_key: str = "",
        suggestions_key: str = "",
        prompt: str = "",
        message: str = "",
        references: list[str] | None = None,
    ):
        await self._emit_event(
            {
                "type": "stage_complete",
                "candidate_id": data.get("candidate_id"),
                "stage": stage,
                "label": label,
                "message": message,
                "desc_key": desc_key,
                "suggestions_key": suggestions_key,
                "prompt": prompt,
                "references": references or [],
            },
            data,
        )

    async def _emit_image_stage(
        self,
        data: Dict[str, Any],
        stage: str,
        label: str,
        image_key: str,
        prompt: str = "",
        desc_key: str = "",
        message: str = "",
    ):
        await self._emit_event(
            {
                "type": "stage_complete",
                "candidate_id": data.get("candidate_id"),
                "stage": stage,
                "label": label,
                "message": message,
                "image_key": image_key,
                "desc_key": desc_key,
                "prompt": prompt,
            },
            data,
        )

    async def _emit_visualizer_stage_events(
        self,
        data: Dict[str, Any],
        task_name: str,
        include_stylist: bool = False,
        critic_round: int | None = None,
    ):
        visualizer_trace = data.get("_trace", {}).get("visualizer", {})
        if critic_round is not None:
            desc_key = f"target_{task_name}_critic_desc{critic_round}"
            image_key = f"{desc_key}_base64_jpg"
            if data.get(image_key):
                prompt = visualizer_trace.get(desc_key, {}).get("prompt", "")
                await self._emit_image_stage(
                    data,
                    stage=f"critic_render_{critic_round}",
                    label=f"Critic Render {critic_round}",
                    image_key=image_key,
                    desc_key=desc_key,
                    prompt=prompt,
                    message=f"Rendered critic round {critic_round} image.",
                )
            return

        planner_desc_key = f"target_{task_name}_desc0"
        planner_image_key = f"{planner_desc_key}_base64_jpg"
        if data.get(planner_image_key):
            await self._emit_image_stage(
                data,
                stage="planner_render",
                label="Planner Render",
                image_key=planner_image_key,
                desc_key=planner_desc_key,
                prompt=visualizer_trace.get(planner_desc_key, {}).get("prompt", ""),
                message="Rendered the planner description into an image.",
            )

        if include_stylist:
            stylist_desc_key = f"target_{task_name}_stylist_desc0"
            stylist_image_key = f"{stylist_desc_key}_base64_jpg"
            if data.get(stylist_image_key):
                await self._emit_image_stage(
                    data,
                    stage="stylist_render",
                    label="Stylist Render",
                    image_key=stylist_image_key,
                    desc_key=stylist_desc_key,
                    prompt=visualizer_trace.get(stylist_desc_key, {}).get("prompt", ""),
                    message="Rendered the stylist-refined description into an image.",
                )

    async def _run_critic_iterations(self, data: Dict[str, Any], task_name: str, max_rounds: int = 3, source: str = "stylist") -> Dict[str, Any]:
        """
        Run multi-round critic iteration (up to max_rounds).
        Returns the data with critic suggestions and updated eval_image_field.
        
        Args:
            data: Input data dictionary
            task_name: Name of the task (e.g., "diagram", "plot")
            max_rounds: Maximum number of critic iterations
            source: Source of the input for round 0 critique ("stylist" or "planner")
        """
        # Determine initial fallback image key based on source
        if source == "planner":
            current_best_image_key = f"target_{task_name}_desc0_base64_jpg"
        else: # default to stylist
            current_best_image_key = f"target_{task_name}_stylist_desc0_base64_jpg"
            
        for round_idx in range(max_rounds):
            data["current_critic_round"] = round_idx
            data = await self.critic_agent.process(data, source=source)
            critic_trace = data.get("_trace", {}).get("critic", {}).get(str(round_idx), {})
            await self._emit_text_stage(
                data,
                stage=f"critic_review_{round_idx}",
                label=f"Critic Round {round_idx}",
                desc_key=f"target_{task_name}_critic_desc{round_idx}",
                suggestions_key=f"target_{task_name}_critic_suggestions{round_idx}",
                prompt=critic_trace.get("prompt", ""),
                message=f"Completed critique for round {round_idx}.",
            )
            
            critic_suggestions_key = f"target_{task_name}_critic_suggestions{round_idx}"
            critic_suggestions = data.get(critic_suggestions_key, "")
            
            if critic_suggestions.strip() == "No changes needed.":
                print(f"[Critic Round {round_idx}] No changes needed. Stopping iteration.")
                await self._emit_event(
                    {
                        "type": "stage_complete",
                        "candidate_id": data.get("candidate_id"),
                        "stage": f"critic_stop_{round_idx}",
                        "label": f"Critic Stop {round_idx}",
                        "message": f"Critic round {round_idx} reported no further changes needed.",
                        "suggestions_key": critic_suggestions_key,
                    },
                    data,
                )
                break
            
            data = await self.visualizer_agent.process(data)
            await self._emit_visualizer_stage_events(
                data,
                task_name=task_name,
                critic_round=round_idx,
            )
            
            # Check if visualization validation succeeded
            new_image_key = f"target_{task_name}_critic_desc{round_idx}_base64_jpg"
            if new_image_key in data and data[new_image_key]:
                current_best_image_key = new_image_key
                print(f"[Critic Round {round_idx}] Completed iteration. Visualization SUCCESS.")
            else:
                print(f"[Critic Round {round_idx}] Visualization FAILED (No valid image). Rolling back to previous best: {current_best_image_key}")
                break
        
        data["eval_image_field"] = current_best_image_key
        return data

    async def process_single_query(
        self, data: Dict[str, Any], do_eval=True
    ) -> Dict[str, Any]:
        """
        Complete processing pipeline for a single query
        """
        # print(f"[DEBUG] -> Entered process_single_query for candidate {data.get('candidate_id', 'N/A')}")
        exp_mode = self.exp_config.exp_mode
        task_name = self.exp_config.task_name.lower()
        retrieval_setting = self.exp_config.retrieval_setting

        # Skip retriever if results were already populated by process_queries_batch
        already_retrieved = "top10_references" in data

        if exp_mode == "vanilla":
            data = await self.vanilla_agent.process(data)
            data["eval_image_field"] = f"vanilla_{task_name}_base64_jpg"
            vanilla_trace = data.get("_trace", {}).get("vanilla", {})
            await self._emit_image_stage(
                data,
                stage="vanilla",
                label="Vanilla",
                image_key=data["eval_image_field"],
                prompt=vanilla_trace.get("prompt", ""),
                message="Generated a candidate directly with the vanilla pipeline.",
            )

        elif exp_mode == "dev_planner":
            if not already_retrieved:
                data = await self.retriever_agent.process(data, retrieval_setting=retrieval_setting)
                retriever_trace = data.get("_trace", {}).get("retriever", {})
                await self._emit_text_stage(
                    data,
                    stage="retriever",
                    label="Retriever",
                    prompt=retriever_trace.get("prompt", ""),
                    message=f"Selected {len(data.get('top10_references', []))} references.",
                    references=data.get("top10_references", []),
                )
            data = await self.planner_agent.process(data)
            planner_trace = data.get("_trace", {}).get("planner", {})
            await self._emit_text_stage(
                data,
                stage="planner",
                label="Planner",
                desc_key=f"target_{task_name}_desc0",
                prompt=planner_trace.get("prompt", ""),
                message="Generated the initial figure description.",
            )
            data = await self.visualizer_agent.process(data)
            await self._emit_visualizer_stage_events(data, task_name=task_name)
            data["eval_image_field"] = f"target_{task_name}_desc0_base64_jpg"

        elif exp_mode == "dev_planner_stylist":
            if not already_retrieved:
                data = await self.retriever_agent.process(data, retrieval_setting=retrieval_setting)
                retriever_trace = data.get("_trace", {}).get("retriever", {})
                await self._emit_text_stage(
                    data,
                    stage="retriever",
                    label="Retriever",
                    prompt=retriever_trace.get("prompt", ""),
                    message=f"Selected {len(data.get('top10_references', []))} references.",
                    references=data.get("top10_references", []),
                )
            data = await self.planner_agent.process(data)
            planner_trace = data.get("_trace", {}).get("planner", {})
            await self._emit_text_stage(
                data,
                stage="planner",
                label="Planner",
                desc_key=f"target_{task_name}_desc0",
                prompt=planner_trace.get("prompt", ""),
                message="Generated the initial figure description.",
            )
            data = await self.stylist_agent.process(data)
            stylist_trace = data.get("_trace", {}).get("stylist", {})
            await self._emit_text_stage(
                data,
                stage="stylist",
                label="Stylist",
                desc_key=f"target_{task_name}_stylist_desc0",
                prompt=stylist_trace.get("prompt", ""),
                message="Refined the description with style guidance.",
            )
            data = await self.visualizer_agent.process(data)
            await self._emit_visualizer_stage_events(data, task_name=task_name, include_stylist=True)
            data["eval_image_field"] = f"target_{task_name}_stylist_desc0_base64_jpg"

        elif exp_mode in ["dev_planner_critic", "demo_planner_critic"]:
            if not already_retrieved:
                data = await self.retriever_agent.process(data, retrieval_setting=retrieval_setting)
                retriever_trace = data.get("_trace", {}).get("retriever", {})
                await self._emit_text_stage(
                    data,
                    stage="retriever",
                    label="Retriever",
                    prompt=retriever_trace.get("prompt", ""),
                    message=f"Selected {len(data.get('top10_references', []))} references.",
                    references=data.get("top10_references", []),
                )
            data = await self.planner_agent.process(data)
            planner_trace = data.get("_trace", {}).get("planner", {})
            await self._emit_text_stage(
                data,
                stage="planner",
                label="Planner",
                desc_key=f"target_{task_name}_desc0",
                prompt=planner_trace.get("prompt", ""),
                message="Generated the initial figure description.",
            )
            data = await self.visualizer_agent.process(data)
            await self._emit_visualizer_stage_events(data, task_name=task_name)
            # Use max_critic_rounds from data if available, otherwise default to 3
            max_rounds = data.get("max_critic_rounds", 3)
            data = await self._run_critic_iterations(data, task_name, max_rounds=max_rounds, source="planner")
            if "demo" in exp_mode: do_eval = False

        elif exp_mode in ["dev_full", "demo_full"]:
            if not already_retrieved:
                data = await self.retriever_agent.process(data, retrieval_setting=retrieval_setting)
                retriever_trace = data.get("_trace", {}).get("retriever", {})
                await self._emit_text_stage(
                    data,
                    stage="retriever",
                    label="Retriever",
                    prompt=retriever_trace.get("prompt", ""),
                    message=f"Selected {len(data.get('top10_references', []))} references.",
                    references=data.get("top10_references", []),
                )
            data = await self.planner_agent.process(data)
            planner_trace = data.get("_trace", {}).get("planner", {})
            await self._emit_text_stage(
                data,
                stage="planner",
                label="Planner",
                desc_key=f"target_{task_name}_desc0",
                prompt=planner_trace.get("prompt", ""),
                message="Generated the initial figure description.",
            )
            data = await self.stylist_agent.process(data)
            stylist_trace = data.get("_trace", {}).get("stylist", {})
            await self._emit_text_stage(
                data,
                stage="stylist",
                label="Stylist",
                desc_key=f"target_{task_name}_stylist_desc0",
                prompt=stylist_trace.get("prompt", ""),
                message="Refined the description with style guidance.",
            )
            data = await self.visualizer_agent.process(data)
            await self._emit_visualizer_stage_events(data, task_name=task_name, include_stylist=True)
            # Use max_critic_rounds from data (if set) or config
            max_rounds = data.get("max_critic_rounds", self.exp_config.max_critic_rounds)
            data = await self._run_critic_iterations(data, task_name, max_rounds=max_rounds, source="stylist")
            if "demo" in exp_mode: do_eval = False
        
        elif exp_mode == "dev_polish":
            data = await self.polish_agent.process(data)
            data["eval_image_field"] = f"polished_{task_name}_base64_jpg"
        
        elif exp_mode == "dev_retriever":
            data = await self.retriever_agent.process(data)
            do_eval = False

        else:
            raise ValueError(f"Unknown experiment name: {exp_mode}")

        if do_eval:
            data_with_eval = await self.evaluation_function(data, exp_config=self.exp_config)
            return data_with_eval
        else:
            return data

    async def process_queries_batch(
        self,
        data_list: List[Dict[str, Any]],
        max_concurrent: int = 50,
        do_eval: bool = True,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Batch process queries with concurrency support.
        Retriever is run once before parallelization to avoid redundant API calls.
        """
        # Run Retriever once and share results across all candidates
        exp_mode = self.exp_config.exp_mode
        retrieval_setting = self.exp_config.retrieval_setting
        needs_retrieval = exp_mode not in ("vanilla", "dev_polish", "dev_retriever")

        if needs_retrieval and data_list:
            await self._emit_event(
                {
                    "type": "run_progress",
                    "stage": "retriever_start",
                    "label": "Retriever",
                    "message": "Running shared retrieval before candidate generation.",
                    "candidate_id": None,
                }
            )
            print("[Retriever] Running retrieval once for all candidates...")
            first_data = data_list[0]
            first_data = await self.retriever_agent.process(first_data, retrieval_setting=retrieval_setting)
            retrieval_keys = ("top10_references", "retrieved_examples")
            for data in data_list[1:]:
                for key in retrieval_keys:
                    if key in first_data:
                        data[key] = first_data[key]
            print(f"[Retriever] Done. Retrieved {len(first_data.get('top10_references', []))} references.")
            retriever_trace = first_data.get("_trace", {}).get("retriever", {})
            await self._emit_event(
                {
                    "type": "stage_complete",
                    "stage": "retriever",
                    "label": "Retriever",
                    "message": f"Shared retrieval selected {len(first_data.get('top10_references', []))} references.",
                    "candidate_id": None,
                    "references": first_data.get("top10_references", []),
                    "prompt": retriever_trace.get("prompt", ""),
                },
                first_data,
            )

        semaphore = asyncio.Semaphore(max_concurrent)
        async def process_with_semaphore(doc):
            async with semaphore:
                return await self.process_single_query(doc, do_eval=do_eval)

        # Create all tasks
        tasks = []
        for data in data_list:
            task = asyncio.create_task(process_with_semaphore(data))
            tasks.append(task)
        
        all_result_list = []
        eval_dims = ["faithfulness", "conciseness", "readability", "aesthetics", "overall"]

        with tqdm(total=len(tasks), desc="Processing concurrently",ascii=True) as pbar:
            # Iterate through completed tasks returned by as_completed
            for future in asyncio.as_completed(tasks):
                result_data = await future
                all_result_list.append(result_data)
                postfix_dict = {}

                for dim in eval_dims:
                    winner_key = f"{dim}_outcome"
                    if winner_key in result_data:
                        winners = [d.get(winner_key) for d in all_result_list]
                        total = len(winners)

                        if total > 0:
                            h_cnt = winners.count("Human")
                            m_cnt = winners.count("Model")
                            t_cnt = winners.count("Tie") + winners.count("Both are good") + winners.count("Both are bad")

                            h_rate = (h_cnt / total) * 100
                            m_rate = (m_cnt / total) * 100
                            t_rate = (t_cnt / total) * 100

                            display_key = dim[:5].capitalize()
                            postfix_dict[display_key] = f"{m_rate:.0f}/{t_rate:.0f}/{h_rate:.0f}"

                pbar.set_postfix(postfix_dict)
                pbar.update(1)
                await self._emit_event(
                    {
                        "type": "candidate_complete",
                        "candidate_id": result_data.get("candidate_id"),
                        "stage": "candidate_complete",
                        "label": "Candidate Complete",
                        "message": f"Candidate {result_data.get('candidate_id')} finished processing.",
                    },
                    result_data,
                )
                yield result_data

    async def evaluation_function(
        self, data: Dict[str, Any], exp_config: ExpConfig
    ) -> Dict[str, Any]:
        """
        Evaluation function - uses referenced setting (GT shown first)
        """
        data = await get_score_for_image_referenced(
            data, task_name=exp_config.task_name, work_dir=exp_config.work_dir
        )
        return data

