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
Parallel Streamlit Demo for PaperVizAgent
Accepts user text input, duplicates it 10 times, and runs parallel processing
to generate multiple diagram candidates for comparison.
"""

import streamlit as st
import asyncio
import base64
import json
from io import BytesIO
from PIL import Image
from pathlib import Path
import sys
import os
from datetime import datetime
import importlib
import zipfile

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.console_utils import setup_console

setup_console()

print("DEBUG: Importing agents...")
import yaml
import shutil
configs_dir = Path(__file__).parent / "configs"
config_path = configs_dir / "model_config.yaml"
template_path = configs_dir / "model_config.template.yaml"

if not config_path.exists() and template_path.exists():
    print(f"DEBUG: {config_path.name} not found. Auto-generating from template")
    shutil.copy2(template_path, config_path)
try:
    from agents.planner_agent import PlannerAgent
    print("DEBUG: Imported PlannerAgent")
    from agents.visualizer_agent import VisualizerAgent
    from agents.stylist_agent import StylistAgent
    from agents.critic_agent import CriticAgent
    from agents.retriever_agent import RetrieverAgent
    from agents.vanilla_agent import VanillaAgent
    from agents.polish_agent import PolishAgent
    print("DEBUG: Imported all agents")
    from utils import config
    from utils import generation_utils
    from utils import skill_library
    from utils import task_history
    from utils.paperviz_processor import PaperVizProcessor
    print("DEBUG: Imported utils")

    model_config_data = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            model_config_data = yaml.safe_load(f) or {}

    def get_config_val(section, key, env_var, default=""):
        val = os.getenv(env_var)
        if not val and section in model_config_data:
            val = model_config_data[section].get(key)
        return val or default

except ImportError as e:
    print(f"DEBUG: ImportError: {e}")
    raise
except Exception as e:
    print(f"DEBUG: Exception during import: {e}")
    raise

st.set_page_config(
    layout="wide",
    page_title="PaperBanana Studio",
    page_icon="🍌"
)

TEXT_PROVIDER_OPTIONS = ["gemini", "openrouter", "openai", "anthropic", "custom"]
IMAGE_PROVIDER_OPTIONS = ["gemini", "openrouter", "openai", "custom"]
IMAGE_SIZE_OPTIONS = ["1K", "2K", "4K"]
FIGURE_TEXT_LANGUAGE_OPTIONS = ["zh", "en"]
PROVIDER_LABELS = {
    "gemini": "Gemini",
    "openrouter": "OpenRouter",
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "custom": "Custom API",
}
LANGUAGE_LABELS = {
    "zh": "中文",
    "en": "English",
}
CUSTOM_OPTION = "__custom__"
EXAMPLE_NONE = "__none__"
EXAMPLE_PAPERVIZ = "__paperviz__"
IMAGE_SOURCE_UPLOAD = "__upload__"
IMAGE_SOURCE_GENERATED = "__generated__"

TRANSLATIONS = {
    "language_label": {"zh": "界面语言", "en": "Language"},
    "app_title": {"zh": "PaperBanana 工作台", "en": "PaperBanana Studio"},
    "app_subtitle": {
        "zh": "用可配置的多模型路由生成、精修并导出学术插图。",
        "en": "Generate, refine, and export academic figures with configurable multi-provider model routing.",
    },
    "tab_generate": {"zh": "生成候选图", "en": "Generate Candidates"},
    "tab_refine": {"zh": "精修图片", "en": "Refine Image"},
    "config_heading": {"zh": "配置模型与 API 提供方", "en": "Configure your models and API providers"},
    "configured_providers": {"zh": "已配置提供方", "en": "Configured Providers"},
    "live_clients": {"zh": "已初始化客户端", "en": "Live Clients"},
    "custom_endpoint": {"zh": "自定义接口", "en": "Custom Endpoint"},
    "enabled": {"zh": "已启用", "en": "Enabled"},
    "not_set": {"zh": "未设置", "en": "Not set"},
    "ready": {"zh": "可用", "en": "Ready"},
    "configured": {"zh": "已配置", "en": "Configured"},
    "missing": {"zh": "缺失", "en": "Missing"},
    "provider_custom": {"zh": "自定义 API", "en": "Custom API"},
    "main_model_provider": {"zh": "主模型提供方", "en": "Main model provider"},
    "main_model_name": {"zh": "主模型名称", "en": "Main model name"},
    "main_model_help": {
        "zh": "会保存为显式的 provider/model 形式，例如 custom/Qwen/Qwen2.5-VL-72B-Instruct",
        "en": "Saved as an explicit provider-prefixed model id, e.g. custom/Qwen/Qwen2.5-VL-72B-Instruct",
    },
    "gemini_api_key": {"zh": "Gemini API Key", "en": "Gemini API key"},
    "anthropic_api_key": {"zh": "Anthropic API Key", "en": "Anthropic API key"},
    "image_model_provider": {"zh": "出图模型提供方", "en": "Image model provider"},
    "image_model_name": {"zh": "出图模型名称", "en": "Image model name"},
    "image_model_help": {
        "zh": "如果使用自定义 OpenAI-compatible 接口，这里填写该接口暴露的模型名。",
        "en": "For custom OpenAI-compatible endpoints, use a model exposed by your base URL.",
    },
    "openai_api_key": {"zh": "OpenAI API Key", "en": "OpenAI API key"},
    "openrouter_api_key": {"zh": "OpenRouter API Key", "en": "OpenRouter API key"},
    "custom_base_url": {"zh": "自定义 OpenAI-compatible Base URL", "en": "Custom OpenAI-compatible base URL"},
    "custom_base_url_help": {
        "zh": "用于自建或代理的 OpenAI 兼容接口。",
        "en": "Use this for self-hosted or proxy endpoints that speak the OpenAI API format.",
    },
    "custom_api_key": {"zh": "自定义 API Key", "en": "Custom API key"},
    "custom_api_key_help": {
        "zh": "如果是无需鉴权的本地接口，这里可以留空。",
        "en": "Can be left blank for local endpoints that do not require auth.",
    },
    "vertex_project": {"zh": "Vertex AI 项目 ID", "en": "Vertex AI project id"},
    "vertex_location": {"zh": "Vertex AI 区域", "en": "Vertex AI location"},
    "save_configuration": {"zh": "保存配置", "en": "Save configuration"},
    "config_saved": {
        "zh": "配置已保存，运行时客户端也已重新加载，新设置现在可以直接使用。",
        "en": "Configuration saved. The runtime clients were reloaded, so the new provider settings are ready to use.",
    },
    "config_tip": {
        "zh": "提示：自定义接口支持 OpenAI-compatible API。界面会自动帮你写入 `custom/<model>` 前缀。",
        "en": "Tip: the custom endpoint supports OpenAI-compatible APIs. Use `custom/<model>` internally; the UI writes that prefix for you automatically.",
    },
    "generate_intro": {
        "zh": "根据论文方法部分和图注，一次生成多张候选学术插图。",
        "en": "Generate multiple diagram candidates from your method section and caption.",
    },
    "defaults_from_config": {
        "zh": "当前默认配置：text=`{text_model}` | image=`{image_model}`",
        "en": "Defaults from config: text=`{text_model}` | image=`{image_model}`",
    },
    "generation_settings": {"zh": "生成设置", "en": "Generation Settings"},
    "pipeline_mode": {"zh": "流水线模式", "en": "Pipeline Mode"},
    "pipeline_mode_help": {"zh": "选择要使用的 Agent 流程", "en": "Select which agent pipeline to use"},
    "retrieval_setting": {"zh": "参考检索方式", "en": "Retrieval Setting"},
    "retrieval_setting_help": {
        "zh": "auto 自动检索，manual 使用预置参考，random 随机参考，none 不使用参考。",
        "en": "How to retrieve reference diagrams: auto (automatic selection), manual (use specified references), random (random selection), none (no retrieval)",
    },
    "num_candidates": {"zh": "候选数量", "en": "Number of Candidates"},
    "num_candidates_help": {"zh": "并行生成多少个候选结果", "en": "How many parallel candidates to generate"},
    "aspect_ratio": {"zh": "宽高比", "en": "Aspect Ratio"},
    "aspect_ratio_help": {"zh": "生成图片的宽高比", "en": "Aspect ratio for the generated diagrams"},
    "max_critic_rounds": {"zh": "Critic 最大轮数", "en": "Max Critic Rounds"},
    "max_critic_rounds_help": {"zh": "最多进行多少轮 Critic 精修", "en": "Maximum number of critic refinement iterations"},
    "custom_option": {"zh": "自定义", "en": "Custom"},
    "custom_main_model": {"zh": "自定义主模型", "en": "Custom Main Model"},
    "custom_image_model": {"zh": "自定义出图模型", "en": "Custom Image Generation Model"},
    "input_heading": {"zh": "输入内容", "en": "Input"},
    "load_example_method": {"zh": "加载示例（方法）", "en": "Load Example (Method)"},
    "load_example_caption": {"zh": "加载示例（图注）", "en": "Load Example (Caption)"},
    "example_none": {"zh": "不加载", "en": "None"},
    "example_paperviz": {"zh": "PaperVizAgent 示例", "en": "PaperVizAgent Framework"},
    "method_content": {"zh": "方法部分内容（推荐 Markdown）", "en": "Method Section Content (Markdown recommended)"},
    "method_content_placeholder": {"zh": "在这里粘贴论文方法部分内容……", "en": "Paste the method section content here..."},
    "method_content_help": {"zh": "用于描述方法流程的论文正文，推荐 Markdown 格式。", "en": "The method section from the paper that describes the approach. Markdown format is recommended."},
    "figure_caption": {"zh": "图注（推荐 Markdown）", "en": "Figure Caption (Markdown recommended)"},
    "figure_caption_placeholder": {"zh": "在这里输入图注……", "en": "Enter the figure caption..."},
    "figure_caption_help": {"zh": "希望生成的图的说明文字，推荐 Markdown 格式。", "en": "The caption or description of the figure to generate. Markdown format is recommended."},
    "generate_button": {"zh": "开始生成候选图", "en": "Generate Candidates"},
    "missing_content_error": {"zh": "请同时填写方法内容和图注。", "en": "Please provide both method content and caption!"},
    "generating_spinner": {"zh": "正在并行生成 {count} 个候选结果，这可能需要几分钟……", "en": "Generating {count} candidates in parallel... This may take a few minutes."},
    "success_generated": {"zh": "成功生成 {count} 个候选结果。", "en": "Successfully generated {count} candidates!"},
    "results_saved_to": {"zh": "结果已保存到：`{name}`", "en": "Results saved to: `{name}`"},
    "save_failed_warning": {"zh": "已生成 {count} 个候选结果，但保存 JSON 失败：{error}", "en": "Generated {count} candidates, but failed to save JSON: {error}"},
    "processing_error": {"zh": "处理过程中出错：{error}", "en": "Error during processing: {error}"},
    "generated_candidates": {"zh": "生成结果", "en": "Generated Candidates"},
    "generated_at": {"zh": "生成时间：{timestamp} | 流水线：{pipeline}", "en": "Generated at: {timestamp} | Pipeline: {pipeline}"},
    "candidates_metric": {"zh": "候选数", "en": "Candidates"},
    "images_ready": {"zh": "已生成图片", "en": "Images Ready"},
    "image_model_metric": {"zh": "出图模型", "en": "Image Model"},
    "download_json": {"zh": "下载 JSON", "en": "Download JSON"},
    "batch_download": {"zh": "批量下载", "en": "Batch Download"},
    "download_zip": {"zh": "下载 ZIP", "en": "Download ZIP"},
    "zip_ready": {"zh": "ZIP 已准备好，可直接下载。", "en": "ZIP file ready for download!"},
    "zip_failed": {"zh": "创建 ZIP 失败：{error}", "en": "Failed to create ZIP: {error}"},
    "dataset_missing_warning": {
        "zh": "未找到本地参考数据集：`{path}`。本次会自动切换为 `none`，不使用 few-shot 参考图。",
        "en": "Local reference dataset was not found at `{path}`. This run will automatically switch to `none` and continue without few-shot reference images.",
    },
    "manual_missing_warning": {
        "zh": "未找到手动参考文件：`{path}`。本次会自动切换为 `none`。",
        "en": "Manual reference file was not found at `{path}`. This run will automatically switch to `none`.",
    },
    "final_stage": {"zh": "最终阶段：`{stage}`", "en": "Final stage: `{stage}`"},
    "candidate_final": {"zh": "候选 {candidate_id}（最终）", "en": "Candidate {candidate_id} (Final)"},
    "download_image": {"zh": "下载图片", "en": "Download"},
    "decode_failed": {"zh": "候选 {candidate_id} 的图片解码失败", "en": "Failed to decode image for Candidate {candidate_id}"},
    "no_image_generated": {"zh": "候选 {candidate_id} 没有生成图片", "en": "No image generated for Candidate {candidate_id}"},
    "view_evolution": {"zh": "查看演化过程（{count} 个阶段）", "en": "View Evolution Timeline ({count} stages)"},
    "evolution_caption": {"zh": "查看图片在不同流水线阶段中的演变过程。", "en": "See how the diagram evolved through different pipeline stages"},
    "description": {"zh": "描述", "en": "Description"},
    "critic_suggestions": {"zh": "Critic 建议", "en": "Critic Suggestions"},
    "no_changes_needed": {"zh": "无需进一步修改，迭代已停止。", "en": "No changes needed - iteration stopped."},
    "view_description": {"zh": "查看描述", "en": "View Description"},
    "no_description_available": {"zh": "暂无描述", "en": "No description available"},
    "stage_planner": {"zh": "Planner", "en": "Planner"},
    "stage_stylist": {"zh": "Stylist", "en": "Stylist"},
    "stage_critic_round": {"zh": "Critic 第 {round_idx} 轮", "en": "Critic Round {round_idx}"},
    "stage_planner_desc": {"zh": "根据方法内容生成的初始图示规划", "en": "Initial diagram plan based on method content"},
    "stage_stylist_desc": {"zh": "经过风格优化后的描述", "en": "Stylistically refined description"},
    "stage_critic_desc": {"zh": "根据 Critic 反馈完成的第 {round_idx} 轮优化", "en": "Refined after critic feedback (iteration {round_idx})"},
    "refine_intro": {"zh": "把已有图示精修或放大到更高分辨率（2K/4K）。", "en": "Refine and upscale your diagram to high resolution (2K/4K)"},
    "refine_caption": {"zh": "上传任意图片或直接选用候选结果，描述修改要求后即可生成高分辨率版本。", "en": "Upload an image from the candidates or any diagram, describe changes, and generate a high-res version"},
    "refinement_settings": {"zh": "精修设置", "en": "Refinement Settings"},
    "target_resolution": {"zh": "目标分辨率", "en": "Target Resolution"},
    "target_resolution_help": {"zh": "更高分辨率会更慢，但效果通常更好。", "en": "Higher resolution takes longer but produces better quality"},
    "upload_heading": {"zh": "上传图片", "en": "Upload Image"},
    "image_source": {"zh": "图片来源", "en": "Image source"},
    "upload_image_option": {"zh": "上传图片", "en": "Upload image"},
    "generated_image_option": {"zh": "使用已生成候选图", "en": "Use a generated candidate"},
    "choose_file": {"zh": "选择图片文件", "en": "Choose an image file"},
    "choose_file_help": {"zh": "上传你想继续精修的图。", "en": "Upload the diagram you want to refine"},
    "choose_candidate": {"zh": "选择一个候选结果", "en": "Choose a generated candidate"},
    "candidate_label": {"zh": "候选 {candidate_id}", "en": "Candidate {candidate_id}"},
    "original_image": {"zh": "原图", "en": "Original Image"},
    "edit_instructions": {"zh": "修改说明", "en": "Edit Instructions"},
    "describe_changes": {"zh": "描述你希望修改的内容", "en": "Describe the changes you want"},
    "edit_placeholder": {"zh": "例如：'换成更适合论文的配色'、'把文字放大加粗'，或 '不改内容，仅提高分辨率'。", "en": "E.g., 'Change the color scheme to match academic paper style' or 'Make the text larger and bolder' or 'Keep everything the same but output in higher resolution'"},
    "edit_help": {"zh": "如果只是放大分辨率，可以写“保持不变，仅提升分辨率”。", "en": "Describe what you want to change or use 'Keep everything the same' for just upscaling"},
    "refine_button": {"zh": "开始精修图片", "en": "Refine Image"},
    "edit_required": {"zh": "请先填写修改说明。", "en": "Please provide edit instructions!"},
    "refining_spinner": {"zh": "正在将图片精修到 {resolution} 分辨率，这可能需要一点时间……", "en": "Refining image to {resolution} resolution... This may take a minute."},
    "refine_error": {"zh": "精修时出错：{error}", "en": "Error during refinement: {error}"},
    "refined_result": {"zh": "精修结果", "en": "Refined Result"},
    "refined_generated_at": {"zh": "生成时间：{timestamp} | 分辨率：{resolution}", "en": "Generated at: {timestamp} | Resolution: {resolution}"},
    "before": {"zh": "修改前", "en": "Before"},
    "after": {"zh": "修改后（{resolution}）", "en": "After ({resolution})"},
    "download_refined": {"zh": "下载 {resolution} 图片", "en": "Download {resolution} Image"},
    "refine_success": {"zh": "图片精修成功！来源：{provider}", "en": "Image refined successfully! (via {provider})"},
    "refine_no_image": {"zh": "当前模型没有返回图片数据。", "en": "No image data returned by the configured model."},
    "refine_runtime_error": {"zh": "精修失败：{error}", "en": "Refinement error: {error}"},
}

TRANSLATIONS.update(
    {
        "generation_resolution": {"zh": "生成分辨率", "en": "Generation Resolution"},
        "generation_resolution_help": {
            "zh": "在候选图生成阶段直接指定输出分辨率。",
            "en": "Choose the target resolution during candidate generation.",
        },
        "figure_language": {"zh": "图内文字语言", "en": "In-Figure Text Language"},
        "figure_language_help": {
            "zh": "会通过内置提示词约束图中的标签、注释和文字语言。",
            "en": "This is enforced through the built-in prompts for labels and annotations inside the figure.",
        },
        "figure_language_zh": {"zh": "中文", "en": "Chinese"},
        "figure_language_en": {"zh": "英文", "en": "English"},
        "custom_text_endpoint": {"zh": "自定义文本接口", "en": "Custom Text Endpoint"},
        "custom_image_endpoint": {"zh": "自定义图片接口", "en": "Custom Image Endpoint"},
        "custom_text_base_url": {"zh": "文本模型 Base URL", "en": "Text model base URL"},
        "custom_text_base_url_help": {
            "zh": "用于文本模型的 OpenAI-compatible 接口地址。",
            "en": "OpenAI-compatible endpoint used for text models.",
        },
        "custom_text_api_key": {"zh": "文本模型 API Key", "en": "Text model API key"},
        "custom_text_api_key_help": {
            "zh": "如果文本接口无需鉴权，可以留空。",
            "en": "Can be empty if the text endpoint does not require authentication.",
        },
        "custom_image_base_url": {"zh": "图片模型 Base URL", "en": "Image model base URL"},
        "custom_image_base_url_help": {
            "zh": "用于图片模型的 OpenAI-compatible 接口地址。",
            "en": "OpenAI-compatible endpoint used for image models.",
        },
        "custom_image_api_key": {"zh": "图片模型 API Key", "en": "Image model API key"},
        "custom_image_api_key_help": {
            "zh": "如果图片接口无需鉴权，可以留空。",
            "en": "Can be empty if the image endpoint does not require authentication.",
        },
        "endpoint_not_configured": {"zh": "未配置", "en": "Not configured"},
        "endpoint_validation_ok": {"zh": "检测成功", "en": "Endpoint validated"},
        "endpoint_validation_failed": {"zh": "检测失败", "en": "Endpoint validation failed"},
        "endpoint_models_detected": {"zh": "检测到 {count} 个模型", "en": "{count} models detected"},
        "endpoint_models_empty": {"zh": "接口可访问，但没有返回模型列表。", "en": "The endpoint is reachable, but it returned no model list."},
        "endpoint_error": {"zh": "错误：{error}", "en": "Error: {error}"},
        "detected_model_choice": {"zh": "使用已检测模型", "en": "Use a detected model"},
        "detected_model_help": {
            "zh": "保存配置后会自动探测接口可用模型，你也可以继续手动填写。",
            "en": "After saving, the app will auto-detect available models. You can still type a model manually.",
        },
        "custom_detection_hint": {
            "zh": "填写并保存自定义接口后，系统会自动检测接口有效性并拉取可用模型列表。",
            "en": "After you save the custom endpoint settings, the app will automatically validate the endpoints and fetch the available model list.",
        },
        "config_saved_notice": {
            "zh": "配置已保存，接口状态和模型列表也已重新检测。",
            "en": "Configuration saved, and endpoint status plus model lists were refreshed.",
        },
        "tab_history": {"zh": "历史任务", "en": "Task History"},
        "history_intro": {
            "zh": "查看过去运行过的任务、阶段事件、提示词和导出结果。",
            "en": "Browse previous runs, stage events, prompts, and exported results.",
        },
        "history_empty": {"zh": "还没有历史任务记录。", "en": "No task history has been recorded yet."},
        "history_select": {"zh": "选择历史任务", "en": "Select a task"},
        "history_refresh": {"zh": "刷新历史列表", "en": "Refresh history"},
        "task_status": {"zh": "任务状态", "en": "Task Status"},
        "task_status_running": {"zh": "运行中", "en": "Running"},
        "task_status_completed": {"zh": "已完成", "en": "Completed"},
        "task_status_failed": {"zh": "失败", "en": "Failed"},
        "task_created_at": {"zh": "创建时间", "en": "Created At"},
        "task_pipeline": {"zh": "流水线", "en": "Pipeline"},
        "task_candidates": {"zh": "候选数", "en": "Candidates"},
        "task_live_progress": {"zh": "实时任务进度", "en": "Live Task Progress"},
        "task_live_caption": {
            "zh": "运行中会实时写入阶段事件；每个事件都可以展开查看提示词和中间结果。",
            "en": "Stage events are recorded live while the task runs. Expand an event to inspect prompts and intermediate outputs.",
        },
        "task_timeline": {"zh": "任务时间线", "en": "Task Timeline"},
        "task_input": {"zh": "任务输入", "en": "Task Input"},
        "task_settings": {"zh": "任务设置", "en": "Task Settings"},
        "task_error_label": {"zh": "错误信息", "en": "Error"},
        "task_event_prompt": {"zh": "提示词", "en": "Prompt"},
        "task_event_output": {"zh": "阶段输出", "en": "Stage Output"},
        "task_event_references": {"zh": "参考结果", "en": "Retrieved References"},
        "task_event_image": {"zh": "阶段预览", "en": "Stage Preview"},
        "task_event_suggestions": {"zh": "Critic 建议", "en": "Critic Suggestions"},
        "task_results": {"zh": "历史结果", "en": "Task Results"},
        "task_results_file": {"zh": "结果文件", "en": "Results File"},
        "task_export_file": {"zh": "导出文件", "en": "Exported JSON"},
        "task_no_results": {"zh": "这个任务还没有保存结果。", "en": "This task does not have saved results yet."},
        "live_progress_waiting": {"zh": "等待任务开始……", "en": "Waiting for a task to start..."},
        "live_progress_current": {"zh": "已完成 {done}/{total} 个候选", "en": "{done}/{total} candidates completed"},
        "history_task_label": {
            "zh": "{task_id} | {created_at} | {status}",
            "en": "{task_id} | {created_at} | {status}",
        },
        "tab_skills": {"zh": "Skills 管理", "en": "Skills"},
        "skills_intro": {
            "zh": "导入、浏览和删除本地 skills 包。支持 zip 压缩包、单个 skill 目录和批量文件夹导入。",
            "en": "Import, browse, and delete local skill packages. Supports zip archives, single skill folders, and batch folder imports.",
        },
        "skills_root": {"zh": "本地技能仓库", "en": "Local Skill Library"},
        "skills_installed_count": {"zh": "已安装 Skills", "en": "Installed Skills"},
        "skills_import_heading": {"zh": "导入 Skills", "en": "Import Skills"},
        "skills_import_path": {"zh": "从本地路径导入", "en": "Import from local path"},
        "skills_import_path_help": {
            "zh": "可以填写单个 zip、单个 skill 目录，或者包含多个 zip 的文件夹路径。",
            "en": "Use a zip archive, a single skill directory, or a folder that contains multiple zip archives.",
        },
        "skills_overwrite": {"zh": "覆盖同名 skill", "en": "Overwrite skills with the same name"},
        "skills_import_button": {"zh": "导入路径中的 Skills", "en": "Import Skills from Path"},
        "skills_upload_label": {"zh": "上传 zip 压缩包导入", "en": "Upload zip archives"},
        "skills_upload_help": {
            "zh": "可以一次上传多个 zip，每个 zip 内应包含一个带 `SKILL.md` 的目录。",
            "en": "Upload one or more zip archives. Each archive should contain a skill directory with `SKILL.md`.",
        },
        "skills_upload_button": {"zh": "导入上传的 Skills", "en": "Import Uploaded Skills"},
        "skills_import_success": {"zh": "成功导入 {count} 个 skill。", "en": "Imported {count} skills successfully."},
        "skills_import_empty": {"zh": "没有发现可导入的 skill。", "en": "No importable skills were found."},
        "skills_import_error": {"zh": "导入失败：{error}", "en": "Import failed: {error}"},
        "skills_delete_heading": {"zh": "删除已安装 Skills", "en": "Delete Installed Skills"},
        "skills_delete_select": {"zh": "选择要删除的 skills", "en": "Select skills to delete"},
        "skills_delete_button": {"zh": "删除所选 Skills", "en": "Delete Selected Skills"},
        "skills_delete_success": {"zh": "已删除 {count} 个 skill。", "en": "Deleted {count} skills."},
        "skills_none_installed": {"zh": "当前还没有安装任何额外 skill。", "en": "No extra skills are installed yet."},
        "skills_preview_heading": {"zh": "已安装 Skills 预览", "en": "Installed Skills Preview"},
        "skills_preview_select": {"zh": "选择一个 skill 查看详情", "en": "Choose a skill to inspect"},
        "skills_description": {"zh": "简介", "en": "Description"},
        "skills_location": {"zh": "位置", "en": "Location"},
        "skills_preview_content": {"zh": "SKILL.md 预览", "en": "SKILL.md Preview"},
    }
)


def tr(key, **kwargs):
    lang = st.session_state.get("ui_language", "zh")
    text = TRANSLATIONS.get(key, {}).get(lang) or TRANSLATIONS.get(key, {}).get("en") or key
    return text.format(**kwargs)


def get_provider_label(provider):
    if provider == "custom":
        return tr("provider_custom")
    return PROVIDER_LABELS.get(provider, provider)

st.markdown(
    """
    <style>
    .pb-card {
        border: 1px solid rgba(49, 51, 63, 0.18);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(248,250,252,0.96));
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        margin-bottom: 0.8rem;
    }
    .pb-chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin-top: 0.4rem;
    }
    .pb-chip {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        border-radius: 999px;
        background: rgba(16, 185, 129, 0.12);
        color: #047857;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .pb-subtle {
        color: #475569;
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_model_config():
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def ensure_config_sections(cfg):
    cfg.setdefault("defaults", {})
    cfg.setdefault("api_keys", {})
    cfg.setdefault("api_base_urls", {})
    cfg.setdefault("google_cloud", {})
    cfg["defaults"].setdefault("main_model_name", "gemini/gemini-3.1-pro-preview")
    cfg["defaults"].setdefault("image_gen_model_name", "gemini/gemini-3.1-flash-image-preview")
    cfg["api_keys"].setdefault("google_api_key", "")
    cfg["api_keys"].setdefault("openai_api_key", "")
    cfg["api_keys"].setdefault("anthropic_api_key", "")
    cfg["api_keys"].setdefault("openrouter_api_key", "")
    cfg["api_keys"].setdefault("custom_api_key", "")
    cfg["api_keys"].setdefault("custom_text_api_key", "")
    cfg["api_keys"].setdefault("custom_image_api_key", "")
    cfg["api_base_urls"].setdefault("custom_base_url", "")
    cfg["api_base_urls"].setdefault("custom_text_base_url", "")
    cfg["api_base_urls"].setdefault("custom_image_base_url", "")
    cfg["google_cloud"].setdefault("project_id", "")
    cfg["google_cloud"].setdefault("location", "global")
    return cfg


def refresh_model_config():
    global model_config_data
    model_config_data = ensure_config_sections(load_model_config())
    return model_config_data


def save_model_config(updated_cfg):
    global model_config_data
    updated_cfg = ensure_config_sections(updated_cfg)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(updated_cfg, f, sort_keys=False, allow_unicode=True)
    model_config_data = updated_cfg


def reload_generation_backends():
    global generation_utils
    refresh_model_config()
    generation_utils = importlib.reload(generation_utils)


def get_custom_endpoint_config(cfg, kind):
    api_keys = cfg.get("api_keys", {})
    api_base_urls = cfg.get("api_base_urls", {})
    return {
        "base_url": (
            api_base_urls.get(f"custom_{kind}_base_url")
            or api_base_urls.get("custom_base_url")
            or ""
        ).strip(),
        "api_key": (
            api_keys.get(f"custom_{kind}_api_key")
            or api_keys.get("custom_api_key")
            or ""
        ).strip(),
    }


def refresh_custom_endpoint_detections(cfg, force=False):
    detections = {}
    for kind in ("text", "image"):
        endpoint_cfg = get_custom_endpoint_config(cfg, kind)
        state_key = f"_custom_{kind}_endpoint_detection"
        signature_key = f"{state_key}_signature"
        signature = (endpoint_cfg["base_url"], endpoint_cfg["api_key"])

        if not endpoint_cfg["base_url"]:
            detection = {
                "configured": False,
                "ok": False,
                "base_url": "",
                "models": [],
                "error": "",
            }
        elif (
            not force
            and st.session_state.get(signature_key) == signature
            and state_key in st.session_state
        ):
            detection = st.session_state[state_key]
        else:
            detection = generation_utils.validate_openai_compatible_endpoint(
                endpoint_cfg["base_url"],
                endpoint_cfg["api_key"],
            )
            detection["configured"] = True
            st.session_state[state_key] = detection
            st.session_state[signature_key] = signature

        detections[kind] = detection

    return detections


def get_detected_model_options(detection, current_model_name=""):
    options = []
    for model_name in detection.get("models", []):
        if model_name not in options:
            options.append(model_name)

    current_model_name = str(current_model_name or "").strip()
    if current_model_name and current_model_name not in options:
        options.insert(0, current_model_name)

    options.append(CUSTOM_OPTION)
    return options


def format_figure_language(value):
    return tr("figure_language_zh") if value == "zh" else tr("figure_language_en")


def render_endpoint_detection_summary(detection):
    if not detection.get("configured"):
        st.caption(tr("endpoint_not_configured"))
        return

    if detection.get("ok"):
        model_count = len(detection.get("models", []))
        st.caption(
            f"{tr('endpoint_validation_ok')} · "
            f"{tr('endpoint_models_detected', count=model_count)}"
        )
        if model_count == 0:
            st.caption(tr("endpoint_models_empty"))
    else:
        st.caption(tr("endpoint_validation_failed"))
        if detection.get("error"):
            st.caption(tr("endpoint_error", error=detection["error"]))


def sync_generation_model_widget_state(force=False):
    default_main_model = get_config_val("defaults", "main_model_name", "MAIN_MODEL_NAME", "gemini-3.1-pro-preview")
    default_image_model = get_config_val("defaults", "image_gen_model_name", "IMAGE_GEN_MODEL_NAME", "gemini-3.1-flash-image-preview")
    sync_token = (default_main_model, default_image_model)

    if force or st.session_state.get("_tab1_model_sync_token") != sync_token:
        st.session_state["tab1_model_name"] = default_main_model
        st.session_state["tab1_image_model_name"] = default_image_model
        st.session_state["_tab1_model_sync_token"] = sync_token


def compose_model_name(provider, raw_model_name):
    raw_model_name = str(raw_model_name or "").strip()
    if not raw_model_name:
        return ""
    prefix = f"{provider}/"
    if raw_model_name.startswith(prefix):
        return raw_model_name
    return f"{provider}/{raw_model_name}"


def split_model_name(model_name, fallback_provider="gemini"):
    model_name = str(model_name or "").strip()
    if not model_name:
        return fallback_provider, ""
    if "/" in model_name:
        provider, actual_model = model_name.split("/", 1)
        if provider in PROVIDER_LABELS:
            return provider, actual_model
    try:
        provider, actual_model = generation_utils.resolve_model_provider(model_name)
    except Exception:
        provider, actual_model = fallback_provider, model_name
    if provider not in PROVIDER_LABELS:
        provider = fallback_provider
    return provider, actual_model


def get_provider_hint(provider, mode="text"):
    hint_map = {
        ("gemini", "text"): "gemini-2.5-pro-exp-03-25",
        ("gemini", "image"): "gemini-2.0-flash-preview-image-generation",
        ("openrouter", "text"): "google/gemini-2.5-pro-preview",
        ("openrouter", "image"): "openai/gpt-image-1",
        ("openai", "text"): "gpt-4.1",
        ("openai", "image"): "gpt-image-1",
        ("anthropic", "text"): "claude-3-7-sonnet-latest",
        ("custom", "text"): "Qwen/Qwen2.5-VL-72B-Instruct",
        ("custom", "image"): "black-forest-labs/FLUX.1-schnell",
    }
    return hint_map.get((provider, mode), "")


def get_connection_status():
    runtime_status = generation_utils.get_provider_status()
    custom_text_endpoint = get_custom_endpoint_config(model_config_data, "text")
    custom_image_endpoint = get_custom_endpoint_config(model_config_data, "image")
    custom_configured = bool(custom_text_endpoint["base_url"] or custom_image_endpoint["base_url"])
    custom_detail_parts = []
    if custom_text_endpoint["base_url"]:
        custom_detail_parts.append(f"Text: {custom_text_endpoint['base_url']}")
    if custom_image_endpoint["base_url"]:
        custom_detail_parts.append(f"Image: {custom_image_endpoint['base_url']}")
    return [
        {
            "provider": "gemini",
            "label": "Gemini",
            "configured": bool(get_config_val("api_keys", "google_api_key", "GOOGLE_API_KEY", "")) or bool(
                get_config_val("google_cloud", "project_id", "GOOGLE_CLOUD_PROJECT", "")
            ),
            "live": runtime_status["gemini"],
            "detail": get_config_val("google_cloud", "project_id", "GOOGLE_CLOUD_PROJECT", "") or "API key or Vertex AI",
        },
        {
            "provider": "openrouter",
            "label": "OpenRouter",
            "configured": bool(get_config_val("api_keys", "openrouter_api_key", "OPENROUTER_API_KEY", "")),
            "live": runtime_status["openrouter"],
            "detail": "https://openrouter.ai/api/v1",
        },
        {
            "provider": "openai",
            "label": "OpenAI",
            "configured": bool(get_config_val("api_keys", "openai_api_key", "OPENAI_API_KEY", "")),
            "live": runtime_status["openai"],
            "detail": "Official OpenAI API",
        },
        {
            "provider": "anthropic",
            "label": "Anthropic",
            "configured": bool(get_config_val("api_keys", "anthropic_api_key", "ANTHROPIC_API_KEY", "")),
            "live": runtime_status["anthropic"],
            "detail": "Official Anthropic API",
        },
        {
            "provider": "custom",
            "label": "Custom API",
            "configured": custom_configured,
            "live": runtime_status["custom_text"] or runtime_status["custom_image"],
            "detail": " | ".join(custom_detail_parts) or "OpenAI-compatible base URL",
        },
    ]


def render_status_cards():
    rows = get_connection_status()
    configured = sum(1 for row in rows if row["configured"])
    live = sum(1 for row in rows if row["live"])
    top_cols = st.columns(3)
    top_cols[0].metric(tr("configured_providers"), configured)
    top_cols[1].metric(tr("live_clients"), live)
    top_cols[2].metric(
        tr("custom_endpoint"),
        tr("enabled") if any(row["provider"] == "custom" and row["configured"] for row in rows) else tr("not_set"),
    )

    cols = st.columns(len(rows))
    for col, row in zip(cols, rows):
        status = tr("ready") if row["live"] else (tr("configured") if row["configured"] else tr("missing"))
        with col:
            st.markdown(
                (
                    f"<div class='pb-card'><strong>{get_provider_label(row['provider'])}</strong><br>"
                    f"<span class='pb-subtle'>{row['detail']}</span>"
                    f"<div class='pb-chip-row'><span class='pb-chip'>{status}</span></div></div>"
                ),
                unsafe_allow_html=True,
            )


def get_dataset_paths(task_name="diagram"):
    dataset_dir = Path(__file__).parent / "data" / "PaperBananaBench" / task_name
    return {
        "dataset_dir": dataset_dir,
        "ref_file": dataset_dir / "ref.json",
        "manual_file": dataset_dir / "agent_selected_12.json",
    }


def get_effective_retrieval_setting(requested_setting, task_name="diagram"):
    paths = get_dataset_paths(task_name)
    if requested_setting in ("auto", "random") and not paths["ref_file"].exists():
        return "none", tr("dataset_missing_warning", path=str(paths["ref_file"]))
    if requested_setting == "manual" and not paths["manual_file"].exists():
        return "none", tr("manual_missing_warning", path=str(paths["manual_file"]))
    return requested_setting, ""


def render_onboarding_page():
    project_root = Path(__file__).parent
    dataset_paths = get_dataset_paths("diagram")
    logs_dir = project_root / "logs"
    stdout_log = logs_dir / "streamlit_stdout.log"
    stderr_log = logs_dir / "streamlit_stderr.log"
    results_dir = project_root / "results" / "demo"

    lang = st.session_state.get("ui_language", "zh")

    if lang == "zh":
        st.markdown("### 欢迎使用 PaperBanana")
        st.markdown(
            "PaperBanana 是一个面向学术插图生成的多智能体工具。你提供论文的方法部分、图注或已有图片，"
            "它会调用文本模型和图像模型，帮助你生成论文风格的框架图、流程图，并支持后续精修和下载。"
        )

        status_cols = st.columns(4)
        status_cols[0].metric("配置文件", "已就绪" if config_path.exists() else "缺失")
        status_cols[1].metric("参考数据集", "已就绪" if dataset_paths["ref_file"].exists() else "未下载")
        status_cols[2].metric("日志目录", "已创建" if logs_dir.exists() else "未创建")
        status_cols[3].metric("结果目录", "已创建" if results_dir.exists() else "未创建")

        with st.expander("这是什么", expanded=True):
            st.markdown(
                """
1. 这是一个“论文内容 -> 学术插图”的可视化工作台。
2. 你可以让它从方法文字和图注直接生成多张候选图，也可以把已有图片上传后继续精修。
3. 它支持多种模型后端，包括 Gemini、OpenAI、OpenRouter、Anthropic，以及你自己的 OpenAI-compatible 接口。
                """
            )

        with st.expander("如何快速使用", expanded=True):
            st.markdown(
                """
1. 先在上方“配置模型与 API 提供方”里填好文本模型、出图模型、URL 和 Key。
2. 自定义接口支持把文本模型和图片模型分别接到不同的 Base URL / API Key，保存后会自动检测接口和模型列表。
3. 如果你没有下载 `PaperBananaBench` 数据集，也可以直接用；系统会自动切换为无参考模式。
4. 打开“生成候选图”页，填写方法部分和图注，并选择分辨率、图内文字语言等参数，然后点击“开始生成候选图”。
5. 生成后可以查看多张候选图、下载 JSON、批量下载 ZIP，或把某个候选图送到“精修图片”页继续优化。
6. 如果只是想放大分辨率，在精修说明里写“保持不变，仅提升分辨率”即可。
                """
            )

        with st.expander("它的工作流程和原理"):
            st.markdown(
                """
1. `Retriever`：如果本地参考数据集存在，会先挑选相似参考图作为 few-shot 示例。
2. `Planner`：把方法内容和图注转成详细的绘图说明。
3. `Stylist`：补充更学术化的视觉风格要求。
4. `Visualizer`：调用图像模型把说明转成图片。
5. `Critic`：检查结果并迭代修正，尽量让结构和表达更准确。
                """
            )

        with st.expander("报错、日志和排查方法"):
            st.markdown(
                """
1. 如果打开网页时看到 `ERR_CONNECTION_REFUSED`，说明本地服务没跑起来，可以执行 `start_paperbanana.cmd`。
2. 如果生成时报 API 错误，请检查模型名是否和你的接口兼容，以及 Base URL / API Key 是否正确。
3. 如果没有下载参考数据集，系统现在会自动降级，不会再因为 `ref.json` 缺失直接崩掉。
4. 运行日志和结果文件都保留在本地，方便排查和复现。
                """
            )
            st.code(
                "\n".join(
                    [
                        f"配置文件: {config_path}",
                        f"参考数据集: {dataset_paths['ref_file']}",
                        f"标准输出日志: {stdout_log}",
                        f"错误日志: {stderr_log}",
                        f"结果目录: {results_dir}",
                        f"启动脚本: {project_root / 'start_paperbanana.cmd'}",
                        f"停止脚本: {project_root / 'stop_paperbanana.cmd'}",
                    ]
                )
            )

    else:
        st.markdown("### Welcome to PaperBanana")
        st.markdown(
            "PaperBanana is a multi-agent workspace for academic figure generation. You provide a method section, a figure caption, "
            "or an existing image, and it uses text + image models to generate paper-style diagrams, refine them, and package the results."
        )

        status_cols = st.columns(4)
        status_cols[0].metric("Config", "Ready" if config_path.exists() else "Missing")
        status_cols[1].metric("Reference Dataset", "Ready" if dataset_paths["ref_file"].exists() else "Not downloaded")
        status_cols[2].metric("Logs", "Created" if logs_dir.exists() else "Not created")
        status_cols[3].metric("Results", "Created" if results_dir.exists() else "Not created")

        with st.expander("What this is", expanded=True):
            st.markdown(
                """
1. This is a workspace that turns paper content into academic figures.
2. You can generate multiple candidate diagrams from a method section and caption, or upload an existing figure for refinement.
3. It supports Gemini, OpenAI, OpenRouter, Anthropic, and your own OpenAI-compatible endpoint.
                """
            )

        with st.expander("Quick start", expanded=True):
            st.markdown(
                """
1. Fill in the model, URL, and key settings in the configuration section above.
2. Custom endpoints can use different Base URLs / API keys for text and image models. After you save, the app auto-validates the endpoints and fetches the model list.
3. You can still run the app without downloading `PaperBananaBench`; it will automatically fall back to no-reference mode.
4. Open the Generate page, paste your method section and caption, choose resolution and in-figure text language, then click Generate.
5. Review the candidate outputs, download JSON or ZIP, or send a candidate into the Refine page.
6. If you only want higher resolution, simply ask to keep the image unchanged and upscale it.
                """
            )

        with st.expander("Workflow and core idea"):
            st.markdown(
                """
1. `Retriever`: selects similar examples if the local reference dataset is available.
2. `Planner`: converts paper content and caption into a detailed figure description.
3. `Stylist`: improves the description to better match academic visual style.
4. `Visualizer`: calls the image model to render the figure.
5. `Critic`: reviews the output and iteratively refines it.
                """
            )

        with st.expander("Errors, logs, and troubleshooting"):
            st.markdown(
                """
1. If you see `ERR_CONNECTION_REFUSED`, the local service is not running; start it with `start_paperbanana.cmd`.
2. If generation fails with an API error, double-check the model name, base URL, and API key.
3. If the reference dataset is missing, the app now auto-falls back instead of crashing on `ref.json`.
4. Logs and results are stored locally so you can inspect and reproduce failures.
                """
            )
            st.code(
                "\n".join(
                    [
                        f"Config file: {config_path}",
                        f"Reference dataset: {dataset_paths['ref_file']}",
                        f"Stdout log: {stdout_log}",
                        f"Stderr log: {stderr_log}",
                        f"Results directory: {results_dir}",
                        f"Start script: {project_root / 'start_paperbanana.cmd'}",
                        f"Stop script: {project_root / 'stop_paperbanana.cmd'}",
                    ]
                )
            )


def get_final_image_info(result, exp_mode):
    task_name = "diagram"
    for round_idx in range(3, -1, -1):
        image_key = f"target_{task_name}_critic_desc{round_idx}_base64_jpg"
        if image_key in result and result[image_key]:
            return image_key, f"critic_round_{round_idx}"

    if exp_mode == "demo_full":
        return f"target_{task_name}_stylist_desc0_base64_jpg", "stylist"
    return f"target_{task_name}_desc0_base64_jpg", "planner"


def build_results_archive(results, exp_mode):
    zip_buffer = BytesIO()
    manifest = []

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for candidate_id, result in enumerate(results):
            final_image_key, stage_name = get_final_image_info(result, exp_mode)
            if final_image_key not in result:
                continue
            img = base64_to_image(result[final_image_key])
            if img is None:
                continue

            img_buffer = BytesIO()
            img.save(img_buffer, format="PNG")
            image_name = f"candidate_{candidate_id}.png"
            zip_file.writestr(image_name, img_buffer.getvalue())
            manifest.append(
                {
                    "candidate_id": candidate_id,
                    "final_stage": stage_name,
                    "image_file": image_name,
                }
            )

    zip_buffer.seek(0)
    return zip_buffer.getvalue(), json.dumps(manifest, ensure_ascii=False, indent=2)


def clean_text(text):
    """Clean text by removing invalid UTF-8 surrogate characters."""
    if not text:
        return text
    if isinstance(text, str):
        # Remove surrogate characters that cause UnicodeEncodeError
        return text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
    return text

def base64_to_image(b64_str):
    """Convert base64 string to PIL Image."""
    if not b64_str:
        return None
    try:
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]
        image_data = base64.b64decode(b64_str)
        return Image.open(BytesIO(image_data))
    except Exception:
        return None

def create_sample_inputs(
    method_content,
    caption,
    diagram_type="Pipeline",
    aspect_ratio="16:9",
    image_size="1K",
    figure_language="en",
    num_copies=10,
    max_critic_rounds=3,
):
    """Create multiple copies of the input data for parallel processing."""
    base_input = {
        "filename": "demo_input",
        "caption": caption,
        "content": method_content,
        "visual_intent": caption,
        "additional_info": {
            "rounded_ratio": aspect_ratio,
            "image_size": image_size,
            "figure_language": figure_language,
        },
        "max_critic_rounds": max_critic_rounds  # Add critic rounds control
    }
    
    # Create num_copies identical inputs, each with a unique identifier
    inputs = []
    for i in range(num_copies):
        input_copy = base_input.copy()
        input_copy["additional_info"] = base_input["additional_info"].copy()
        input_copy["filename"] = f"demo_input_candidate_{i}"
        input_copy["candidate_id"] = i
        inputs.append(input_copy)
    
    return inputs

async def process_parallel_candidates(
    data_list,
    exp_mode="dev_planner_critic",
    retrieval_setting="auto",
    main_model_name="",
    image_gen_model_name="",
    event_callback=None,
):
    """Process multiple candidates in parallel using PaperVizProcessor."""
    # Create experiment config
    exp_config = config.ExpConfig(
        dataset_name="Demo",
        split_name="demo",
        exp_mode=exp_mode,
        retrieval_setting=retrieval_setting,
        main_model_name=main_model_name,
        image_gen_model_name=image_gen_model_name,
        work_dir=Path(__file__).parent,
    )
    
    # Initialize processor with all agents
    processor = PaperVizProcessor(
        exp_config=exp_config,
        vanilla_agent=VanillaAgent(exp_config=exp_config),
        planner_agent=PlannerAgent(exp_config=exp_config),
        visualizer_agent=VisualizerAgent(exp_config=exp_config),
        stylist_agent=StylistAgent(exp_config=exp_config),
        critic_agent=CriticAgent(exp_config=exp_config),
        retriever_agent=RetrieverAgent(exp_config=exp_config),
        polish_agent=PolishAgent(exp_config=exp_config),
        event_callback=event_callback,
    )
    
    # Process all candidates in parallel (concurrency controlled by processor)
    results = []
    concurrent_num = max(1, min(10, len(data_list)))
    
    async for result_data in processor.process_queries_batch(
        data_list, max_concurrent=concurrent_num, do_eval=False
    ):
        results.append(result_data)
    
    return results

async def refine_image_with_nanoviz(image_bytes, edit_prompt, aspect_ratio="21:9", image_size="2K"):
    """
    Refine an image using the currently selected image generation backend.
    
    Args:
        image_bytes: Image data in bytes
        edit_prompt: Text description of desired changes
        aspect_ratio: Output aspect ratio (21:9, 16:9, 3:2)
        image_size: Output resolution (2K or 4K)
    
    Returns:
        Tuple of (edited_image_bytes, success_message)
    """
    image_model = get_config_val("defaults", "image_gen_model_name", "IMAGE_GEN_MODEL_NAME", "")
    try:
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        contents = [
            {"type": "text", "text": edit_prompt},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_b64,
                },
            },
        ]
        result = await generation_utils.call_image_model_with_retry_async(
            model_name=image_model,
            prompt=edit_prompt,
            contents=contents,
            config={
                "system_prompt": "",
                "temperature": 1.0,
                "aspect_ratio": aspect_ratio,
                "image_size": generation_utils.normalize_generation_image_size(image_size),
                "quality": "high",
                "background": "opaque",
                "output_format": "png",
            },
            max_attempts=3,
            retry_delay=10,
            error_context="refine_image",
        )
        provider, _ = generation_utils.resolve_model_provider(image_model)
        if result and result[0] != "Error":
            return base64.b64decode(result[0]), tr("refine_success", provider=get_provider_label(provider))
        return None, tr("refine_no_image")
    except Exception as e:
        return None, tr("refine_runtime_error", error=str(e))


def get_evolution_stages(result, exp_mode):
    """Extract all evolution stages (images and descriptions) from the result."""
    task_name = "diagram"
    stages = []
    
    # Stage 1: Planner output
    planner_img_key = f"target_{task_name}_desc0_base64_jpg"
    planner_desc_key = f"target_{task_name}_desc0"
    if planner_img_key in result and result[planner_img_key]:
        stages.append({
            "name": tr("stage_planner"),
            "image_key": planner_img_key,
            "desc_key": planner_desc_key,
            "description": tr("stage_planner_desc")
        })
    
    # Stage 2: Stylist output (only for demo_full)
    if exp_mode == "demo_full":
        stylist_img_key = f"target_{task_name}_stylist_desc0_base64_jpg"
        stylist_desc_key = f"target_{task_name}_stylist_desc0"
        if stylist_img_key in result and result[stylist_img_key]:
            stages.append({
                "name": tr("stage_stylist"),
                "image_key": stylist_img_key,
                "desc_key": stylist_desc_key,
                "description": tr("stage_stylist_desc")
            })
    
    # Stage 3+: Critic iterations
    for round_idx in range(4):  # Check up to 4 rounds
        critic_img_key = f"target_{task_name}_critic_desc{round_idx}_base64_jpg"
        critic_desc_key = f"target_{task_name}_critic_desc{round_idx}"
        critic_sugg_key = f"target_{task_name}_critic_suggestions{round_idx}"
        
        if critic_img_key in result and result[critic_img_key]:
            stages.append({
                "name": tr("stage_critic_round", round_idx=round_idx),
                "image_key": critic_img_key,
                "desc_key": critic_desc_key,
                "suggestions_key": critic_sugg_key,
                "description": tr("stage_critic_desc", round_idx=round_idx)
            })
    
    return stages

def display_candidate_result(result, candidate_id, exp_mode, key_prefix="current"):
    """Display a single candidate result."""
    task_name = "diagram"
    final_image_key, final_stage = get_final_image_info(result, exp_mode)
    final_desc_key = None
    if final_stage.startswith("critic_round_"):
        round_idx = final_stage.split("_")[-1]
        final_desc_key = f"target_{task_name}_critic_desc{round_idx}"
    elif final_stage == "stylist":
        final_desc_key = f"target_{task_name}_stylist_desc0"
    else:
        final_desc_key = f"target_{task_name}_desc0"
    
    # Display the final image
    if final_image_key and final_image_key in result:
        img = base64_to_image(result[final_image_key])
        if img:
            st.caption(tr("final_stage", stage=final_stage))
            st.image(img, use_container_width=True, caption=tr("candidate_final", candidate_id=candidate_id))
            
            # Add download button
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            st.download_button(
                label=tr("download_image"),
                data=buffered.getvalue(),
                file_name=f"candidate_{candidate_id}.png",
                mime="image/png",
                key=f"{key_prefix}_download_candidate_{candidate_id}",
                use_container_width=True
            )
        else:
            st.error(tr("decode_failed", candidate_id=candidate_id))
    else:
        st.warning(tr("no_image_generated", candidate_id=candidate_id))
    
    # Show evolution timeline in an expander
    stages = get_evolution_stages(result, exp_mode)
    if len(stages) > 1:
        with st.expander(tr("view_evolution", count=len(stages)), expanded=False):
            st.caption(tr("evolution_caption"))
            
            for idx, stage in enumerate(stages):
                st.markdown(f"### {stage['name']}")
                st.caption(stage['description'])
                
                # Display the image for this stage
                stage_img = base64_to_image(result.get(stage['image_key']))
                if stage_img:
                    st.image(stage_img, use_container_width=True)
                
                # Show description
                if stage['desc_key'] in result:
                    with st.expander(tr("description"), expanded=False):
                        cleaned_desc = clean_text(result[stage['desc_key']])
                        st.write(cleaned_desc)
                
                # Show critic suggestions if available
                if 'suggestions_key' in stage and stage['suggestions_key'] in result:
                    suggestions = result[stage['suggestions_key']]
                    with st.expander(tr("critic_suggestions"), expanded=False):
                        cleaned_sugg = clean_text(suggestions)
                        if cleaned_sugg.strip() == "No changes needed.":
                            st.success(tr("no_changes_needed"))
                        else:
                            st.write(cleaned_sugg)
                
                # Add separator between stages (except for the last one)
                if idx < len(stages) - 1:
                    st.divider()
    else:
        # If only one stage, show description in simpler expander
        with st.expander(tr("view_description"), expanded=False):
            if final_desc_key and final_desc_key in result:
                # Clean the text to remove invalid UTF-8 characters
                cleaned_desc = clean_text(result[final_desc_key])
                st.write(cleaned_desc)
            else:
                st.info(tr("no_description_available"))


def get_task_status_label(status):
    normalized = str(status or "").strip().lower()
    if normalized == "completed":
        return tr("task_status_completed")
    if normalized == "failed":
        return tr("task_status_failed")
    return tr("task_status_running")


def trim_task_text(text, limit=6000):
    cleaned = clean_text(text or "")
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit] + "\n\n...[truncated]"


def create_image_preview_bytes(b64_str, max_size=(420, 280)):
    img = base64_to_image(b64_str)
    if img is None:
        return None
    img = img.convert("RGB")
    img.thumbnail(max_size)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def summarize_result_bundle(results, exp_mode):
    ready_count = sum(
        1
        for result in results
        if get_final_image_info(result, exp_mode)[0] in result
        and result[get_final_image_info(result, exp_mode)[0]]
    )
    return {
        "generated_count": len(results),
        "ready_count": ready_count,
    }


def create_task_tracking_state(
    *,
    method_content,
    caption,
    exp_mode,
    retrieval_setting,
    num_candidates,
    aspect_ratio,
    generation_resolution,
    figure_language,
    max_critic_rounds,
    main_model_name,
    image_gen_model_name,
):
    work_dir = Path(__file__).parent
    task_id = task_history.make_task_id("demo")
    record = task_history.build_task_record(
        task_id,
        {
            "pipeline": exp_mode,
            "retrieval_setting": retrieval_setting,
            "main_model_name": main_model_name,
            "image_gen_model_name": image_gen_model_name,
            "input": {
                "method_content": method_content,
                "caption": caption,
                "aspect_ratio": aspect_ratio,
                "generation_resolution": generation_resolution,
                "figure_language": figure_language,
                "max_critic_rounds": max_critic_rounds,
                "num_candidates": num_candidates,
            },
        },
    )
    task_history.save_task_record(work_dir, record)
    return {
        "task_id": task_id,
        "work_dir": work_dir,
        "record": record,
        "live_events": [],
        "completed_candidates": 0,
        "total_candidates": num_candidates,
    }


def build_live_event(event, data=None):
    desc_key = event.get("desc_key", "")
    suggestions_key = event.get("suggestions_key", "")
    image_key = event.get("image_key", "")
    output_text = ""
    suggestions_text = ""
    preview_bytes = None
    if data:
        if desc_key:
            output_text = trim_task_text(data.get(desc_key, ""))
        if suggestions_key:
            suggestions_text = trim_task_text(data.get(suggestions_key, ""))
        if image_key and data.get(image_key):
            preview_bytes = create_image_preview_bytes(data.get(image_key))
    prompt = trim_task_text(event.get("prompt", ""), limit=8000)
    references = event.get("references", []) or []
    return {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "type": event.get("type", "stage_complete"),
        "candidate_id": event.get("candidate_id"),
        "stage": event.get("stage", ""),
        "label": event.get("label", event.get("stage", "Event")),
        "message": clean_text(event.get("message", "")),
        "prompt": prompt,
        "desc_key": desc_key,
        "suggestions_key": suggestions_key,
        "image_key": image_key,
        "output_text": output_text,
        "suggestions_text": suggestions_text,
        "references": references,
        "preview_bytes": preview_bytes,
    }


def append_task_event(task_state, event, data=None):
    live_event = build_live_event(event, data)
    task_state["live_events"].append(live_event)
    task_state["live_events"] = task_state["live_events"][-30:]

    if event.get("type") == "candidate_complete":
        task_state["completed_candidates"] = min(
            task_state["completed_candidates"] + 1,
            task_state["total_candidates"],
        )

    persisted_event = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "type": live_event["type"],
        "candidate_id": live_event["candidate_id"],
        "stage": live_event["stage"],
        "label": live_event["label"],
        "message": live_event["message"],
        "prompt": live_event["prompt"],
        "desc_key": live_event["desc_key"],
        "suggestions_key": live_event["suggestions_key"],
        "image_key": live_event["image_key"],
        "references": live_event["references"],
        "output_preview": live_event["output_text"],
        "suggestions_preview": live_event["suggestions_text"],
    }
    task_state["record"]["events"].append(persisted_event)
    task_history.save_task_record(task_state["work_dir"], task_state["record"])


def finalize_task_tracking(task_state, *, status, results=None, error="", export_json_file=""):
    task_state["record"]["status"] = status
    task_state["record"]["error"] = clean_text(error)
    if results is not None:
        results_path = task_history.save_task_results(
            task_state["work_dir"],
            task_state["task_id"],
            results,
        )
        task_state["record"]["results_file"] = str(results_path.relative_to(task_state["work_dir"]))
        task_state["record"]["summary"] = summarize_result_bundle(
            results,
            task_state["record"].get("pipeline", "demo_full"),
        )
    if export_json_file:
        export_path = Path(export_json_file)
        if export_path.exists():
            task_state["record"]["export_json_file"] = str(export_path.relative_to(task_state["work_dir"]))
    task_history.save_task_record(task_state["work_dir"], task_state["record"])


def render_task_event(event, *, expanded=False):
    candidate_id = event.get("candidate_id")
    prefix = f"Candidate {candidate_id} · " if candidate_id is not None else "Shared · "
    title = f"{event.get('timestamp', '')} · {prefix}{event.get('label', event.get('stage', 'Event'))}"
    with st.expander(title, expanded=expanded):
        if event.get("message"):
            st.caption(event["message"])
        if event.get("references"):
            st.markdown(f"**{tr('task_event_references')}**")
            st.json(event["references"])
        if event.get("prompt"):
            st.markdown(f"**{tr('task_event_prompt')}**")
            st.code(event["prompt"], language="markdown")
        if event.get("output_text"):
            st.markdown(f"**{tr('task_event_output')}**")
            st.write(event["output_text"])
        if event.get("suggestions_text"):
            st.markdown(f"**{tr('task_event_suggestions')}**")
            st.write(event["suggestions_text"])
        if event.get("preview_bytes"):
            st.markdown(f"**{tr('task_event_image')}**")
            st.image(event["preview_bytes"], use_container_width=True)


def render_live_task_panel(task_state, container):
    with container.container():
        st.markdown(f"### {tr('task_live_progress')}")
        st.caption(tr("task_live_caption"))
        status_cols = st.columns(3)
        status_cols[0].metric(tr("task_status"), get_task_status_label(task_state["record"].get("status")))
        status_cols[1].metric(
            tr("task_candidates"),
            tr(
                "live_progress_current",
                done=task_state["completed_candidates"],
                total=task_state["total_candidates"],
            ),
        )
        status_cols[2].metric("Task ID", task_state["task_id"])
        progress_value = (
            task_state["completed_candidates"] / task_state["total_candidates"]
            if task_state["total_candidates"]
            else 0.0
        )
        if task_state["record"].get("status") == "completed":
            progress_value = 1.0
        st.progress(progress_value)
        if not task_state["live_events"]:
            st.info(tr("live_progress_waiting"))
        else:
            for idx, event in enumerate(reversed(task_state["live_events"])):
                render_task_event(event, expanded=idx == 0)


def render_history_page():
    work_dir = Path(__file__).parent
    records = task_history.list_task_records(work_dir, limit=30)
    st.markdown(f"### {tr('tab_history')}")
    st.caption(tr("history_intro"))
    if st.button(tr("history_refresh"), key="history_refresh_button"):
        st.rerun()
    if not records:
        st.info(tr("history_empty"))
        return

    options = [record["task_id"] for record in records]
    current_task_id = st.session_state.get("selected_history_task_id")
    index = options.index(current_task_id) if current_task_id in options else 0
    selected_task_id = st.selectbox(
        tr("history_select"),
        options=options,
        index=index,
        format_func=lambda task_id: tr(
            "history_task_label",
            task_id=task_id,
            created_at=next((r.get("created_at", "") for r in records if r["task_id"] == task_id), ""),
            status=get_task_status_label(next((r.get("status", "") for r in records if r["task_id"] == task_id), "")),
        ),
    )
    st.session_state["selected_history_task_id"] = selected_task_id

    record = next(record for record in records if record["task_id"] == selected_task_id)
    results = task_history.load_task_results(work_dir, selected_task_id)
    result_map = {
        result.get("candidate_id", idx): result
        for idx, result in enumerate(results)
    }

    meta_cols = st.columns(4)
    meta_cols[0].metric(tr("task_status"), get_task_status_label(record.get("status")))
    meta_cols[1].metric(tr("task_created_at"), record.get("created_at", ""))
    meta_cols[2].metric(tr("task_pipeline"), record.get("pipeline", ""))
    meta_cols[3].metric(tr("task_candidates"), str(record.get("input", {}).get("num_candidates", len(results))))

    with st.expander(tr("task_input"), expanded=False):
        st.markdown(f"**{tr('method_content')}**")
        st.write(clean_text(record.get("input", {}).get("method_content", "")))
        st.markdown(f"**{tr('figure_caption')}**")
        st.write(clean_text(record.get("input", {}).get("caption", "")))

    with st.expander(tr("task_settings"), expanded=False):
        settings_payload = {
            "pipeline": record.get("pipeline", ""),
            "retrieval_setting": record.get("retrieval_setting", ""),
            "main_model_name": record.get("main_model_name", ""),
            "image_gen_model_name": record.get("image_gen_model_name", ""),
            "aspect_ratio": record.get("input", {}).get("aspect_ratio", ""),
            "generation_resolution": record.get("input", {}).get("generation_resolution", ""),
            "figure_language": record.get("input", {}).get("figure_language", ""),
            "max_critic_rounds": record.get("input", {}).get("max_critic_rounds", ""),
        }
        st.json(settings_payload)

    if record.get("error"):
        st.error(f"{tr('task_error_label')}: {record['error']}")

    with st.expander(tr("task_timeline"), expanded=True):
        if not record.get("events"):
            st.info(tr("live_progress_waiting"))
        else:
            for idx, event in enumerate(reversed(record["events"])):
                candidate_result = result_map.get(event.get("candidate_id"))
                preview_bytes = None
                image_key = event.get("image_key", "")
                if candidate_result and image_key and candidate_result.get(image_key):
                    preview_bytes = create_image_preview_bytes(candidate_result.get(image_key))
                historical_event = {
                    "timestamp": event.get("timestamp", ""),
                    "candidate_id": event.get("candidate_id"),
                    "label": event.get("label", ""),
                    "stage": event.get("stage", ""),
                    "message": event.get("message", ""),
                    "prompt": event.get("prompt", ""),
                    "references": event.get("references", []),
                    "output_text": event.get("output_preview", ""),
                    "suggestions_text": event.get("suggestions_preview", ""),
                    "preview_bytes": preview_bytes,
                }
                render_task_event(historical_event, expanded=idx == 0)

    st.divider()
    st.markdown(f"## {tr('task_results')}")
    if record.get("results_file"):
        results_path = work_dir / record["results_file"]
        if results_path.exists():
            st.caption(f"{tr('task_results_file')}: `{record['results_file']}`")
            st.download_button(
                label=tr("download_json"),
                data=results_path.read_text(encoding="utf-8"),
                file_name=results_path.name,
                mime="application/json",
                use_container_width=False,
                key=f"history_download_{selected_task_id}",
            )
    if record.get("export_json_file"):
        st.caption(f"{tr('task_export_file')}: `{record['export_json_file']}`")

    if not results:
        st.info(tr("task_no_results"))
        return

    current_mode = record.get("pipeline", "demo_full")
    num_cols = 3
    for row_start in range(0, len(results), num_cols):
        cols = st.columns(num_cols)
        for col_idx in range(num_cols):
            result_idx = row_start + col_idx
            if result_idx < len(results):
                with cols[col_idx]:
                    display_candidate_result(
                        results[result_idx],
                        result_idx,
                        current_mode,
                        key_prefix=f"history_{selected_task_id}",
                    )


def render_skills_page():
    work_dir = Path(__file__).parent
    managed_dir = skill_library.get_managed_skills_dir(work_dir)
    installed_skills = skill_library.list_installed_skills(work_dir)

    st.markdown(f"### {tr('tab_skills')}")
    st.caption(tr("skills_intro"))

    metric_cols = st.columns([1, 3])
    metric_cols[0].metric(tr("skills_installed_count"), len(installed_skills))
    metric_cols[1].info(f"{tr('skills_root')}: `{managed_dir}`")

    st.divider()
    st.markdown(f"## {tr('skills_import_heading')}")
    import_col1, import_col2 = st.columns([3, 1])
    with import_col1:
        import_path = st.text_input(
            tr("skills_import_path"),
            value=st.session_state.get("skills_import_path", r"C:\Users\Administrator\Desktop\skills2"),
            help=tr("skills_import_path_help"),
            key="skills_import_path_input",
        )
    with import_col2:
        overwrite_skills = st.checkbox(
            tr("skills_overwrite"),
            value=True,
            key="skills_overwrite_checkbox",
        )

    if st.button(tr("skills_import_button"), use_container_width=True, key="skills_import_button_action"):
        try:
            imported = skill_library.import_skills_from_path(import_path, work_dir, overwrite=overwrite_skills)
            imported_count = sum(1 for item in imported if item.get("status") == "imported")
            skipped_count = sum(1 for item in imported if item.get("status") == "skipped")
            st.session_state["skills_import_path"] = import_path
            if imported_count:
                st.success(tr("skills_import_success", count=imported_count))
            if skipped_count:
                st.info(f"Skipped {skipped_count} existing skills.")
            if not imported_count and not skipped_count:
                st.info(tr("skills_import_empty"))
            st.rerun()
        except Exception as e:
            st.error(tr("skills_import_error", error=e))

    uploaded_skill_archives = st.file_uploader(
        tr("skills_upload_label"),
        type=["zip"],
        accept_multiple_files=True,
        help=tr("skills_upload_help"),
        key="skills_upload_archives",
    )
    if st.button(tr("skills_upload_button"), use_container_width=True, key="skills_upload_button_action"):
        try:
            if not uploaded_skill_archives:
                st.info(tr("skills_import_empty"))
            else:
                imported = skill_library.import_uploaded_archives(
                    uploaded_skill_archives,
                    work_dir,
                    overwrite=overwrite_skills,
                )
                imported_count = sum(1 for item in imported if item.get("status") == "imported")
                st.success(tr("skills_import_success", count=imported_count))
                st.rerun()
        except Exception as e:
            st.error(tr("skills_import_error", error=e))

    st.divider()
    st.markdown(f"## {tr('skills_delete_heading')}")
    if installed_skills:
        installed_names = [skill["name"] for skill in installed_skills]
        to_delete = st.multiselect(
            tr("skills_delete_select"),
            installed_names,
            key="skills_delete_multiselect",
        )
        if st.button(tr("skills_delete_button"), use_container_width=True, key="skills_delete_button_action"):
            try:
                deleted = skill_library.delete_installed_skills(work_dir, to_delete)
                deleted_count = sum(1 for item in deleted if item.get("status") == "deleted")
                st.success(tr("skills_delete_success", count=deleted_count))
                st.rerun()
            except Exception as e:
                st.error(tr("skills_import_error", error=e))
    else:
        st.info(tr("skills_none_installed"))

    st.divider()
    st.markdown(f"## {tr('skills_preview_heading')}")
    if not installed_skills:
        st.info(tr("skills_none_installed"))
        return

    preview_map = {skill["name"]: skill for skill in installed_skills}
    selected_skill_name = st.selectbox(
        tr("skills_preview_select"),
        options=list(preview_map.keys()),
        key="skills_preview_selectbox",
    )
    selected_skill = preview_map[selected_skill_name]
    st.markdown(f"**{tr('skills_description')}**")
    st.write(selected_skill.get("description", ""))
    st.markdown(f"**{tr('skills_location')}**")
    st.code(selected_skill["path"])
    with st.expander(tr("skills_preview_content"), expanded=False):
        st.code(selected_skill.get("preview", ""), language="markdown")

def main():
    refresh_model_config()
    st.session_state.setdefault("ui_language", "zh")
    lang_col1, lang_col2 = st.columns([5, 1])
    with lang_col2:
        st.selectbox(
            tr("language_label"),
            options=list(LANGUAGE_LABELS.keys()),
            format_func=lambda code: LANGUAGE_LABELS[code],
            key="ui_language",
        )
    lang = st.session_state.get("ui_language", "zh")
    st.title(tr("app_title"))
    st.markdown(tr("app_subtitle"))
    tab0 = st.container()
    
    # Create tabs
    guide_tab_label = "使用指南" if lang == "zh" else "Guide"
    tab_guide, tab1, tab2, tab3, tab4 = st.tabs(
        [guide_tab_label, tr("tab_generate"), tr("tab_refine"), tr("tab_history"), tr("tab_skills")]
    )
    
    with tab_guide:
        render_onboarding_page()
    
    with tab0:
        st.markdown(f"### {tr('config_heading')}")
        render_status_cards()

        current_cfg = ensure_config_sections(load_model_config())
        custom_detections = refresh_custom_endpoint_detections(current_cfg)
        text_detection = custom_detections["text"]
        image_detection = custom_detections["image"]
        default_text_provider, default_text_model = split_model_name(
            current_cfg["defaults"].get("main_model_name", ""),
            fallback_provider="gemini",
        )
        default_image_provider, default_image_model = split_model_name(
            current_cfg["defaults"].get("image_gen_model_name", ""),
            fallback_provider="gemini",
        )

        if st.session_state.pop("config_saved_notice_pending", False):
            st.success(tr("config_saved_notice"))

        with st.form("config_form"):
            col_a, col_b = st.columns(2)
            with col_a:
                text_provider = st.selectbox(
                    tr("main_model_provider"),
                    TEXT_PROVIDER_OPTIONS,
                    index=TEXT_PROVIDER_OPTIONS.index(default_text_provider),
                    format_func=lambda x: get_provider_label(x),
                )
                if text_provider == "custom" and text_detection.get("models"):
                    text_model_options = get_detected_model_options(text_detection, default_text_model)
                    text_model_selection = st.selectbox(
                        tr("detected_model_choice"),
                        text_model_options,
                        index=text_model_options.index(default_text_model)
                        if default_text_model in text_model_options
                        else 0,
                        key="config_text_detected_model_choice",
                        format_func=lambda value: tr("custom_option") if value == CUSTOM_OPTION else value,
                        help=tr("detected_model_help"),
                    )
                    if text_model_selection == CUSTOM_OPTION:
                        text_model_name = st.text_input(
                            tr("main_model_name"),
                            value=default_text_model if default_text_model not in text_detection.get("models", []) else "",
                            placeholder=get_provider_hint(text_provider, "text"),
                            help=tr("main_model_help"),
                        )
                    else:
                        text_model_name = text_model_selection
                        st.caption(f"{tr('main_model_name')}: `{text_model_name}`")
                else:
                    text_model_name = st.text_input(
                        tr("main_model_name"),
                        value=default_text_model,
                        placeholder=get_provider_hint(text_provider, "text"),
                        help=tr("main_model_help"),
                    )
                google_api_key = st.text_input(
                    tr("gemini_api_key"),
                    value=current_cfg["api_keys"].get("google_api_key", ""),
                    type="password",
                )
                anthropic_api_key = st.text_input(
                    tr("anthropic_api_key"),
                    value=current_cfg["api_keys"].get("anthropic_api_key", ""),
                    type="password",
                )

            with col_b:
                image_provider = st.selectbox(
                    tr("image_model_provider"),
                    IMAGE_PROVIDER_OPTIONS,
                    index=IMAGE_PROVIDER_OPTIONS.index(default_image_provider if default_image_provider in IMAGE_PROVIDER_OPTIONS else "gemini"),
                    format_func=lambda x: get_provider_label(x),
                )
                if image_provider == "custom" and image_detection.get("models"):
                    image_model_options = get_detected_model_options(image_detection, default_image_model)
                    image_model_selection = st.selectbox(
                        tr("detected_model_choice"),
                        image_model_options,
                        index=image_model_options.index(default_image_model)
                        if default_image_model in image_model_options
                        else 0,
                        key="config_image_detected_model_choice",
                        format_func=lambda value: tr("custom_option") if value == CUSTOM_OPTION else value,
                        help=tr("detected_model_help"),
                    )
                    if image_model_selection == CUSTOM_OPTION:
                        image_model_name = st.text_input(
                            tr("image_model_name"),
                            value=default_image_model if default_image_model not in image_detection.get("models", []) else "",
                            placeholder=get_provider_hint(image_provider, "image"),
                            help=tr("image_model_help"),
                        )
                    else:
                        image_model_name = image_model_selection
                        st.caption(f"{tr('image_model_name')}: `{image_model_name}`")
                else:
                    image_model_name = st.text_input(
                        tr("image_model_name"),
                        value=default_image_model,
                        placeholder=get_provider_hint(image_provider, "image"),
                        help=tr("image_model_help"),
                    )
                openai_api_key = st.text_input(
                    tr("openai_api_key"),
                    value=current_cfg["api_keys"].get("openai_api_key", ""),
                    type="password",
                )
                openrouter_api_key = st.text_input(
                    tr("openrouter_api_key"),
                    value=current_cfg["api_keys"].get("openrouter_api_key", ""),
                    type="password",
                )

            custom_col1, custom_col2 = st.columns(2)
            with custom_col1:
                st.markdown(f"#### {tr('custom_text_endpoint')}")
                custom_text_base_url = st.text_input(
                    tr("custom_text_base_url"),
                    value=current_cfg["api_base_urls"].get("custom_text_base_url", "") or current_cfg["api_base_urls"].get("custom_base_url", ""),
                    placeholder="https://api.example.com/v1",
                    help=tr("custom_text_base_url_help"),
                )
                custom_text_api_key = st.text_input(
                    tr("custom_text_api_key"),
                    value=current_cfg["api_keys"].get("custom_text_api_key", "") or current_cfg["api_keys"].get("custom_api_key", ""),
                    type="password",
                    help=tr("custom_text_api_key_help"),
                )
                render_endpoint_detection_summary(text_detection)
            with custom_col2:
                st.markdown(f"#### {tr('custom_image_endpoint')}")
                custom_image_base_url = st.text_input(
                    tr("custom_image_base_url"),
                    value=current_cfg["api_base_urls"].get("custom_image_base_url", "") or current_cfg["api_base_urls"].get("custom_base_url", ""),
                    placeholder="https://api.example.com/v1",
                    help=tr("custom_image_base_url_help"),
                )
                custom_image_api_key = st.text_input(
                    tr("custom_image_api_key"),
                    value=current_cfg["api_keys"].get("custom_image_api_key", "") or current_cfg["api_keys"].get("custom_api_key", ""),
                    type="password",
                    help=tr("custom_image_api_key_help"),
                )
                render_endpoint_detection_summary(image_detection)

            vertex_col1, vertex_col2 = st.columns(2)
            with vertex_col1:
                google_project_id = st.text_input(
                    tr("vertex_project"),
                    value=current_cfg["google_cloud"].get("project_id", ""),
                )
            with vertex_col2:
                google_location = st.text_input(
                    tr("vertex_location"),
                    value=current_cfg["google_cloud"].get("location", "global"),
                )

            save_clicked = st.form_submit_button(tr("save_configuration"), use_container_width=True)

        if save_clicked:
            text_model_name = str(text_model_name or "").strip()
            image_model_name = str(image_model_name or "").strip()
            if not text_model_name:
                text_model_name = default_text_model or get_provider_hint(text_provider, "text")
            if not image_model_name:
                image_model_name = default_image_model or get_provider_hint(image_provider, "image")
            custom_text_base_url = str(custom_text_base_url or "").strip()
            custom_image_base_url = str(custom_image_base_url or "").strip()
            custom_text_api_key = str(custom_text_api_key or "").strip()
            custom_image_api_key = str(custom_image_api_key or "").strip()

            current_cfg["defaults"]["main_model_name"] = compose_model_name(text_provider, text_model_name)
            current_cfg["defaults"]["image_gen_model_name"] = compose_model_name(image_provider, image_model_name)
            current_cfg["api_keys"]["google_api_key"] = google_api_key.strip()
            current_cfg["api_keys"]["anthropic_api_key"] = anthropic_api_key.strip()
            current_cfg["api_keys"]["openai_api_key"] = openai_api_key.strip()
            current_cfg["api_keys"]["openrouter_api_key"] = openrouter_api_key.strip()
            current_cfg["api_keys"]["custom_text_api_key"] = custom_text_api_key
            current_cfg["api_keys"]["custom_image_api_key"] = custom_image_api_key
            current_cfg["api_keys"]["custom_api_key"] = custom_text_api_key or custom_image_api_key
            current_cfg["api_base_urls"]["custom_text_base_url"] = custom_text_base_url
            current_cfg["api_base_urls"]["custom_image_base_url"] = custom_image_base_url
            current_cfg["api_base_urls"]["custom_base_url"] = custom_text_base_url or custom_image_base_url
            current_cfg["google_cloud"]["project_id"] = google_project_id.strip()
            current_cfg["google_cloud"]["location"] = google_location.strip() or "global"
            save_model_config(current_cfg)
            reload_generation_backends()
            refresh_custom_endpoint_detections(current_cfg, force=True)
            sync_generation_model_widget_state(force=True)
            st.session_state["config_saved_notice_pending"] = True
            st.rerun()

        st.caption(tr("custom_detection_hint"))
        st.info(tr("config_tip"))

    # ==================== TAB 1: Generate Candidates ====================
    with tab1:
        sync_generation_model_widget_state()
        st.markdown(f"### {tr('generate_intro')}")
        st.caption(
            tr(
                "defaults_from_config",
                text_model=get_config_val('defaults', 'main_model_name', 'MAIN_MODEL_NAME', ''),
                image_model=get_config_val('defaults', 'image_gen_model_name', 'IMAGE_GEN_MODEL_NAME', ''),
            )
        )
        live_task_container = st.empty()
        if st.session_state.get("active_task_state"):
            render_live_task_panel(st.session_state["active_task_state"], live_task_container)
        
        # Sidebar configuration for Tab 1
        with st.sidebar:
            st.title(tr("generation_settings"))
            
            exp_mode = st.selectbox(
                tr("pipeline_mode"),
                ["demo_full", "demo_planner_critic"],
                index=0,
                key="tab1_exp_mode",
                help=tr("pipeline_mode_help")
            )
            
            mode_info = {
                "demo_planner_critic": "Planner -> Visualizer -> Critic -> Visualizer" if lang == "en" else "Planner -> Visualizer -> Critic -> Visualizer",
                "demo_full": (
                    "Retriever -> Planner -> Stylist -> Visualizer -> Critic -> Visualizer. The stylist can make the diagram more aesthetically pleasing, but may oversimplify, so trying both modes is recommended."
                    if lang == "en"
                    else "Retriever -> Planner -> Stylist -> Visualizer -> Critic -> Visualizer。Stylist 会提升美观度，但有时会过度简化，建议两种模式都试一下。"
                )
            }
            st.info(mode_info[exp_mode])
            
            retrieval_setting = st.selectbox(
                tr("retrieval_setting"),
                ["auto", "manual", "random", "none"],
                index=0,
                key="tab1_retrieval_setting",
                help=tr("retrieval_setting_help")
            )
            
            num_candidates = st.number_input(
                tr("num_candidates"),
                min_value=1,
                max_value=20,
                value=10,
                key="tab1_num_candidates",
                help=tr("num_candidates_help")
            )
            
            aspect_ratio = st.selectbox(
                tr("aspect_ratio"),
                ["21:9", "16:9", "3:2"],
                key="tab1_aspect_ratio",
                help=tr("aspect_ratio_help")
            )

            generation_resolution = st.selectbox(
                tr("generation_resolution"),
                IMAGE_SIZE_OPTIONS,
                index=0,
                key="tab1_generation_resolution",
                help=tr("generation_resolution_help"),
            )

            figure_language = st.selectbox(
                tr("figure_language"),
                FIGURE_TEXT_LANGUAGE_OPTIONS,
                index=0 if lang == "zh" else 1,
                key="tab1_figure_language",
                format_func=format_figure_language,
                help=tr("figure_language_help"),
            )
            
            max_critic_rounds = st.number_input(
                tr("max_critic_rounds"),
                min_value=1,
                max_value=5,
                value=3,
                key="tab1_max_critic_rounds",
                help=tr("max_critic_rounds_help")
            )
            
            default_model = get_config_val("defaults", "main_model_name", "MAIN_MODEL_NAME", "gemini-3.1-pro-preview")
            text_model_presets = [default_model] if default_model else ["gemini-3.1-pro-preview"]
            if "gemini-3-flash-preview" not in text_model_presets:
                text_model_presets.append("gemini-3-flash-preview")
            if "gemini-3.1-pro-preview" not in text_model_presets:
                text_model_presets.insert(0, "gemini-3.1-pro-preview")
            text_model_presets.append(CUSTOM_OPTION)
            text_model_selection = st.selectbox(
                tr("main_model_name"),
                text_model_presets,
                index=0,
                key="tab1_model_name",
                help=tr("main_model_help"),
                format_func=lambda value: tr("custom_option") if value == CUSTOM_OPTION else value,
            )
            if text_model_selection == CUSTOM_OPTION:
                main_model_name = st.text_input(
                    tr("custom_main_model"),
                    value="",
                    key="tab1_main_model_name_custom",
                    placeholder="e.g., openrouter/google/gemini-3.1-pro"
                )
            else:
                main_model_name = text_model_selection

            default_image_model = get_config_val("defaults", "image_gen_model_name", "IMAGE_GEN_MODEL_NAME", "gemini-3.1-flash-image-preview")
            image_model_presets = [default_image_model] if default_image_model else ["gemini-3.1-flash-image-preview"]
            if "gemini-3-pro-image-preview" not in image_model_presets:
                image_model_presets.append("gemini-3-pro-image-preview")
            if "gemini-3.1-flash-image-preview" not in image_model_presets:
                image_model_presets.insert(0, "gemini-3.1-flash-image-preview")
            image_model_presets.append(CUSTOM_OPTION)
            image_model_selection = st.selectbox(
                tr("image_model_name"),
                image_model_presets,
                index=0,
                key="tab1_image_model_name",
                help=tr("image_model_help"),
                format_func=lambda value: tr("custom_option") if value == CUSTOM_OPTION else value,
            )
            if image_model_selection == CUSTOM_OPTION:
                image_gen_model_name = st.text_input(
                    tr("custom_image_model"),
                    value="",
                    key="tab1_image_gen_model_name_custom",
                    placeholder="e.g., openrouter/openai/gpt-image-1"
                )
            else:
                image_gen_model_name = image_model_selection

            st.caption(f"Text: `{main_model_name}`")
            st.caption(f"Image: `{image_gen_model_name}`")
        
        st.divider()
        
        # Input section
        st.markdown(f"## {tr('input_heading')}")
        
        # Example content
        example_method = r"""## Methodology: The PaperVizAgent Framework
        
        In this section, we present the architecture of PaperVizAgent, a reference-driven agentic framework for automated academic illustration. As illustrated in Figure \ref{fig:methodology_diagram}, PaperVizAgent orchestrates a collaborative team of five specialized agents—Retriever, Planner, Stylist, Visualizer, and Critic—to transform raw scientific content into publication-quality diagrams and plots. (See Appendix \ref{app_sec:agent_prompts} for prompts)

### Retriever Agent

Given the source context $S$ and the communicative intent $C$, the Retriever Agent identifies $N$ most relevant examples $\mathcal{E} = \{E_n\}_{n=1}^{N} \subset \mathcal{R}$ from the fixed reference set $\mathcal{R}$ to guide the downstream agents. As defined in Section \ref{sec:task_formulation}, each example $E_i \in \mathcal{R}$ is a triplet $(S_i, C_i, I_i)$.
To leverage the reasoning capabilities of VLMs, we adopt a generative retrieval approach where the VLM performs selection over candidate metadata:
$$
\mathcal{E} = \text{VLM}_{\text{Ret}} \left( S, C, \{ (S_i, C_i) \}_{E_i \in \mathcal{R}} \right)
$$
Specifically, the VLM is instructed to rank candidates by matching both research domain (e.g., Agent & Reasoning) and diagram type (e.g., pipeline, architecture), with visual structure being prioritized over topic similarity. By explicitly reasoned selection of reference illustrations $I_i$ whose corresponding contexts $(S_i, C_i)$ best match the current requirements, the Retriever provides a concrete foundation for both structural logic and visual style.

### Planner Agent

The Planner Agent serves as the cognitive core of the system. It takes the source context $S$, communicative intent $C$, and retrieved examples $\mathcal{E}$ as inputs. By performing in-context learning from the demonstrations in $\mathcal{E}$, the Planner translates the unstructured or structured data in $S$ into a comprehensive and detailed textual description $P$ of the target illustration:
$$
P = \text{VLM}_{\text{plan}}(S, C, \{ (S_i, C_i, I_i) \}_{E_i \in \mathcal{E}})
$$

### Stylist Agent

To ensure the output adheres to the aesthetic standards of modern academic manuscripts, the Stylist Agent acts as a design consultant.
A primary challenge lies in defining a comprehensive “academic style,” as manual definitions are often incomplete.
To address this, the Stylist traverses the entire reference collection $\mathcal{R}$ to automatically synthesize an *Aesthetic Guideline* $\mathcal{G}$ covering key dimensions such as color palette, shapes and containers, lines and arrows, layout and composition, and typography and icons (see Appendix \ref{app_sec:auto_summarized_style_guide} for the summarized guideline and implementation details). Armed with this guideline, the Stylist refines each initial description $P$ into a stylistically optimized version $P^*$:
$$
P^* = \text{VLM}_{\text{style}}(P, \mathcal{G})
$$
This ensures that the final illustration is not only accurate but also visually professional.

### Visualizer Agent

After receiving the stylistically optimized description $P^*$, the Visualizer Agent collaborates with the Critic Agent to render academic illustrations and iteratively refine their quality. The Visualizer Agent leverages an image generation model to transform textual descriptions into visual output. In each iteration $t$, given a description $P_t$, the Visualizer generates:
$$
I_t = \text{Image-Gen}(P_t)
$$
where the initial description $P_0$ is set to $P^*$.

### Critic Agent

The Critic Agent forms a closed-loop refinement mechanism with the Visualizer by closely examining the generated image $I_t$ and providing refined description $P_{t+1}$ to the Visualizer. Upon receiving the generated image $I_t$ at iteration $t$, the Critic inspects it against the original source context $(S, C)$ to identify factual misalignments, visual glitches, or areas for improvement. It then provides targeted feedback and produces a refined description $P_{t+1}$ that addresses the identified issues:
$$
P_{t+1} = \text{VLM}_{\text{critic}}(I_t, S, C, P_t)
$$
This revised description is then fed back to the Visualizer for regeneration. The Visualizer-Critic loop iterates for $T=3$ rounds, with the final output being $I = I_T$. This iterative refinement process ensures that the final illustration meets the high standards required for academic dissemination.

### Extension to Statistical Plots

The framework extends to statistical plots by adjusting the Visualizer and Critic agents. For numerical precision, the Visualizer converts the description $P_t$ into executable Python Matplotlib code: $I_t = \text{VLM}_{\text{code}}(P_t)$. The Critic evaluates the rendered plot and generates a refined description $P_{t+1}$ addressing inaccuracies or imperfections: $P_{t+1} = \text{VLM}_{\text{critic}}(I_t, S, C, P_t)$. The same $T=3$ round iterative refinement process applies. While we prioritize this code-based approach for accuracy, we also explore direct image generation in Section \ref{sec:discussion}. See Appendix \ref{app_sec:plot_agent_prompt} for adjusted prompts."""

        example_caption = "Figure 1: Overview of our PaperVizAgent framework. Given the source context and communicative intent, we first apply a Linear Planning Phase to retrieve relevant reference examples and synthesize a stylistically optimized description. We then use an Iterative Refinement Loop (consisting of Visualizer and Critic agents) to transform the description into visual output and conduct multi-round refinements to produce the final academic illustration."
        
        col_input1, col_input2 = st.columns([3, 2])
        
        with col_input1:
            # Example selector for method content
            method_example = st.selectbox(
                tr("load_example_method"),
                [EXAMPLE_NONE, EXAMPLE_PAPERVIZ],
                key="method_example_selector",
                format_func=lambda value: tr("example_none") if value == EXAMPLE_NONE else tr("example_paperviz")
            )
            
            # Set value based on example selection or session state
            if method_example == EXAMPLE_PAPERVIZ:
                method_value = example_method
            else:
                method_value = st.session_state.get("method_content", "")
            
            method_content = st.text_area(
                tr("method_content"),
                value=method_value,
                height=250,
                placeholder=tr("method_content_placeholder"),
                help=tr("method_content_help")
            )
        
        with col_input2:
            # Example selector for caption
            caption_example = st.selectbox(
                tr("load_example_caption"),
                [EXAMPLE_NONE, EXAMPLE_PAPERVIZ],
                key="caption_example_selector",
                format_func=lambda value: tr("example_none") if value == EXAMPLE_NONE else tr("example_paperviz")
            )
            
            # Set value based on example selection or session state
            if caption_example == EXAMPLE_PAPERVIZ:
                caption_value = example_caption
            else:
                caption_value = st.session_state.get("caption", "")
            
            caption = st.text_area(
                tr("figure_caption"),
                value=caption_value,
                height=250,
                placeholder=tr("figure_caption_placeholder"),
                help=tr("figure_caption_help")
            )
        
        # Process button
        effective_retrieval_setting, retrieval_warning = get_effective_retrieval_setting(retrieval_setting, "diagram")
        if retrieval_warning:
            st.warning(retrieval_warning)

        if st.button(tr("generate_button"), type="primary", use_container_width=True):
            if not method_content or not caption:
                st.error(tr("missing_content_error"))
            else:
                # Save to session state
                st.session_state["method_content"] = method_content
                st.session_state["caption"] = caption
                task_state = create_task_tracking_state(
                    method_content=method_content,
                    caption=caption,
                    exp_mode=exp_mode,
                    retrieval_setting=effective_retrieval_setting,
                    num_candidates=num_candidates,
                    aspect_ratio=aspect_ratio,
                    generation_resolution=generation_resolution,
                    figure_language=figure_language,
                    max_critic_rounds=max_critic_rounds,
                    main_model_name=main_model_name,
                    image_gen_model_name=image_gen_model_name,
                )
                st.session_state["active_task_state"] = task_state
                st.session_state["selected_history_task_id"] = task_state["task_id"]
                render_live_task_panel(task_state, live_task_container)

                def handle_task_event(event, data=None):
                    append_task_event(task_state, event, data)
                    st.session_state["active_task_state"] = task_state
                    render_live_task_panel(task_state, live_task_container)
                
                with st.spinner(tr("generating_spinner", count=num_candidates)):
                    # Create input data list
                    input_data_list = create_sample_inputs(
                        method_content=method_content,
                        caption=caption,
                        aspect_ratio=aspect_ratio,
                        image_size=generation_resolution,
                        figure_language=figure_language,
                        num_copies=num_candidates,
                        max_critic_rounds=max_critic_rounds
                    )
                    
                    # Process in parallel
                    try:
                        results = asyncio.run(process_parallel_candidates(
                            input_data_list,
                            exp_mode=exp_mode,
                            retrieval_setting=effective_retrieval_setting,
                            main_model_name=main_model_name,
                            image_gen_model_name=image_gen_model_name,
                            event_callback=handle_task_event,
                        ))
                        st.session_state["results"] = results
                        st.session_state["exp_mode"] = exp_mode
                        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.session_state["timestamp"] = timestamp_str
                        
                        # Save results to JSON file
                        try:
                            # Create results directory if it doesn't exist
                            results_dir = Path(__file__).parent / "results" / "demo"
                            results_dir.mkdir(parents=True, exist_ok=True)
                            
                            # Generate filename with timestamp
                            json_filename = results_dir / f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                            
                            # Save to JSON with proper encoding handling (like main.py)
                            with open(json_filename, "w", encoding="utf-8", errors="surrogateescape") as f:
                                json_string = json.dumps(results, ensure_ascii=False, indent=4)
                                # Clean invalid UTF-8 characters
                                json_string = json_string.encode("utf-8", "ignore").decode("utf-8")
                                f.write(json_string)
                            
                            st.session_state["json_file"] = str(json_filename)
                            finalize_task_tracking(
                                task_state,
                                status="completed",
                                results=results,
                                export_json_file=str(json_filename),
                            )
                            st.session_state["active_task_state"] = task_state
                            render_live_task_panel(task_state, live_task_container)
                            st.success(tr("success_generated", count=len(results)))
                            st.info(tr("results_saved_to", name=json_filename.name))
                        except Exception as e:
                            finalize_task_tracking(
                                task_state,
                                status="completed",
                                results=results,
                                error=f"Export warning: {e}",
                            )
                            st.session_state["active_task_state"] = task_state
                            render_live_task_panel(task_state, live_task_container)
                            st.warning(tr("save_failed_warning", count=len(results), error=e))
                    except Exception as e:
                        finalize_task_tracking(
                            task_state,
                            status="failed",
                            error=str(e),
                        )
                        st.session_state["active_task_state"] = task_state
                        render_live_task_panel(task_state, live_task_container)
                        st.error(tr("processing_error", error=e))
                        import traceback
                        st.code(traceback.format_exc())
        
        # Display results
        if "results" in st.session_state and st.session_state["results"]:
            results = st.session_state["results"]
            current_mode = st.session_state.get("exp_mode", exp_mode)
            timestamp = st.session_state.get("timestamp", "N/A")
            generated_count = len(results)
            ready_count = sum(
                1
                for result in results
                if get_final_image_info(result, current_mode)[0] in result
                and result[get_final_image_info(result, current_mode)[0]]
            )
            
            st.divider()
            st.markdown(f"## {tr('generated_candidates')}")
            st.caption(tr("generated_at", timestamp=timestamp, pipeline=mode_info.get(current_mode, current_mode)))
            metric_cols = st.columns(3)
            metric_cols[0].metric(tr("candidates_metric"), generated_count)
            metric_cols[1].metric(tr("images_ready"), ready_count)
            metric_cols[2].metric(tr("image_model_metric"), image_gen_model_name or "default")
            
            # Show JSON file download if available
            if "json_file" in st.session_state:
                json_file_path = Path(st.session_state["json_file"])
                if json_file_path.exists():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.info(tr("results_saved_to", name=json_file_path.relative_to(Path.cwd())))
                    with col2:
                        with open(json_file_path, "r", encoding="utf-8") as f:
                            json_data = f.read()
                        st.download_button(
                            label=tr("download_json"),
                            data=json_data,
                            file_name=json_file_path.name,
                            mime="application/json",
                            use_container_width=True
                        )
            
            # Display results in a grid (3 columns)
            num_cols = 3
            num_results = len(results)
            
            for row_start in range(0, num_results, num_cols):
                cols = st.columns(num_cols)
                for col_idx in range(num_cols):
                    result_idx = row_start + col_idx
                    if result_idx < num_results:
                        with cols[col_idx]:
                            display_candidate_result(results[result_idx], result_idx, current_mode)
            
            # Add ZIP download button
            st.divider()
            st.markdown(f"### {tr('batch_download')}")
            
            try:
                import zipfile
                
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    task_name = "diagram"
                    
                    for candidate_id, result in enumerate(results):
                        
                        # Find the final image key (same logic as display)
                        final_image_key = None
                        
                        # Try to find the last critic round
                        for round_idx in range(3, -1, -1):
                            image_key = f"target_{task_name}_critic_desc{round_idx}_base64_jpg"
                            if image_key in result and result[image_key]:
                                final_image_key = image_key
                                break
                        
                        # Fallback if no critic rounds completed
                        if not final_image_key:
                            if current_mode == "demo_full":
                                final_image_key = f"target_{task_name}_stylist_desc0_base64_jpg"
                            else:
                                final_image_key = f"target_{task_name}_desc0_base64_jpg"
                        
                        if final_image_key and final_image_key in result:
                            img = base64_to_image(result[final_image_key])
                            if img:
                                img_buffer = BytesIO()
                                img.save(img_buffer, format="PNG")
                                zip_file.writestr(
                                    f"candidate_{candidate_id}.png",
                                    img_buffer.getvalue()
                                )
                
                zip_buffer.seek(0)
                st.download_button(
                    label=tr("download_zip"),
                    data=zip_buffer.getvalue(),
                    file_name=f"papervizagent_candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                    use_container_width=True
                )
                st.success(tr("zip_ready"))
            except Exception as e:
                st.error(tr("zip_failed", error=e))
    
    # ==================== TAB 2: Refine Image ====================
    with tab2:
        st.markdown(f"### {tr('refine_intro')}")
        st.caption(tr("refine_caption"))
        
        # Sidebar for refinement settings
        with st.sidebar:
            st.title(tr("refinement_settings"))
            
            refine_resolution = st.selectbox(
                tr("target_resolution"),
                ["2K", "4K"],
                index=0,
                key="refine_resolution",
                help=tr("target_resolution_help")
            )
            
            refine_aspect_ratio = st.selectbox(
                tr("aspect_ratio"),
                ["21:9", "16:9", "3:2"],
                index=0,
                key="refine_aspect_ratio",
                help=tr("aspect_ratio_help")
            )
        
        st.divider()
        
        # Upload section
        st.markdown(f"## {tr('upload_heading')}")
        source_options = [IMAGE_SOURCE_UPLOAD]
        if "results" in st.session_state and st.session_state["results"]:
            source_options.append(IMAGE_SOURCE_GENERATED)
        source_mode = st.radio(
            tr("image_source"),
            source_options,
            horizontal=True,
            format_func=lambda value: tr("upload_image_option") if value == IMAGE_SOURCE_UPLOAD else tr("generated_image_option"),
        )

        uploaded_file = None
        selected_generated_image = None
        if source_mode == IMAGE_SOURCE_UPLOAD:
            uploaded_file = st.file_uploader(
                tr("choose_file"),
                type=["png", "jpg", "jpeg"],
                help=tr("choose_file_help")
            )
        else:
            candidate_options = list(range(len(st.session_state["results"])))
            selected_candidate_id = st.selectbox(
                tr("choose_candidate"),
                candidate_options,
                format_func=lambda x: tr("candidate_label", candidate_id=x),
            )
            selected_result = st.session_state["results"][selected_candidate_id]
            final_image_key, _ = get_final_image_info(
                selected_result,
                st.session_state.get("exp_mode", "demo_full"),
            )
            if final_image_key in selected_result:
                selected_generated_image = base64_to_image(selected_result[final_image_key])
                if selected_generated_image is not None:
                    st.image(selected_generated_image, use_container_width=True, caption=tr("candidate_label", candidate_id=selected_candidate_id))
        
        if uploaded_file is not None or selected_generated_image is not None:
            # Display uploaded image
            uploaded_image = Image.open(uploaded_file) if uploaded_file is not None else selected_generated_image
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"### {tr('original_image')}")
                st.image(uploaded_image, use_container_width=True)
            
            with col2:
                st.markdown(f"### {tr('edit_instructions')}")
                edit_prompt = st.text_area(
                    tr("describe_changes"),
                    height=200,
                    placeholder=tr("edit_placeholder"),
                    help=tr("edit_help"),
                    key="edit_prompt"
                )
                
                if st.button(tr("refine_button"), type="primary", use_container_width=True):
                    if not edit_prompt:
                        st.error(tr("edit_required"))
                    else:
                        with st.spinner(tr("refining_spinner", resolution=refine_resolution)):
                            try:
                                # Convert PIL image to bytes
                                img_byte_arr = BytesIO()
                                uploaded_image.save(img_byte_arr, format='JPEG')
                                image_bytes = img_byte_arr.getvalue()
                                
                                # Call nanoviz API
                                refined_bytes, message = asyncio.run(
                                    refine_image_with_nanoviz(
                                        image_bytes=image_bytes,
                                        edit_prompt=edit_prompt,
                                        aspect_ratio=refine_aspect_ratio,
                                        image_size=refine_resolution
                                    )
                                )
                                
                                if refined_bytes:
                                    st.session_state["refined_image"] = refined_bytes
                                    st.session_state["refine_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    st.success(message)
                                    st.rerun()
                                else:
                                    st.error(message)
                            except Exception as e:
                                st.error(tr("refine_error", error=e))
                                import traceback
                                st.code(traceback.format_exc())
            
            # Display refined result if available
            if "refined_image" in st.session_state:
                st.divider()
                st.markdown(f"## {tr('refined_result')}")
                st.caption(tr("refined_generated_at", timestamp=st.session_state.get('refine_timestamp', 'N/A'), resolution=refine_resolution))
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"### {tr('before')}")
                    st.image(uploaded_image, use_container_width=True)
                
                with col2:
                    st.markdown(f"### {tr('after', resolution=refine_resolution)}")
                    refined_image = Image.open(BytesIO(st.session_state["refined_image"]))
                    st.image(refined_image, use_container_width=True)
                    
                    # Download button
                    st.download_button(
                        label=tr("download_refined", resolution=refine_resolution),
                        data=st.session_state["refined_image"],
                        file_name=f"refined_{refine_resolution}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png",
                        use_container_width=True
                    )

    with tab3:
        render_history_page()

    with tab4:
        render_skills_page()

if __name__ == "__main__":
    main()
