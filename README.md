# PaperBanana Studio

面向大众、可视化、可自定义 API 的学术插图生成工作台。

> Fork Notice
>
> This repository is an independent, community-enhanced derivative of [dwzhu-pku/PaperBanana](https://github.com/dwzhu-pku/PaperBanana), which itself builds on Google's [PaperVizAgent](https://github.com/google-research/papervizagent). This project is not the official upstream release. It is a downstream enhancement focused on usability, custom API integration, UI experience, local deployment, and public accessibility.

本项目基于原始开源项目 [dwzhu-pku/PaperBanana](https://github.com/dwzhu-pku/PaperBanana) 演进而来，而上游又与 Google Research 的 [PaperVizAgent](https://github.com/google-research/papervizagent) 一脉相承。这个版本的目标不是只做论文复现，而是把它真正打磨成一个普通用户也能直接上手的本地 Web 应用。

![PaperBanana Examples](assets/teaser_figure.jpg)

## 这和原版相比，有哪些重磅更新

### 1. 不再局限于单一官方接口
- 支持 `custom/<model>` 路由。
- 文本模型和图片模型可以分别配置不同的 API URL、API Key。
- 支持 OpenAI-compatible 接口。
- 支持在界面里自动检测 API 是否可用，并自动拉取可选模型列表。

### 2. 更适合真实使用的 Web 界面
- 增加中英文界面切换。
- 增加新手介绍页，解释项目是什么、怎么用、背后流程和常见报错。
- 增加历史任务页，可以直接回看过去任务的参数、结果和中间过程。
- 增加运行中实时进度展示，可查看阶段进展、提示词和中间产物。

### 3. 生成流程更完整
- 候选图生成阶段就可以直接选择分辨率，而不只是精修阶段。
- 可以指定图内文字语言为中文或英文，并把要求注入到内部提示词。
- 精修链路支持更高分辨率输出。

### 4. 更强的可扩展能力
- 增加 Skills 管理页。
- 可以通过本地目录或 ZIP 包导入自定义 skills。
- 可以在界面里删除已安装 skills。
- 支持预览技能说明文件，便于扩展工作流。

### 5. 更稳的本地运行体验
- 修复了 Windows 下 `gbk` 编码导致的启动和日志崩溃问题。
- 对缺失 `PaperBananaBench` 参考数据时做了自动降级，不再直接报错退出。
- 增加 Windows 启停脚本。
- 提供便携版 EXE 打包方案，双击即可启动本地网页。

## 适合谁用

- 需要给论文、报告、项目方案生成学术流程图、系统图、方法图的人
- 希望把自己的模型接口接进来，而不是被固定供应商绑定的人
- 想要一个本地可视化操作台，而不是只会跑脚本的人
- 想持续沉淀历史任务、技能和输出结果的人

## 核心能力

- 学术图候选生成
- 多 agent 协同规划、风格约束、出图、批评迭代
- 图片精修和高分辨率导出
- 中文 / 英文图内文字控制
- 历史任务管理
- 实时进度查看
- Skills 导入 / 删除 / 预览
- 自定义 API 接入和模型探测

## 工作流程

```mermaid
flowchart LR
    A["输入论文方法描述 / 图注"] --> B["Retriever 检索参考样例"]
    B --> C["Planner 规划图像语义和结构"]
    C --> D["Stylist 约束学术风格"]
    D --> E["Visualizer 调用图片模型生成候选图"]
    E --> F["Critic 评审并给出修改建议"]
    F --> G["多轮迭代优化"]
    G --> H["候选结果保存 / 下载"]
    H --> I["Polish 精修与高分辨率导出"]
```

如果本地没有 `PaperBananaBench`，系统会自动切换到无参考样例模式继续运行，只是 few-shot 检索增强能力会减弱。

## 界面总览

当前 Web 应用主要包含这些页面：

- `使用指南`
  解释项目定位、上手步骤、工作原理、日志位置和常见问题。
- `生成候选图`
  输入论文内容、图注和生成参数，运行多 agent 出图流程。
- `精修图片`
  对已有图片继续优化、修改和放大。
- `历史任务`
  查看过去任务的参数、过程记录、结果图和下载文件。
- `Skills 管理`
  导入、删除、预览 skills。

## 快速开始

### 方式一：从源码运行

#### 1. 克隆项目

```bash
git clone https://github.com/damchungtien-lab/paperbanana-studio.git
cd paperbanana-studio
```

#### 2. 创建虚拟环境并安装依赖

推荐 Python 3.12：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

#### 3. 配置模型

复制模板配置：

```bash
copy configs\model_config.template.yaml configs\model_config.yaml
```

然后编辑 `configs/model_config.yaml`。

注意：
- `configs/model_config.yaml` 是本地私有文件，不应提交到 Git。
- 模板文件 `configs/model_config.template.yaml` 不包含你的真实密钥。

#### 4. 启动

```bash
streamlit run demo.py
```

Windows 也可以直接用：

```bash
start_paperbanana.cmd
```

停止：

```bash
stop_paperbanana.cmd
```

### 方式二：打包为 Windows 便携版 EXE

仓库内提供了打包脚本：

```bash
.venv\Scripts\python.exe scripts\build_portable_exe.py
```

打包完成后，产物位于：

```text
dist/PaperBanana.exe
```

便携版的设计原则：
- EXE 内置的是脱敏模板配置，不会打包你的真实 API 信息。
- 首次启动会把运行文件解包到当前用户本机目录。
- 服务就绪后会自动打开浏览器。

## 如何使用

### 第一步：配置 API

打开网页后，先进入顶部的 API 设置区域。

你可以分别配置：

- 文本模型 provider / model
- 图片模型 provider / model
- 文本模型专用 API URL / Key
- 图片模型专用 API URL / Key

支持的常见路由形式：

- `gemini/<model>`
- `openrouter/<model>`
- `openai/<model>`
- `anthropic/<model>`
- `custom/<model>`

其中 `custom/<model>` 适合你自己的 OpenAI-compatible 接口。

配置后可以直接在界面里做：

- API 有效性检查
- 模型列表自动探测
- 保存默认配置

### 第二步：生成候选图

进入 `生成候选图` 页签后：

1. 粘贴方法描述或论文片段
2. 填写图注
3. 选择图类型和运行模式
4. 选择候选数量、critic 轮数、长宽比、分辨率
5. 指定图内文字语言为中文或英文
6. 点击生成

运行过程中，你可以在前端直接看到：

- 当前阶段
- 关键提示词
- 中间产物
- 最终候选图

### 第三步：精修

进入 `精修图片` 页签后：

1. 上传已有图片，或直接复用刚生成的候选图
2. 描述你想修改的地方
3. 选择目标分辨率和长宽比
4. 导出精修结果

### 第四步：查看历史任务

进入 `历史任务` 页签后，可以查看：

- 每个任务的时间和参数
- 每轮中间记录
- 生成结果和下载产物
- 失败任务的报错信息

### 第五步：管理 Skills

进入 `Skills 管理` 页签后，可以：

- 从本地目录导入 skills
- 上传 ZIP 包导入 skills
- 删除已安装 skills
- 预览 `SKILL.md`

## PaperBananaBench 是什么

`PaperBananaBench` 是这个项目配套的参考数据集 / 基准集，主要用于：

- Retriever few-shot 检索参考
- Planner 的参考引导
- 评测和对比

它不是强制依赖，但通常有它效果更稳。没有它时，系统会自动降级继续运行。

推荐放置路径：

```text
data/PaperBananaBench/
```

数据集地址：

- [PaperBananaBench on Hugging Face](https://huggingface.co/datasets/dwzhu/PaperBananaBench)

## 日志、历史和输出位置

常见目录如下：

- `configs/model_config.yaml`
  本地私有模型配置
- `results/`
  结果图、历史任务和导出文件
- `logs/`
  运行日志

如果使用便携版 EXE，运行数据会写入当前用户本机的应用目录，而不是仓库源码目录。

## 常见问题

### 1. 我用的是自定义 API，为什么还提示缺 Gemini Key？

说明某个流程节点仍在使用旧模型选择状态。当前版本已经修复了这个同步问题，生成页会强制以保存后的默认配置为准。

### 2. 为什么 Windows 会出现编码报错？

旧版本里某些日志和配置读取在 Windows 本地编码下会失败。这个版本已经统一修复为更稳的 UTF-8 / 安全日志输出。

### 3. 为什么选择了 4K，结果还是 1K？

原始通用调用方式并不能正确把分辨率传给部分图片接口。当前版本已经针对 12ai 的 Gemini 图片协议做了适配，`4K` 会真正写入图片生成请求。

### 4. 没有数据集能不能用？

可以。只是检索增强能力会弱一些，系统会自动切换到无参考模式。

## 与原版仓库的关系

本仓库不是对原版的简单镜像，而是一个面向实际使用场景的增强版工作台，重点补齐了：

- API 自定义接入
- UI 易用性
- 历史任务和进度可视化
- Skills 扩展能力
- Windows 使用和分发体验

如果你想看原始研究版本，请参考：

- [dwzhu-pku/PaperBanana](https://github.com/dwzhu-pku/PaperBanana)
- [google-research/papervizagent](https://github.com/google-research/papervizagent)

请在任何再分发、介绍、截图展示或二次修改中，明确说明本项目来自上述上游开源仓库，并在此基础上做了增强和工程化改造。

## 开源说明

- 本仓库保留对原始开源项目的致谢与引用。
- 请遵循仓库内已有许可证文件。
- 不要把你自己的 `configs/model_config.yaml`、真实密钥、打包产物和本地运行结果提交到公开仓库。

## 致谢

感谢原始 PaperBanana / PaperVizAgent 作者团队提供研究基础，也感谢所有把它推进到更易用、更开放版本的贡献者。
