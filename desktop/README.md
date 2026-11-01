# 桌面版 GUI（pywebview + React + Ant Design）

以窗口应用形式使用本项目的全部功能；命令行用法完全保留。

## 功能对照

| 功能 | CLI 原方式 | 桌面版 |
|------|-----------|--------|
| 转录/翻译（CPU/GPU/低显存/批处理） | 拖放 .bat | 任务页选择参数后点"开始" |
| 选择输出目录 / 覆盖 / 字幕格式 | 改 .bat 参数 | 任务页表单 |
| VAD / 合并 / 智能切分参数 | 改 generation_config.json5 | 任务页高级面板（即时覆盖）或生成参数页 |
| 下载模型（VAD/whisper-base/HF 模型） | download_models.py | 模型管理页（带进度日志） |
| 模型查看/选择 | 手动翻 models 目录 | 模型管理页表格 |
| Modal 云端推理 | python modal_infer.py（questionary 交互） | 云端推理页（表单，非交互） |
| generation_config.json5 编辑 | 文本编辑器 | 生成参数页（保留注释原样保存） |
| 环境诊断 | --console / 报错排查 | 环境诊断页（一键深度诊断） |

## 开发模式

```bash
# 终端 1：前端
cd frontend
pnpm install
pnpm dev          # http://localhost:5173

# 终端 2：桌面窗口（加载 Vite 开发服务器，支持热更新）
python desktop/app.py --dev --debug
```

## 打包发布

```bash
# 1. 构建 CLI 版（原有流程，得到含 infer.exe 的目录）
pyinstaller project.spec --noconfirm
# 产物: dist/faster_whisper_transwithai_chickenrice/

# 2. 构建 GUI 并合并（自动先构建前端）
python desktop/build_desktop.py --merge dist/faster_whisper_transwithai_chickenrice
# GUI exe 会拷贝进 CLI 目录，最终一个目录既是 CLI 又是桌面应用
```

仅想单独出 GUI 产物：

```bash
python desktop/build_desktop.py
# 产物: dist/FasterWhisperTransWithAI-GUI/
# 注意：该目录不含 infer.exe，运行任务时会提示缺失，请用上面的合并方式
```

## 结构

```
desktop/
  app.py            pywebview 入口 + JsApi（后端 API 实现）
  modal_cli.py      Modal 云端推理非交互 CLI（复用 modal_infer.py 流水线）
  desktop.spec      GUI PyInstaller 配置
  build_desktop.py  一键构建脚本
  API_CONTRACT.md   前后端 API 契约文档
frontend/
  src/api.ts        pywebview API 封装 + 事件订阅
  src/pages/        任务 / 模型 / 云端 / 参数 / 诊断 / 关于
```

## 架构说明

- GUI 主程序（Python）通过 `js_api` 暴露方法给前端；推理、下载、Modal 均以**子进程**执行，
  避免阻塞 UI，也可随时终止。子进程日志按行推送回前端（`window.__onEvent`），
  并解析中英文进度行驱动进度条。
- 打包后 GUI 与 `infer.exe` 同目录；GUI 调用 infer.exe，二者共享 models/、generation_config.json5。
- 开发模式下 GUI 直接调用 `python infer.py`，无需打包即可完整调试。
