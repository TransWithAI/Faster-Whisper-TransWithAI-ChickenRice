# 🎙️ Faster Whisper TransWithAI ChickenRice

[![GitHub Release](https://img.shields.io/github/v/release/TransWithAI/Faster-Whisper-TransWithAI-ChickenRice)](https://github.com/TransWithAI/Faster-Whisper-TransWithAI-ChickenRice/releases)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

高性能音视频转录和翻译工具 - 基于 Faster Whisper 和音声优化 VAD 的日文转中文优化版本

High-performance audio/video transcription and translation tool - Japanese-to-Chinese optimized version based on Faster Whisper and voice-optimized VAD

## ⚠️ 重要声明 / Important Notice

> **本软件为开源软件 / This software is open source**
>
> 🔗 **开源地址 / Repository**: https://github.com/TransWithAI/Faster-Whisper-TransWithAI-ChickenRice
>
> 👥 **开发团队 / Development Team**: AI汉化组 (https://t.me/transWithAI)
>
> 本软件完全免费开源 / This software is completely free and open source

## 🙏 致谢 / Acknowledgments

- 🚀 基于 [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) 开发
- 🐔 使用 [chickenrice0721/whisper-large-v2-translate-zh-v0.2-st-ct2](https://huggingface.co/chickenrice0721/whisper-large-v2-translate-zh-v0.2-st-ct2) 日文转中文优化模型
- 📝 可使用 [TransWithAI/whisper-ja-1.5B-ct2](https://huggingface.co/TransWithAI/whisper-ja-1.5B-ct2) 日文原文转录 bf16 模型，由 [efwkjn/whisper-ja-1.5B](https://huggingface.co/efwkjn/whisper-ja-1.5B) 原版模型转换而来
- 🔊 使用 [TransWithAI/Whisper-Vad-EncDec-ASMR-onnx](https://huggingface.co/TransWithAI/Whisper-Vad-EncDec-ASMR-onnx) 音声优化 VAD 模型
- ☁️ 感谢 [@Randomless](https://github.com/Randomless) 贡献 Modal 云端推理功能
- 💪 **感谢某匿名群友的算力和技术支持**

## ✨ 功能特性 / Features

- 🎯 **高精度日文转中文翻译**: 基于5000小时音频数据训练的"海南鸡v2"日文转中文优化模型
- 🚀 **GPU加速**: 支持 NVIDIA CUDA 11.8/12.2/12.8 及 AMD ROCm/HIP（RDNA1–RDNA4）
- ☁️ **云端推理**: 支持 Modal 云端 GPU 推理，无本地显卡也能使用
- 📝 **多格式输出**: 支持SRT、VTT、LRC等多种字幕格式
- 🎬 **音视频支持**: 支持常见音频(mp3/wav/flac等)和视频格式(mp4/mkv/avi等)
- 💾 **智能缓存**: 自动跳过已处理文件，提高批量处理效率
- 🔧 **灵活配置**: 可自定义转录参数，满足不同场景需求

## 📦 版本说明 / Package Variants

### 无主模型版 (No-Model Package)
- ✅ 所有 GPU 依赖项
- ✅ 音声优化 VAD（语音活动检测）模型
- ✅ whisper-base 特征提取文件（离线使用）
- ✅ 翻译和转录两套 `.bat` 启动脚本
- ❌ 不含主 Whisper 模型（需自行下载或替换）

默认 `.bat` 会从 `models/` 根目录加载主模型；如果模型保存在 `models/模型文件夹/`，请给对应 `.bat` 添加 `--model_name_or_path="models\模型文件夹"`。

### 翻译版 (Translate Package)
- ✅ 所有 GPU 依赖项
- ✅ 音声优化 VAD（语音活动检测）模型
- ✅ whisper-base 特征提取文件（离线使用）
- ✅ **"海南鸡v2 5000小时"** 日文转中文优化模型（开箱即用）
- ✅ 仅包含翻译 `.bat` 启动脚本

### 转录版 (Transcribe Package)
- ✅ 所有 GPU 依赖项
- ✅ 音声优化 VAD（语音活动检测）模型
- ✅ whisper-base 特征提取文件（离线使用）
- ✅ **TransWithAI/whisper-ja-1.5B-ct2** 日文原文转录 bf16 模型（开箱即用）
- ✅ 仅包含转录 `.bat` 启动脚本

## 🚀 快速开始 / Quick Start

### 1. 选择适合的CUDA版本 / Choose CUDA Version

运行 `nvidia-smi` 查看您的CUDA版本：

| 显卡系列 | 推荐 CUDA 版本 |
|---------|--------------|
| GTX 10/16系列 | CUDA 11.8 |
| RTX 20/30系列 | CUDA 11.8 或 12.2 |
| RTX 40系列 | CUDA 12.2 或 12.8 |
| RTX 50系列 | **必须使用 CUDA 12.8** |

### AMD 显卡（ROCm/HIP）/ AMD GPU (ROCm/HIP)

AMD 显卡用户（Windows）：请下载带有 `gfx***` 后缀的版本，每个 ZIP 对应一类 `gfx` 架构。AMD 版本已内置 ROCm/HIP 运行时 DLL，一般无需单独安装 ROCm。

For AMD GPU users (Windows): download the ZIP matching your GPU's `gfx` family. ROCm/HIP runtime DLLs are bundled — no separate ROCm install needed in most cases.

| 显卡 / GPU | 下载后缀 / Suffix | GFX 架构 / Architecture |
|---|---|---|
| RX 5000 / RDNA1 | `gfx101x_dgpu` | gfx1010 / gfx1011 / gfx1012 |
| RX 6000 / RDNA2 | `gfx103x_dgpu` | gfx1030 / gfx1031 / gfx1032 / gfx1034 |
| RX 7000 / RDNA3 | `gfx110x_all` | gfx1100 / gfx1101 / gfx1102（部分 iGPU 为 gfx1103） |
| RX 9000 / RDNA4 | `gfx120x_all` | gfx1200 / gfx1201 |

> **不知道自己的显卡型号？/ Don't know your GPU model?**
> Windows：打开 “任务管理器 -> 性能 -> GPU” 或 “设备管理器 -> 显示适配器”
> Windows: open “Task Manager -> Performance -> GPU” or “Device Manager -> Display adapters”

**快速自查 / Quick Self-Check**

| 显卡型号 / GPU Model | 下载后缀 / Download Suffix |
|---|---|
| RX 5300 / RX 5500 / RX 5600 / RX 5700 系列 | `gfx101x_dgpu` |
| RX 6400 / RX 6500 XT / RX 6600 / RX 6700 / RX 6800 / RX 6900 系列 | `gfx103x_dgpu` |
| RX 7600 / RX 7700 XT / RX 7800 XT / RX 7900 系列 | `gfx110x_all` |
| RX 9060 / RX 9060 XT / RX 9070（含 GRE/XT） | `gfx120x_all` |
| iGPU: Radeon 890M / 8060S / 860M (gfx115x) | ⚠️ 暂不支持 / Not yet supported |

> iGPU 用户请使用 CPU 版或 Modal 云端推理 / iGPU users: use the CPU build or Modal cloud inference

<details>
<summary><b>完整型号列表（按系列）/ Full model lists (by series)</b></summary>

- **RX 5000 (RDNA1)** -> `gfx101x_dgpu`
  - Desktop: RX 5300, RX 5300 XT, RX 5500, RX 5500 XT, RX 5600, RX 5600 XT, RX 5700, RX 5700 XT (incl. 50th Anniversary Edition)
  - Mobile dGPU: RX 5300M, RX 5500M, RX 5600M, RX 5700M

- **RX 6000 (RDNA2)** -> `gfx103x_dgpu`
  - Desktop: RX 6300 (OEM), RX 6400, RX 6500 XT, RX 6600, RX 6600 XT, RX 6650 XT, RX 6700, RX 6700 XT, RX 6750 GRE, RX 6750 XT, RX 6800, RX 6800 XT, RX 6900 XT, RX 6950 XT
  - Mobile dGPU: RX 6300M, RX 6450M, RX 6500M, RX 6550S, RX 6550M, RX 6600S, RX 6600M, RX 6650M, RX 6650M XT, RX 6700S, RX 6700M, RX 6800S, RX 6800M, RX 6850M XT

- **RX 7000 (RDNA3)** -> `gfx110x_all`
  - Desktop: RX 7400 (OEM), RX 7600, RX 7600 XT, RX 7650 GRE, RX 7700 (OEM), RX 7700 XT, RX 7800 XT, RX 7900 GRE, RX 7900 XT, RX 7900 XTX
  - Mobile dGPU: RX 7600S, RX 7600M XT, RX 7700S, RX 7800M, RX 7900M

- **RX 9000 (RDNA4)** -> `gfx120x_all`
  - Desktop: RX 9060, RX 9060 XT, RX 9070 GRE, RX 9070, RX 9070 XT

</details>

使用方式与 NVIDIA 版本相同，运行 `运行(翻译)(GPU).bat` 或 `运行(转录)(GPU).bat` 即可（内部使用 `--device=cuda`，这是 CTranslate2 HIP 后端的约定）。命令行也可使用 `--device=amd`（等同于 `--device=cuda`）。

Usage is the same as NVIDIA builds — just run `运行(翻译)(GPU).bat` or `运行(转录)(GPU).bat` (internally uses `--device=cuda`, which is CTranslate2's HIP convention). CLI also accepts `--device=amd` (alias for `--device=cuda`).

### 2. 下载对应版本 / Download

从 [Releases](https://github.com/TransWithAI/Faster-Whisper-TransWithAI-ChickenRice/releases) 页面下载对应版本

### 3. 使用方法 / Usage

将音视频文件拖放到相应的批处理文件：

```bash
# 翻译 GPU模式（推荐，显存≥6GB）
运行(翻译)(GPU).bat

# 翻译 GPU低显存模式（int8_float16，显存4GB）
运行(翻译)(GPU,低显存模式).bat

# 翻译 CPU模式（无显卡用户）
运行(翻译)(CPU).bat

# 转录 GPU模式
运行(转录)(GPU).bat

# 转录 CPU模式
运行(转录)(CPU).bat
```

## ☁️ Modal 云端推理 / Cloud Inference

无本地 GPU 或显存不足？使用 Modal 云端 GPU 进行推理：

### 1. 环境配置

```bash
# 使用现有 Conda 环境（已包含 modal 支持）
conda activate faster-whisper-cu118  # 或 cu122, cu128

# 或手动安装
pip install modal questionary
```

### 2. Modal 账号设置

```bash
# 注册账号：https://modal.com/（新用户每月 $30 免费额度）
# 配置 Token
modal token new
```

### 3. 运行云端推理

```bash
# 使用打包版本
modal_infer.exe

# 或使用 Python
python modal_infer.py
```

程序会交互式询问 GPU 类型、模型选择、输入文件等参数。

**推荐配置**：T4 GPU 性价比最高，适合一般转录任务。

> ⚠️ 本项目与 Modal 无任何关联，如有赞助意向，请提交 Issue。
> Not affiliated with Modal. For sponsorship inquiries, please open an issue.

详细说明请参考 [使用说明](使用说明.txt) 中的 "Modal 云端推理模式" 部分。

## 📖 详细文档 / Documentation

- 📝 [使用说明](使用说明.txt) - 详细的使用指南和参数配置
- 📋 [发行说明](RELEASE_NOTES_CN.md) - 版本更新日志和选择指南
- ⚙️ [生成配置](generation_config.json5) - 转录参数配置文件

## 🛠️ 高级配置 / Advanced Configuration

### 命令行参数

编辑批处理文件，在 `infer.exe` 后添加参数：

```batch
# 覆盖已存在的字幕文件
--overwrite

# 指定输出文件夹
--output_dir="路径"

# 自定义文件格式
--audio_suffixes="mp3,wav"
--sub_formats="srt,vtt,lrc"

# 调整日志级别
--log_level="INFO"

# 覆盖 Whisper 任务（translate 或 transcribe）
--task="transcribe"
```

### 转录参数调整

编辑 `generation_config.json5` 文件调整转录参数。

`generation_config.json5` 中可设置 `"task": "translate"` 或 `"task": "transcribe"`。命令行 `--task` 会覆盖配置文件中的 `task`。

参数详情请参考 [Faster Whisper 文档](https://github.com/SYSTRAN/faster-whisper/blob/dea24cbcc6cbef23ff599a63be0bbb647a0b23d6/faster_whisper/transcribe.py#L733)

补充：字幕合并/去重（`segment_merge`）
- 用于合并一些重复/重叠的片段，减少重复字幕。
- 如遇到“单条字幕持续时间异常过长”的情况，可调小 `segment_merge.max_gap_ms` 或 `segment_merge.max_duration_ms`，或将 `segment_merge.enabled` 设为 `false`。

## 🔗 相关链接 / Links

- **Faster Whisper**: https://github.com/SYSTRAN/faster-whisper
- **海南鸡模型**: https://huggingface.co/chickenrice0721/whisper-large-v2-translate-zh-v0.2-st-ct2
- **TransWithAI 日文转录模型**: https://huggingface.co/TransWithAI/whisper-ja-1.5B-ct2
- **日文转录原版模型**: https://huggingface.co/efwkjn/whisper-ja-1.5B
- **音声优化 VAD 模型**: https://huggingface.co/TransWithAI/Whisper-Vad-EncDec-ASMR-onnx
- **OpenAI Whisper**: https://github.com/openai/whisper
- **Modal 云端平台**: https://modal.com/
- **AI汉化组**: https://t.me/transWithAI

## 💡 常见问题 / FAQ

**Q: GPU模式无法运行？**
A: 确认显卡驱动已更新到最新版本。NVIDIA 用户请确认 CUDA 版本匹配；AMD 用户请确认下载了对应 gfx 架构的版本

**Q: 字幕未生成？**
A: 检查文件格式是否支持，查看控制台错误信息，尝试使用 `--overwrite` 参数

**Q: 内存/显存不足？**
A: 使用低显存模式、切换到CPU模式，或使用 Modal 云端推理

**Q: 如何选择CUDA版本？**
A: 运行 `nvidia-smi` 查看CUDA Version，参考[发行说明](RELEASE_NOTES_CN.md)中的兼容性表

## 📞 技术支持 / Support

如遇到问题，请：
1. 查看[使用说明](使用说明.txt)和[发行说明](RELEASE_NOTES_CN.md)
2. 检查显卡驱动是否为最新版本
3. 确认选择了正确的CUDA版本
4. 提交Issue到项目仓库

## ⭐ 小星星 / Star History

[![Star History Chart](https://star-history.dera.page/svg?repos=TransWithAI/Faster-Whisper-TransWithAI-ChickenRice&type=Date)](https://star-history.dera.page/#TransWithAI/Faster-Whisper-TransWithAI-ChickenRice&Date)

## 📄 许可证 / License

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

*本工具基于 Faster Whisper 开发，海南鸡模型经过5000小时音频数据优化训练，专门针对日文转中文翻译场景。*
*由AI汉化组开源维护，永久免费。*

**再次感谢某匿名群友的算力和技术支持！**
