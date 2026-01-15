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
- 🐔 使用 [chickenrice0721/whisper-large-v2-translate-zh-v0.2-st](https://huggingface.co/chickenrice0721/whisper-large-v2-translate-zh-v0.2-st) 日文转中文优化模型
- 🔊 使用 [TransWithAI/Whisper-Vad-EncDec-ASMR-onnx](https://huggingface.co/TransWithAI/Whisper-Vad-EncDec-ASMR-onnx) 音声优化 VAD 模型
- ☁️ 感谢 [@Randomless](https://github.com/Randomless) 贡献 Modal 云端推理功能
- 💪 **感谢某匿名群友的算力和技术支持**

## ✨ 功能特性 / Features

- 🎯 **高精度日文转中文翻译**: 基于5000小时音频数据训练的"海南鸡v2"日文转中文优化模型
- 🚀 **GPU加速**: 支持CUDA 11.8/12.2/12.8，充分利用NVIDIA显卡性能
- ☁️ **云端推理**: 支持 Modal 云端 GPU 推理，无本地显卡也能使用
- 📝 **多格式输出**: 支持SRT、VTT、LRC等多种字幕格式
- 🎬 **音视频支持**: 支持常见音频(mp3/wav/flac等)和视频格式(mp4/mkv/avi等)
- 💾 **智能缓存**: 自动跳过已处理文件，提高批量处理效率
- 🔧 **灵活配置**: 可自定义转录参数，满足不同场景需求

## 📦 版本说明 / Package Variants

### 基础版 (Base Package) - 约 2.2GB
- ✅ 所有 GPU 依赖项
- ✅ 音声优化 VAD（语音活动检测）模型
- ❌ 不含 Whisper 模型（需自行下载）

### 海南鸡版 (ChickenRice Edition) - 约 4.4GB
- ✅ 所有 GPU 依赖项
- ✅ 音声优化 VAD（语音活动检测）模型
- ✅ **"海南鸡v2 5000小时"** 日文转中文优化模型（开箱即用）

## 🚀 快速开始 / Quick Start

### 1. 选择适合的CUDA版本 / Choose CUDA Version

运行 `nvidia-smi` 查看您的CUDA版本：

| 显卡系列 | 推荐 CUDA 版本 |
|---------|--------------|
| GTX 10/16系列 | CUDA 11.8 |
| RTX 20/30系列 | CUDA 11.8 或 12.2 |
| RTX 40系列 | CUDA 12.2 或 12.8 |
| RTX 50系列 | **必须使用 CUDA 12.8** |

### 2. 下载对应版本 / Download

从 [Releases](https://github.com/TransWithAI/Faster-Whisper-TransWithAI-ChickenRice/releases) 页面下载对应版本

### 3. 使用方法 / Usage

将音视频文件拖放到相应的批处理文件：

```bash
# GPU模式（推荐，显存≥6GB）
运行(GPU).bat

# GPU低显存模式（显存4GB）
运行(GPU,低显存模式).bat

# CPU模式（无显卡用户）
运行(CPU).bat

# 视频专用模式
运行(翻译视频)(GPU).bat
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
```

### 转录参数调整

编辑 `generation_config.json5` 文件调整转录参数。

参数详情请参考 [Faster Whisper 文档](https://github.com/SYSTRAN/faster-whisper/blob/dea24cbcc6cbef23ff599a63be0bbb647a0b23d6/faster_whisper/transcribe.py#L733)

补充：字幕合并/去重（`segment_merge`）
- 用于合并一些重复/重叠的片段，减少重复字幕。
- 如遇到“单条字幕持续时间异常过长”的情况，可调小 `segment_merge.max_gap_ms` 或 `segment_merge.max_duration_ms`，或将 `segment_merge.enabled` 设为 `false`。

## 🔗 相关链接 / Links

- **Faster Whisper**: https://github.com/SYSTRAN/faster-whisper
- **海南鸡模型**: https://huggingface.co/chickenrice0721/whisper-large-v2-translate-zh-v0.2-st
- **音声优化 VAD 模型**: https://huggingface.co/TransWithAI/Whisper-Vad-EncDec-ASMR-onnx
- **OpenAI Whisper**: https://github.com/openai/whisper
- **Modal 云端平台**: https://modal.com/
- **AI汉化组**: https://t.me/transWithAI

## 💡 常见问题 / FAQ

**Q: GPU模式无法运行？**
A: 确认是否为NVIDIA显卡，更新显卡驱动到最新版本

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

[![Star History Chart](https://api.star-history.com/svg?repos=TransWithAI/Faster-Whisper-TransWithAI-ChickenRice&type=Date)](https://star-history.com/#TransWithAI/Faster-Whisper-TransWithAI-ChickenRice&Date)

## 📄 许可证 / License

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

*本工具基于 Faster Whisper 开发，海南鸡模型经过5000小时音频数据优化训练，专门针对日文转中文翻译场景。*
*由AI汉化组开源维护，永久免费。*

**再次感谢某匿名群友的算力和技术支持！**
