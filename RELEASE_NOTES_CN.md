# 🎙️ Faster Whisper 转录工具 - 发行说明

## ⚠️ 重要声明

> **本软件为开源软件**
>
> 🔗 **开源地址**: https://github.com/TransWithAI/Faster-Whisper-TransWithAI-ChickenRice
>
> 👥 **开发团队**: AI汉化组 (https://t.me/transWithAI)

---

## 📦 发行包说明

本发行版包含多个变体版本，请根据您的显卡型号选择合适的版本：

### 🎯 版本类型说明

#### 基础版（Base Package）
- **下载大小**：约 2.2GB
- **包含内容**：
  - ✅ 所有 GPU 依赖项
  - ✅ 音声优化 VAD（语音活动检测）ONNX 模型
  - ❌ 不含 Whisper 模型（需自行下载）
- **适用场景**：需要使用自定义模型的用户

#### 海南鸡版（Chickenrice Edition）
- **下载大小**：约 4.4GB
- **包含内容**：
  - ✅ 所有 GPU 依赖项
  - ✅ 音声优化 VAD（语音活动检测）ONNX 模型
  - ✅ **"海南鸡v2 5000小时"** 日文转中文优化模型
- **适用场景**：开箱即用的日文转中文翻译
- **模型说明**：包含经过5000小时音频数据训练的海南鸡v2版本模型，专门优化日文转中文翻译

### 📌 文件命名规则

| 文件名后缀 | CUDA版本 | 模型类型 |
|-----------|---------|---------|
| `*_cu118.zip` | CUDA 11.8 | 基础版 |
| `*_cu118-chickenrice.zip` | CUDA 11.8 | 海南鸡版 |
| `*_cu122.zip` | CUDA 12.2 | 基础版 |
| `*_cu122-chickenrice.zip` | CUDA 12.2 | 海南鸡版 |
| `*_cu128.zip` | CUDA 12.8 | 基础版 |
| `*_cu128-chickenrice.zip` | CUDA 12.8 | 海南鸡版 |

---

## 🔍 如何选择正确的 CUDA 版本

### 方法一：通过 nvidia-smi 查询

1. 打开命令提示符或终端
2. 输入命令：`nvidia-smi`
3. 查看输出中的 **Driver Version** 和 **CUDA Version**

```
+-------------------------------------------------------------------------+
| NVIDIA-SMI 570.00       Driver Version: 570.00       CUDA Version: 12.8|
+-------------------------------------------------------------------------+
```

### 方法二：通过显卡型号和驱动版本对照表

#### 📊 NVIDIA 驱动版本与 CUDA 版本兼容性表

| CUDA 版本 | 最低驱动要求（Windows） | 最低驱动要求（Linux） | 推荐使用场景 |
|----------|------------------------|---------------------|------------|
| **CUDA 11.8** | ≥452.39 | ≥450.80.02 | 较旧的显卡（GTX 10系列、RTX 20/30系列） |
| **CUDA 12.2** | ≥525.60.13 | ≥525.60.13 | RTX 30/40系列，较新的驱动 |
| **CUDA 12.8** | ≥570.65 | ≥570.26 | RTX 40/50系列，最新驱动 |

#### 🎮 显卡型号推荐表

| 显卡系列 | 推荐 CUDA 版本 | 说明 |
|---------|--------------|------|
| GTX 10系列（1060/1070/1080等） | **CUDA 11.8** | 兼容性最好 |
| GTX 16系列（1650/1660等） | **CUDA 11.8** | 兼容性最好 |
| RTX 20系列（2060/2070/2080等） | **CUDA 11.8** 或 **12.2** | 根据驱动版本选择 |
| RTX 30系列（3060/3070/3080/3090等） | **CUDA 12.2** | 推荐使用 |
| RTX 40系列（4060/4070/4080/4090等） | **CUDA 12.2** 或 **12.8** | 最新驱动用12.8 |
| **RTX 50系列（5090/5080/5070等）** | **🔴 必须使用 CUDA 12.8** | ⚠️ 注意：RTX 50系列必须使用CUDA 12.8版本 |

### ⚠️ 重要提示

- **RTX 50系列用户**：由于新架构要求，**必须使用 CUDA 12.8 版本**，驱动版本必须 ≥570.00
- **驱动版本查询**：在 nvidia-smi 中显示的 CUDA Version 是您的驱动**支持的最高**CUDA版本
- **向下兼容**：高版本驱动可以运行低版本CUDA程序（例如：570驱动可以运行CUDA 11.8程序）
- **性能考虑**：使用与驱动匹配的CUDA版本可获得最佳性能

---

## 📥 模型下载说明

### 基础版用户（需自行下载模型）

基础版包含VAD模型，但**不包含**Whisper语音识别模型。您需要：

1. **从 Hugging Face 下载模型**
   - 示例模型地址：https://huggingface.co/chickenrice0721/whisper-large-v2-translate-zh-v0.2-st
   - 这是"海南鸡v2 5000小时"版本的日文转中文优化模型

2. **放置模型文件**
   ```
   将下载的模型文件放入：
   faster_whisper_transwithai_chickenrice/
   └── models/
       └── [您下载的模型文件夹]/
   ```

3. **其他可用模型**
   - OpenAI Whisper官方模型
   - 其他社区优化模型

### 海南鸡版用户（开箱即用）

海南鸡版已包含：
- ✅ 音声优化 VAD 语音活动检测模型
- ✅ "海南鸡v2 5000小时"日文转中文优化版Whisper模型
- ✅ 所有必要的配置文件

**无需额外下载**，解压后直接运行即可使用！

---

## 🚀 快速开始指南

### 1. 选择版本
根据上述表格，选择适合您显卡的CUDA版本

### 2. 下载对应版本
- 仅转录/翻译：下载基础版 + 自行下载模型
- 日文转中文优化：下载海南鸡版（推荐）

### 3. 解压并运行
```bash
# GPU模式（推荐）
将音视频文件拖放到 "运行(GPU).bat"

# CPU模式（无显卡用户）
将音视频文件拖放到 "运行(CPU).bat"

# 低显存模式（4GB显存）
将音视频文件拖放到 "运行(GPU,低显存模式).bat"
```

---

## 💡 常见问题

**Q: 我应该选择哪个CUDA版本？**
A: 运行 `nvidia-smi` 查看您的驱动版本，然后对照上表选择。

**Q: 海南鸡版和基础版有什么区别？**
A: 海南鸡版包含预训练的日文转中文优化模型（5000小时训练），基础版需要自行下载模型。

**Q: RTX 4090 应该用哪个版本？**
A: 推荐使用 CUDA 12.2 或 12.8 版本，取决于您的驱动版本。

**Q: 显存不足怎么办？**
A: 使用"低显存模式"批处理文件，或切换到CPU模式。

---

## 📝 更新日志

### v1.2 (2025-11-15)
- ⚡ CPU模式性能优化：添加 int16 计算类型支持，提升CPU处理速度
- 🎮 GPU兼容性改进：强制使用 float16 替代模型精度 bfloat16，提升显卡兼容性
- 🔧 精简批处理文件：合并视频翻译功能到主批处理文件中
- 📝 新增日志记录功能：自动保存运行日志到 latest.log 文件，方便问题反馈

### v1.1 (2025-11-14)
- 🌐 离线支持改进：预下载 whisper-base 模型文件，实现完全离线运行
- 📥 自动模型管理：WhisperFeatureExtractor 优先使用本地模型，避免网络超时
- 🔧 优化下载流程：支持从已有模型文件夹复制，减少重复下载
- 🚀 提升稳定性：解决网络不稳定环境下的 HuggingFace 连接超时问题

### v1.0 (2025-11-13)
- 🎯 支持多CUDA版本（11.8/12.2/12.8）
- 🚀 优化的日文转中文翻译效果（海南鸡v2版本）
- 🔊 音声优化的VAD语音活动检测
- 💾 改进的缓存机制，加快CI/CD构建速度
- 📦 分离的基础版和完整版，满足不同需求
- 🔧 自动VAD模型下载和管理

---

## 📞 技术支持

如遇到问题，请：
1. 检查显卡驱动是否为最新版本
2. 确认选择了正确的CUDA版本
3. 查看控制台输出的错误信息
4. 提交Issue到项目仓库: https://github.com/TransWithAI/Faster-Whisper-TransWithAI-ChickenRice

### 🔗 官方链接
- **GitHub仓库**: https://github.com/TransWithAI/Faster-Whisper-TransWithAI-ChickenRice
- **音声优化 VAD 模型**: https://huggingface.co/TransWithAI/Whisper-Vad-EncDec-ASMR-onnx
- **Telegram群组**: https://t.me/transWithAI
- **开发团队**: AI汉化组

---

## 🙏 致谢

- 🚀 基于 [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) 开发
- 🐔 使用 [chickenrice0721/whisper-large-v2-translate-zh-v0.2-st](https://huggingface.co/chickenrice0721/whisper-large-v2-translate-zh-v0.2-st) 日文转中文优化模型
- 🔊 使用 [TransWithAI/Whisper-Vad-EncDec-ASMR-onnx](https://huggingface.co/TransWithAI/Whisper-Vad-EncDec-ASMR-onnx) 音声优化 VAD 模型
- 🎙️ [OpenAI Whisper](https://github.com/openai/whisper) 原始项目
- 💪 **感谢某匿名群友的算力和技术支持**

---

*本工具基于 Faster Whisper 开发，海南鸡模型经过5000小时音频数据优化训练，专门针对日文转中文翻译场景。*
*由AI汉化组开源维护，永久免费。*
