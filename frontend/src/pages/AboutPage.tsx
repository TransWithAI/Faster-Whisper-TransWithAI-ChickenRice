import React from "react";
import { Card, Space, Typography } from "antd";

export default function AboutPage() {
  return (
    <Card size="small" style={{ maxWidth: 860 }}>
      <Typography.Title level={4}>🎙️ Faster Whisper TransWithAI ChickenRice</Typography.Title>
      <Typography.Paragraph>
        高性能音视频转录和翻译工具 — 基于 Faster Whisper 与音声优化 VAD 的日文转中文优化版本。
      </Typography.Paragraph>

      <Typography.Title level={5}>⚠️ 重要声明</Typography.Title>
      <Typography.Paragraph>
        本软件为开源软件，完全免费开源。任何第三方非免费下载均为智商税。
      </Typography.Paragraph>

      <Typography.Title level={5}>🙏 致谢</Typography.Title>
      <ul style={{ lineHeight: "24px" }}>
        <li>基于 <a href="https://github.com/SYSTRAN/faster-whisper" target="_blank" rel="noreferrer">SYSTRAN/faster-whisper</a> 开发</li>
        <li>使用 <a href="https://huggingface.co/chickenrice0721/whisper-large-v2-translate-zh-v0.2-st-ct2" target="_blank" rel="noreferrer">海南鸡v2</a> 日文转中文优化模型（5000 小时音频训练）</li>
        <li>可使用 <a href="https://huggingface.co/TransWithAI/whisper-ja-1.5B-ct2" target="_blank" rel="noreferrer">TransWithAI/whisper-ja-1.5B-ct2</a> 日文原文转录 bf16 模型</li>
        <li>使用 <a href="https://huggingface.co/TransWithAI/Whisper-Vad-EncDec-ASMR-onnx" target="_blank" rel="noreferrer">音声优化 VAD 模型</a></li>
        <li>感谢某匿名群友的算力和技术支持</li>
      </ul>

      <Typography.Title level={5}>🔗 链接</Typography.Title>
      <Space direction="vertical">
        <a href="https://github.com/TransWithAI/Faster-Whisper-TransWithAI-ChickenRice" target="_blank" rel="noreferrer">
          开源地址：github.com/TransWithAI/Faster-Whisper-TransWithAI-ChickenRice
        </a>
        <Typography.Text>开发团队：AI汉化组（https://t.me/transWithAI）</Typography.Text>
        <Typography.Text type="secondary">MIT License</Typography.Text>
      </Space>
    </Card>
  );
}
