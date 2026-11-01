import React, { useEffect, useState } from "react";
import { Button, Card, Col, Row, Space, Tag, Typography, message } from "antd";
import { FolderOpenOutlined, ReloadOutlined, ThunderboltOutlined } from "@ant-design/icons";
import { getApi, SystemInfo } from "../api";
import LogPanel from "../components/LogPanel";

function tag(ok: boolean | null, okText = "正常", badText = "不可用") {
  if (ok === null) return <Tag>检测中…</Tag>;
  return ok ? <Tag color="green">{okText}</Tag> : <Tag color="red">{badText}</Tag>;
}

export default function EnvPage() {
  const [api, setApi] = useState<ReturnType<typeof getApi> | null>(null);
  const [info, setInfo] = useState<SystemInfo | null>(null);
  const [logKey, setLogKey] = useState(0);
  const [checking, setChecking] = useState(false);

  useEffect(() => {
    const w = window as unknown as { pywebview?: { api: unknown } };
    const timer = setInterval(() => {
      if (w.pywebview && w.pywebview.api) {
        setApi(getApi());
        clearInterval(timer);
      }
    }, 200);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!api) return;
    api.getSystemInfo().then(setInfo);
  }, [api]);

  const runCheck = async () => {
    if (!api) return;
    setLogKey((k) => k + 1);
    setChecking(true);
    const r = await api.runEnvCheck();
    setChecking(false);
    if (!r.ok) message.error(r.error || "诊断失败");
    api.getSystemInfo().then(setInfo);
  };

  return (
    <Row gutter={16}>
      <Col span={14}>
        <Card
          size="small"
          title="运行环境"
          extra={
            <Space>
              <Button size="small" icon={<ReloadOutlined />} onClick={runCheck} loading={checking}>
                深度诊断
              </Button>
            </Space>
          }
        >
          {info && (
            <>
              <Row gutter={[8, 8]}>
                <Col span={12}>
                  <Typography.Text>Python：</Typography.Text>
                  <Typography.Text code>{info.python_version}</Typography.Text>
                </Col>
                <Col span={12}>
                  <Typography.Text>CTranslate2：</Typography.Text>
                  <Typography.Text code>{info.ctranslate2_version || "未安装"}</Typography.Text>
                </Col>
                <Col span={12}>
                  <Typography.Text>NVIDIA GPU：</Typography.Text>
                  {tag(info.cuda_available && info.cuda_device_count > 0, `检测到 ${info.cuda_device_count} 个`, "未检测到")}
                  {info.gpu_names.length > 0 && (
                    <Typography.Text code style={{ marginLeft: 4 }}>
                      {info.gpu_names.join(", ")}
                    </Typography.Text>
                  )}
                </Col>
                <Col span={12}>
                  <Typography.Text>ONNX Runtime（VAD）：</Typography.Text>
                  {tag(info.onnxruntime_available)}
                </Col>
              </Row>
              {info.onnxruntime_available && info.onnx_providers.length > 0 && (
                <Typography.Paragraph type="secondary" style={{ fontSize: 12, marginTop: 8 }}>
                  ONNX providers: {info.onnx_providers.join(", ")}
                </Typography.Paragraph>
              )}
              <Space style={{ marginTop: 8 }}>
                <Button size="small" icon={<FolderOpenOutlined />} onClick={async () => {
                  if (!api) return;
                  await api.openPath(info.model_dir);
                }}>
                  打开模型目录
                </Button>
                <Button size="small" icon={<FolderOpenOutlined />} onClick={async () => {
                  if (!api) return;
                  await api.openLogsFolder();
                }}>
                  打开日志目录
                </Button>
              </Space>
            </>
          )}
        </Card>
        <Card size="small" title="故障排除提示" style={{ marginTop: 12 }}>
          <ul style={{ fontSize: 13, paddingLeft: 18, lineHeight: "22px" }}>
            <li>GPU 模式无法运行：确认 NVIDIA 显卡、更新驱动到最新、检查 CUDA 版本（RTX 50 系必须 CUDA 12.8）。</li>
            <li>字幕未生成：检查文件格式是否支持；尝试勾选"覆盖已有字幕"重新生成。</li>
            <li>显存不足：使用 int8_float16 低显存精度或改用 CPU 模式。</li>
            <li>AMD 显卡：使用带 gfx*** 后缀的发行版，ROCm/HIP 运行时已内置。</li>
          </ul>
        </Card>
      </Col>
      <Col span={10}>
        <Card
          size="small"
          title="深度诊断输出"
          extra={<ThunderboltOutlined />}
          style={{ height: "100%" }}
        >
          <LogPanel
            key={logKey}
            types={["task_log"]}
            toLine={(e) => (e.type === "task_log" && e.stream === "stdout" ? e.line : null)}
            height={520}
          />
        </Card>
      </Col>
    </Row>
  );
}
