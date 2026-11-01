import React, { useCallback, useEffect, useState } from "react";
import {
  Button,
  Card,
  Col,
  Form,
  Input,
  Row,
  Select,
  Space,
  Table,
  Tag,
  Typography,
  message,
} from "antd";
import { CaretRightOutlined, ReloadOutlined, StopOutlined } from "@ant-design/icons";
import { getApi, onEvent, DownloadSource, ModelInfo } from "../api";
import LogPanel from "../components/LogPanel";

const PRESETS = [
  {
    repo: "chickenrice0721/whisper-large-v2-translate-zh-v0.2-st-ct2",
    target: "chickenrice-v2",
    label: "海南鸡v2（日文→中文翻译）",
  },
  {
    repo: "TransWithAI/whisper-ja-1.5B-ct2",
    target: "whisper-ja-1.5B-ct2",
    label: "TransWithAI 日文转录 1.5B",
  },
  {
    repo: "openai/whisper-base",
    target: "whisper-base",
    label: "基础版 whisper-base",
  },
];

// 后端不可用时的兜底清单（正常情况下由 app.py 的 DOWNLOAD_SOURCES 提供）
const FALLBACK_SOURCES: DownloadSource[] = [
  {
    id: "auto",
    label: "自动（官方优先，失败切换 hf-mirror）",
    endpoint: "",
    note: "默认：优先官方源，失败或超时时自动改用 hf-mirror.com",
  },
  {
    id: "hf-mirror",
    label: "hf-mirror.com（国内推荐）",
    endpoint: "https://hf-mirror.com",
    note: "国内公益镜像，与 HuggingFace 协议完全一致，国内网络通常最快最稳",
  },
  {
    id: "official",
    label: "Hugging Face 官方",
    endpoint: "https://huggingface.co",
    note: "官方源；海外网络或已配置代理时使用",
  },
];

const CUSTOM_NOTE =
  "自定义端点需兼容 HuggingFace 协议（支持 /api/models 与 /resolve 路径），例如自建反代。";

export default function ModelsPage() {
  const [api, setApi] = useState<ReturnType<typeof getApi> | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelDir, setModelDir] = useState("");
  const [form] = Form.useForm();
  const [logKey, setLogKey] = useState(0);
  const [downloading, setDownloading] = useState(false);
  const [sources, setSources] = useState<DownloadSource[]>(FALLBACK_SOURCES);
  const downloadingRef = React.useRef(false);
  const sourceValue = Form.useWatch("source", form) as string | undefined;

  const currentSource =
    sources.find((s) => s.id === (sourceValue || "auto")) || sources[0];
  const currentNote = sourceValue === "custom" ? CUSTOM_NOTE : currentSource?.note || "";

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

  const refresh = useCallback(async () => {
    if (!api) return;
    const r = await api.listModels();
    if (r.ok) {
      setModels(r.models);
      setModelDir(r.model_dir);
    }
  }, [api]);

  useEffect(() => {
    if (api) refresh();
  }, [api, refresh]);

  // 下载源清单由后端提供，读取失败时沿用本地兜底列表
  useEffect(() => {
    if (!api) return;
    api
      .getDownloadSources()
      .then((r) => {
        if (r.ok && r.sources && r.sources.length > 0) setSources(r.sources);
      })
      .catch(() => {});
  }, [api]);

  useEffect(() => {
    const off = onEvent((e) => {
      if (e.type === "download_exit") {
        downloadingRef.current = false;
        setDownloading(false);
        if (e.code === 0) {
          message.success("模型下载完成");
        } else {
          message.error(`下载退出（代码 ${e.code}）`);
        }
        refresh();
      }
    });
    return off;
  }, [refresh]);

  const startDownload = async () => {
    if (!api) return;
    const v = form.getFieldsValue();
    if (!v.repo) {
      message.warning("请选择或输入 HuggingFace 模型仓库");
      return;
    }
    setLogKey((k) => k + 1);
    const endpoint =
      v.source === "custom"
        ? String(v.custom_endpoint || "").trim()
        : currentSource?.endpoint || undefined;
    if (v.source === "custom" && !endpoint) {
      message.warning("请填写自定义下载源地址");
      return;
    }
    const r = await api.downloadModel({
      repo: v.repo,
      target_dir: v.target_dir || undefined,
      force: !!v.force,
      endpoint: endpoint || undefined,
    });
    if (r.ok) {
      downloadingRef.current = true;
      setDownloading(true);
    } else {
      message.error(r.error || "下载启动失败");
    }
  };

  return (
    <Row gutter={16}>
      <Col span={14}>
        <Card
          size="small"
          title={`已安装模型（${modelDir || "models/"}）`}
          extra={
            <Button size="small" icon={<ReloadOutlined />} onClick={refresh}>
              刷新
            </Button>
          }
        >
          <Table
            size="small"
            dataSource={models}
            rowKey="path"
            pagination={false}
            columns={[
              {
                title: "名称",
                dataIndex: "name",
                key: "name",
                render: (v: string, r: ModelInfo) => (
                  <Space>
                    {v}
                    {r.is_root && <Tag>根目录</Tag>}
                    {r.has_model && <Tag color="green">可用</Tag>}
                  </Space>
                ),
              },
              {
                title: "大小",
                dataIndex: "size_mb",
                key: "size_mb",
                width: 120,
                render: (v: number) => `${v.toFixed(0)} MB`,
              },
              {
                title: "路径",
                dataIndex: "path",
                key: "path",
                ellipsis: true,
              },
            ]}
          />
          <Typography.Text type="secondary" style={{ fontSize: 12, marginTop: 8, display: "block" }}>
            提示：模型目录下应包含 model.bin / model.safetensors 与 config.json 等文件。
          </Typography.Text>
        </Card>

        <Card size="small" title="下载新模型" style={{ marginTop: 12 }}>
          <Form form={form} layout="vertical" size="small" initialValues={{ source: "auto" }}>
            <Form.Item name="repo" label="HuggingFace 仓库">
              <Select
                showSearch
                allowClear
                placeholder="选择预设或输入 user/repo"
                options={PRESETS.map((p) => ({
                  value: p.repo,
                  label: `${p.label} — ${p.repo}`,
                }))}
                filterOption={(input, option) =>
                  String(option?.label ?? "").toLowerCase().includes(input.toLowerCase()) ||
                  String(option?.value ?? "").toLowerCase().includes(input.toLowerCase())
                }
              />
            </Form.Item>
            <Form.Item name="source" label="下载源（HuggingFace 端点）">
              <Select
                options={[
                  ...sources.map((s) => ({ value: s.id, label: s.label })),
                  { value: "custom", label: "自定义端点…" },
                ]}
              />
            </Form.Item>
            {sourceValue === "custom" && (
              <Form.Item name="custom_endpoint" label="自定义端点地址（需兼容 HuggingFace 协议）">
                <Input placeholder="https://your-mirror.example.com" allowClear />
              </Form.Item>
            )}
            <Row gutter={12}>
              <Col span={12}>
                <Form.Item name="target_dir" label="目标子目录（留空 = 仓库名）">
                  <Input placeholder="models/ 下子目录名" />
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item name="force" valuePropName="checked" label="强制重新下载">
                  <Select
                    allowClear
                    placeholder="否"
                    options={[
                      { value: true, label: "是" },
                      { value: false, label: "否" },
                    ]}
                  />
                </Form.Item>
              </Col>
            </Row>
            <Space>
              <Button
                type="primary"
                icon={<CaretRightOutlined />}
                onClick={startDownload}
                disabled={downloading}
              >
                开始下载
              </Button>
              <Button
                danger
                icon={<StopOutlined />}
                disabled={!downloading}
                onClick={async () => {
                  if (!api) return;
                  await api.cancelDownload();
                }}
              >
                停止
              </Button>
            </Space>
            <Typography.Text type="secondary" style={{ fontSize: 12, display: "block", marginTop: 8 }}>
              当前源：{currentNote}
            </Typography.Text>
            <Typography.Text type="secondary" style={{ fontSize: 12, display: "block", marginTop: 4 }}>
              下载始终包含音声优化 VAD 模型与 whisper-base 特征提取文件（离线运行所需）。
              国内推荐 <strong>hf-mirror.com</strong>（与 HuggingFace 协议一致，可整体换源）；
              ModelScope（魔搭）是独立模型社区、接口不兼容，只能作为替代渠道手动下载。
            </Typography.Text>
          </Form>
        </Card>
      </Col>
      <Col span={10}>
        <Card size="small" title="下载日志" style={{ height: "100%" }}>
          <LogPanel
            key={logKey}
            types={["download_log", "download_exit"]}
            toLine={(e) =>
              e.type === "download_log"
                ? e.line
                : e.type === "download_exit"
                  ? `[exit] ${e.code}`
                  : null
            }
            height={480}
          />
        </Card>
      </Col>
    </Row>
  );
}
