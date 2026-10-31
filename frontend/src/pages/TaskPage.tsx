import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  Alert,
  Button,
  Card,
  Checkbox,
  Col,
  Collapse,
  Form,
  Input,
  InputNumber,
  Progress,
  Row,
  Select,
  Space,
  Switch,
  Table,
  Tag,
  Typography,
  message,
} from "antd";
import {
  FileAddOutlined,
  FolderAddOutlined,
  DeleteOutlined,
  CaretRightOutlined,
  StopOutlined,
} from "@ant-design/icons";
import { getApi, onEvent, ModelInfo } from "../api";
import LogPanel from "../components/LogPanel";

const AUDIO_SUFFIXES_DEFAULT = "mp3,wav,flac,m4a,aac,ogg,wma,mp4,mkv,avi,mov,webm,flv,wmv";

interface FileRow {
  path: string;
  key: string;
}

export default function TaskPage() {
  const [api, setApi] = useState<ReturnType<typeof getApi> | null>(null);
  const [files, setFiles] = useState<FileRow[]>([]);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<{ current: number; total: number; file: string } | null>(null);
  const [vadProgress, setVadProgress] = useState<{ current: number; total: number; percent: number } | null>(null);
  const [logKey, setLogKey] = useState(0);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [form] = Form.useForm();
  const runningRef = useRef(false);

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
    api.listModels().then((r) => {
      if (r.ok) setModels(r.models.filter((m) => m.has_model));
    });
  }, [api]);

  useEffect(() => {
    const off = onEvent((e) => {
      if (e.type === "task_progress") {
        setProgress({ current: e.current, total: e.total, file: e.file });
      } else if (e.type === "task_vad_progress") {
        setVadProgress({ current: e.current, total: e.total, percent: e.percent });
      } else if (e.type === "task_exit") {
        runningRef.current = false;
        setRunning(false);
        if (e.code === 0) message.success("任务完成");
        else message.error(`任务退出（代码 ${e.code}）`);
      }
    });
    return off;
  }, []);

  const addFiles = useCallback(
    async (paths: string[]) => {
      setFiles((prev) => {
        const set = new Set(prev.map((f) => f.path));
        const next = [...prev];
        paths.forEach((p) => {
          if (!set.has(p)) next.push({ path: p, key: p });
        });
        return next;
      });
    },
    []
  );

  const pickFiles = async () => {
    if (!api) return;
    const r = await api.selectFiles("选择音频/视频文件");
    if (r.ok && r.paths.length) addFiles(r.paths);
  };

  const pickFolder = async () => {
    if (!api) return;
    const r = await api.selectFolder("选择包含音视频的文件夹");
    if (r.ok && r.path) addFiles([r.path]);
  };

  const start = async () => {
    if (!api || files.length === 0) {
      message.warning("请先添加文件或文件夹");
      return;
    }
    const v = form.getFieldsValue();
    if (!runningRef.current) {
      setProgress(null);
      setVadProgress(null);
      setLogKey((k) => k + 1);
    }
    setError(null);
    const numOrNull = (x: unknown): number | null =>
      typeof x === "number" && !Number.isNaN(x) ? x : null;
    const opts = {
      files: files.map((f) => f.path),
      task: v.task,
      model_path: v.model_path || undefined,
      device: v.device,
      compute_type: v.compute_type,
      sub_formats: (v.sub_formats || []) as string[],
      output_dir: v.output_dir || undefined,
      overwrite: !!v.overwrite,
      audio_suffixes: v.audio_suffixes || undefined,
      enable_batching: !!v.enable_batching,
      batch_size: numOrNull(v.batch_size),
      max_batch_size: v.max_batch_size ?? 8,
      vad_threshold: numOrNull(v.vad_threshold),
      vad_min_speech_duration_ms: numOrNull(v.vad_min_speech_duration_ms),
      vad_min_silence_duration_ms: numOrNull(v.vad_min_silence_duration_ms),
      vad_speech_pad_ms: numOrNull(v.vad_speech_pad_ms),
      merge_enabled: v.merge_enabled ?? null,
      merge_max_gap_ms: numOrNull(v.merge_max_gap_ms),
      merge_max_duration_ms: numOrNull(v.merge_max_duration_ms),
      smart_split: v.smart_split ?? null,
      target_chunk_duration_s: numOrNull(v.target_chunk_duration_s),
    };
    const r = await api.startTask(opts);
    if (r.ok) {
      runningRef.current = true;
      setRunning(true);
    } else {
      setError(r.error || "启动失败");
    }
  };

  const cancel = async () => {
    if (!api) return;
    await api.cancelTask();
  };

  const pickOutputDir = async () => {
    if (!api) return;
    const r = await api.selectFolder("选择输出目录（取消则输出到源文件目录）");
    if (r.ok && r.path) form.setFieldValue("output_dir", r.path);
  };

  const pickModelPath = async () => {
    if (!api) return;
    const r = await api.selectFolder("选择模型目录");
    if (r.ok && r.path) form.setFieldValue("model_path", r.path);
  };

  const percent = progress && progress.total > 0 ? Math.round((progress.current / progress.total) * 100) : 0;

  return (
    <Row gutter={16} style={{ height: "100%" }}>
      <Col span={14} style={{ overflow: "auto", paddingRight: 4 }}>
        <Card size="small" title="输入文件 / Input" extra={
          <Space>
            <Button size="small" icon={<FileAddOutlined />} onClick={pickFiles}>添加文件</Button>
            <Button size="small" icon={<FolderAddOutlined />} onClick={pickFolder}>添加文件夹</Button>
            <Button size="small" danger icon={<DeleteOutlined />} onClick={() => setFiles([])} disabled={running} />
          </Space>
        }>
          <div
            style={{
              border: "1px dashed #888",
              borderRadius: 6,
              padding: 8,
              marginBottom: 8,
              minHeight: 100,
              maxHeight: 220,
              overflow: "auto",
            }}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              const dropped = Array.from(e.dataTransfer.files)
                .map((f) => (f as File & { path?: string }).path)
                .filter((p): p is string => !!p);
              if (dropped.length === 0) {
                message.info("浏览器拖放未提供完整路径，请使用「添加文件/添加文件夹」按钮选择");
                return;
              }
              addFiles(dropped);
            }}
          >
            {files.length === 0 ? (
              <Typography.Text type="secondary">拖入音视频文件/文件夹，或点击右上角按钮添加（支持 mp3/wav/flac/m4a/mp4/mkv 等）</Typography.Text>
            ) : (
              <Table
                size="small"
                dataSource={files}
                columns={[
                  { title: "路径", dataIndex: "path", key: "path", ellipsis: true },
                  {
                    title: "",
                    key: "op",
                    width: 60,
                    render: (_: unknown, row: FileRow) => (
                      <Button
                        size="small"
                        type="text"
                        danger
                        icon={<DeleteOutlined />}
                        disabled={running}
                        onClick={() => setFiles((prev) => prev.filter((f) => f.path !== row.path))}
                      />
                    ),
                  },
                ]}
                pagination={false}
              />
            )}
          </div>
        </Card>

        <Card size="small" title="参数 / Options" style={{ marginTop: 12 }}>
          <Form
            form={form}
            layout="vertical"
            size="small"
            initialValues={{
              task: "translate",
              device: "auto",
              compute_type: "auto",
              sub_formats: ["srt", "vtt", "lrc"],
              overwrite: false,
              audio_suffixes: AUDIO_SUFFIXES_DEFAULT,
              enable_batching: false,
              max_batch_size: 8,
            }}
          >
            <Row gutter={12}>
              <Col span={8}>
                <Form.Item name="task" label="任务类型">
                  <Select
                    options={[
                      { value: "translate", label: "翻译（日文→中文）" },
                      { value: "transcribe", label: "转录（日文原文）" },
                    ]}
                  />
                </Form.Item>
              </Col>
              <Col span={8}>
                <Form.Item name="device" label="设备">
                  <Select
                    options={[
                      { value: "auto", label: "自动检测" },
                      { value: "cuda", label: "GPU (CUDA/ROCm)" },
                      { value: "cpu", label: "CPU" },
                    ]}
                  />
                </Form.Item>
              </Col>
              <Col span={8}>
                <Form.Item name="compute_type" label="计算精度">
                  <Select
                    options={[
                      { value: "auto", label: "自动" },
                      { value: "float16", label: "float16" },
                      { value: "int8_float16", label: "int8_float16（低显存）" },
                      { value: "int8", label: "int8" },
                      { value: "float32", label: "float32" },
                    ]}
                  />
                </Form.Item>
              </Col>
            </Row>

            <Row gutter={12}>
              <Col span={16}>
                <Form.Item name="model_path" label="模型（models/ 下的子目录，留空使用 models 根目录）">
                  <Select
                    allowClear
                    showSearch
                    placeholder="models"
                    options={[
                      { value: "", label: "models（根目录）" },
                      ...models.map((m) => ({ value: m.path, label: `${m.name} (${m.size_mb.toFixed(0)} MB)` })),
                    ]}
                  />
                </Form.Item>
              </Col>
              <Col span={8} style={{ display: "flex", alignItems: "flex-end", paddingBottom: 24 }}>
                <Button size="small" onClick={pickModelPath}>浏览…</Button>
              </Col>
            </Row>

            <Row gutter={12}>
              <Col span={16}>
                <Form.Item name="output_dir" label="输出目录（留空 = 源文件所在目录）">
                  <Input placeholder="输出到源文件所在目录" />
                </Form.Item>
              </Col>
              <Col span={8} style={{ display: "flex", alignItems: "flex-end", paddingBottom: 24 }}>
                <Button size="small" onClick={pickOutputDir}>选择…</Button>
              </Col>
            </Row>

            <Row gutter={12}>
              <Col span={12}>
                <Form.Item name="sub_formats" label="输出字幕格式">
                  <Checkbox.Group
                    options={[
                      { value: "srt", label: "SRT" },
                      { value: "vtt", label: "VTT" },
                      { value: "lrc", label: "LRC" },
                      { value: "txt", label: "TXT" },
                    ]}
                  />
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item name="overwrite" valuePropName="checked" label="覆盖已有字幕">
                  <Switch />
                </Form.Item>
              </Col>
            </Row>

            <Collapse
              size="small"
              items={[
                {
                  key: "batch",
                  label: "批处理加速（高显存模式）",
                  children: (
                    <>
                      <Form.Item name="enable_batching" valuePropName="checked" noStyle>
                        <Checkbox>启用批处理推理（建议 8GB+ 显存）</Checkbox>
                      </Form.Item>
                      <Row gutter={12} style={{ marginTop: 8 }}>
                        <Col span={8}>
                          <Form.Item name="batch_size" label="批次大小（留空自动探测）">
                            <InputNumber min={1} max={64} style={{ width: "100%" }} placeholder="自动" />
                          </Form.Item>
                        </Col>
                        <Col span={8}>
                          <Form.Item name="max_batch_size" label="最大自动批次">
                            <InputNumber min={1} max={64} style={{ width: "100%" }} />
                          </Form.Item>
                        </Col>
                      </Row>
                    </>
                  ),
                },
                {
                  key: "vad",
                  label: "VAD 参数",
                  children: (
                    <Row gutter={12}>
                      <Col span={6}>
                        <Form.Item name="vad_threshold" label="阈值 (0.3~0.7)">
                          <InputNumber min={0} max={1} step={0.05} style={{ width: "100%" }} placeholder="默认" />
                        </Form.Item>
                      </Col>
                      <Col span={6}>
                        <Form.Item name="vad_min_speech_duration_ms" label="最小语音时长(ms)">
                          <InputNumber min={0} style={{ width: "100%" }} placeholder="默认" />
                        </Form.Item>
                      </Col>
                      <Col span={6}>
                        <Form.Item name="vad_min_silence_duration_ms" label="最小静音时长(ms)">
                          <InputNumber min={0} style={{ width: "100%" }} placeholder="默认" />
                        </Form.Item>
                      </Col>
                      <Col span={6}>
                        <Form.Item name="vad_speech_pad_ms" label="语音填充(ms)">
                          <InputNumber min={0} style={{ width: "100%" }} placeholder="默认" />
                        </Form.Item>
                      </Col>
                    </Row>
                  ),
                },
                {
                  key: "merge",
                  label: "字幕合并 / 智能切分",
                  children: (
                    <>
                      <Row gutter={12}>
                        <Col span={8}>
                          <Form.Item name="merge_enabled" label="合并重复片段">
                            <Select
                              allowClear
                              placeholder="默认（配置文件）"
                              options={[
                                { value: true, label: "开启" },
                                { value: false, label: "关闭" },
                              ]}
                            />
                          </Form.Item>
                        </Col>
                        <Col span={8}>
                          <Form.Item name="merge_max_gap_ms" label="最大间隔(ms)">
                            <InputNumber min={0} style={{ width: "100%" }} placeholder="默认" />
                          </Form.Item>
                        </Col>
                        <Col span={8}>
                          <Form.Item name="merge_max_duration_ms" label="单条最大时长(ms)">
                            <InputNumber min={0} style={{ width: "100%" }} placeholder="默认" />
                          </Form.Item>
                        </Col>
                      </Row>
                      <Row gutter={12}>
                        <Col span={12}>
                          <Form.Item name="smart_split" label="智能 VAD 切分">
                            <Select
                              allowClear
                              placeholder="默认（配置文件）"
                              options={[
                                { value: true, label: "开启" },
                                { value: false, label: "关闭" },
                              ]}
                            />
                          </Form.Item>
                        </Col>
                        <Col span={12}>
                          <Form.Item name="target_chunk_duration_s" label="切分块目标时长(秒)">
                            <InputNumber min={1} max={30} style={{ width: "100%" }} placeholder="默认" />
                          </Form.Item>
                        </Col>
                      </Row>
                    </>
                  ),
                },
                {
                  key: "misc",
                  label: "其他",
                  children: (
                    <Form.Item name="audio_suffixes" label="音频后缀（逗号分隔）">
                      <Input />
                    </Form.Item>
                  ),
                },
              ]}
            />
          </Form>
        </Card>
      </Col>

      <Col span={10} style={{ display: "flex", flexDirection: "column", minHeight: 0 }}>
        <Card size="small" title="运行 / Run" style={{ flexShrink: 0 }}>
          <Space>
            <Button
              type="primary"
              icon={<CaretRightOutlined />}
              onClick={start}
              disabled={running}
            >
              开始
            </Button>
            <Button danger icon={<StopOutlined />} onClick={cancel} disabled={!running}>
              停止
            </Button>
          </Space>
          <div style={{ marginTop: 12 }}>
            {progress ? (
              <>
                <Progress percent={percent} size="small" />
                <Typography.Text type="secondary" style={{ fontSize: 12 }} ellipsis>
                  [{progress.current}/{progress.total}] {progress.file}
                </Typography.Text>
              </>
            ) : (
              <Typography.Text type="secondary" style={{ fontSize: 12 }}>等待开始…</Typography.Text>
            )}
            {vadProgress && (
              <Tag style={{ marginTop: 4 }}>
                VAD: {vadProgress.current}/{vadProgress.total} ({vadProgress.percent.toFixed(1)}%)
              </Tag>
            )}
          </div>
          {error && (
            <Alert type="error" message={error} style={{ marginTop: 8 }} closable />
          )}
        </Card>
        <div style={{ marginTop: 12, flex: 1, minHeight: 0 }}>
          <LogPanel
            key={logKey}
            types={["task_log"]}
            toLine={(e) => (e.type === "task_log" ? `[${e.stream}] ${e.line}` : null)}
            height="100%"
          />
        </div>
      </Col>
    </Row>
  );
}
