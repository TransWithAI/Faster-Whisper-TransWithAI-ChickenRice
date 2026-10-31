import React, { useEffect, useRef, useState } from "react";
import {
  Alert,
  Button,
  Card,
  Checkbox,
  Col,
  Form,
  Input,
  InputNumber,
  Row,
  Select,
  Space,
  Typography,
  message,
} from "antd";
import { CaretRightOutlined, FolderAddOutlined, StopOutlined } from "@ant-design/icons";
import { getApi, onEvent } from "../api";
import LogPanel from "../components/LogPanel";

const GPU_CHOICES = ["T4", "L4", "L40S", "A10G", "A100-40GB", "A100-80GB", "H100", "H200", "B200"];

export default function ModalPage() {
  const [api, setApi] = useState<ReturnType<typeof getApi> | null>(null);
  const [form] = Form.useForm();
  const [logKey, setLogKey] = useState(0);
  const [running, setRunning] = useState(false);
  const [installed, setInstalled] = useState<boolean | null>(null);
  const [installMsg, setInstallMsg] = useState<string | undefined>();
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
    api.modalCheck().then((r) => {
      setInstalled(r.installed);
      setInstallMsg(r.message);
    });
  }, [api]);

  useEffect(() => {
    const off = onEvent((e) => {
      if (e.type === "modal_exit") {
        runningRef.current = false;
        setRunning(false);
        if (e.code === 0) message.success("云端任务完成");
        else message.error(`云端任务退出（代码 ${e.code}）`);
      }
    });
    return off;
  }, []);

  const pickInput = async () => {
    if (!api) return;
    const fr = await api.selectFiles("选择音频文件");
    if (fr.ok && fr.paths.length) {
      form.setFieldValue("input_path", fr.paths[0]);
      return;
    }
    const dr = await api.selectFolder("或选择包含音频的文件夹");
    if (dr.ok && dr.path) form.setFieldValue("input_path", dr.path);
  };

  const run = async () => {
    if (!api) return;
    const v = form.getFieldsValue();
    if (!v.input_path) {
      message.warning("请选择输入文件或文件夹");
      return;
    }
    if (v.model === "custom" && !v.custom_repo) {
      message.warning("自定义模型需填写 HuggingFace 仓库");
      return;
    }
    setLogKey((k) => k + 1);
    const r = await api.modalRun({
      input_path: v.input_path,
      gpu: v.gpu || "T4",
      model: v.model,
      custom_repo: v.custom_repo,
      custom_target_dir: v.custom_target_dir,
      enable_batching: !!v.enable_batching,
      batch_size: v.batch_size ?? null,
      max_batch_size: v.max_batch_size ?? 8,
      timeout_minutes: v.timeout_minutes ?? 60,
    });
    if (r.ok) {
      runningRef.current = true;
      setRunning(true);
    } else {
      message.error(r.error || "启动失败");
    }
  };

  return (
    <Row gutter={16}>
      <Col span={14}>
        {installed === false && (
          <Alert
            type="warning"
            showIcon
            style={{ marginBottom: 12 }}
            message="未安装 modal 包"
            description={installMsg || "请先运行: pip install modal questionary，并执行 modal token new 完成授权。"}
          />
        )}
        <Card size="small" title="Modal 云端 GPU 推理（无本地显卡也可用）">
          <Form
            form={form}
            layout="vertical"
            size="small"
            initialValues={{
              gpu: "T4",
              model: "chickenrice",
              enable_batching: false,
              max_batch_size: 8,
              timeout_minutes: 60,
              custom_target_dir: "custom-model",
            }}
          >
            <Row gutter={12}>
              <Col span={20}>
                <Form.Item name="input_path" label="输入文件 / 文件夹">
                  <Input placeholder="音频文件或包含音频的文件夹" />
                </Form.Item>
              </Col>
              <Col span={4} style={{ display: "flex", alignItems: "flex-end", paddingBottom: 24 }}>
                <Button size="small" icon={<FolderAddOutlined />} onClick={pickInput}>
                  选择
                </Button>
              </Col>
            </Row>

            <Row gutter={12}>
              <Col span={8}>
                <Form.Item name="gpu" label="GPU 类型（推荐 T4）">
                  <Select options={GPU_CHOICES.map((g) => ({ value: g, label: g }))} />
                </Form.Item>
              </Col>
              <Col span={16}>
                <Form.Item name="model" label="模型">
                  <Select
                    options={[
                      { value: "chickenrice", label: "海南鸡（日文转中文优化）— 建议 L40S 以上" },
                      { value: "jim-ja-transcribe", label: "TransWithAI 日文转录（whisper-ja-1.5B bf16）" },
                      { value: "base", label: "基础版（whisper-base）" },
                      { value: "custom", label: "自定义 HuggingFace 模型" },
                    ]}
                  />
                </Form.Item>
              </Col>
            </Row>

            <Form.Item noStyle shouldUpdate={(a, b) => a.model !== b.model}>
              {({ getFieldValue }) =>
                getFieldValue("model") === "custom" ? (
                  <Row gutter={12}>
                    <Col span={14}>
                      <Form.Item name="custom_repo" label="HuggingFace 仓库（user/repo）">
                        <Input placeholder="例如 myuser/my-whisper-model" />
                      </Form.Item>
                    </Col>
                    <Col span={10}>
                      <Form.Item name="custom_target_dir" label="models 子目录名">
                        <Input />
                      </Form.Item>
                    </Col>
                  </Row>
                ) : null
              }
            </Form.Item>

            <Row gutter={12}>
              <Col span={8}>
                <Form.Item name="enable_batching" valuePropName="checked" label=" ">
                  <Checkbox>启用批处理加速（需更高显存）</Checkbox>
                </Form.Item>
              </Col>
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

            <Row gutter={12}>
              <Col span={8}>
                <Form.Item name="timeout_minutes" label="任务超时（分钟）">
                  <InputNumber min={1} max={600} style={{ width: "100%" }} />
                </Form.Item>
              </Col>
            </Row>

            <Space>
              <Button type="primary" icon={<CaretRightOutlined />} onClick={run} disabled={running}>
                开始云端任务
              </Button>
              <Button danger icon={<StopOutlined />} disabled={!running} onClick={async () => {
                if (!api) return;
                await api.modalCancel();
              }}>
                停止
              </Button>
            </Space>

            <Typography.Text type="secondary" style={{ fontSize: 12, display: "block", marginTop: 8 }}>
              说明：首次运行需构建云端镜像（约数分钟）；模型缓存在 Modal Volume，后续运行更快。
              新用户每月有 $30 免费额度；使用前需完成 modal token new 授权并建议在
              modal.com/secrets 配置 huggingface-secret。字幕文件将自动下载回源文件目录。
            </Typography.Text>
          </Form>
        </Card>
      </Col>
      <Col span={10}>
        <Card size="small" title="云端任务日志" style={{ height: "100%" }}>
          <LogPanel
            key={logKey}
            types={["modal_log", "modal_exit"]}
            toLine={(e) => (e.type === "modal_log" ? e.line : e.type === "modal_exit" ? `[exit] ${e.code}` : null)}
            height={560}
          />
        </Card>
      </Col>
    </Row>
  );
}
