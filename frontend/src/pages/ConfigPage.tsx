import React, { useEffect, useState } from "react";
import { Alert, Button, Card, Space, Typography, message } from "antd";
import { SaveOutlined, UndoOutlined } from "@ant-design/icons";
import { getApi } from "../api";

export default function ConfigPage() {
  const [api, setApi] = useState<ReturnType<typeof getApi> | null>(null);
  const [raw, setRaw] = useState("");
  const [path, setPath] = useState("");
  const [dirty, setDirty] = useState(false);
  const [loading, setLoading] = useState(true);

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
    api.getGenerationConfig().then((r) => {
      if (r.ok) {
        setRaw(r.raw);
        setPath(r.path);
      } else {
        message.error(r.error || "读取失败");
      }
      setLoading(false);
    });
  }, [api]);

  const save = async () => {
    if (!api) return;
    const r = await api.saveGenerationConfig(raw);
    if (r.ok) {
      message.success("已保存");
      setDirty(false);
    } else {
      message.error(r.error || "保存失败");
    }
  };

  return (
    <Card
      size="small"
      title={`生成参数配置（${path || "generation_config.json5"}）`}
      loading={loading}
      extra={
        <Space>
          <Button
            size="small"
            icon={<UndoOutlined />}
            disabled={!dirty}
            onClick={async () => {
              if (!api) return;
              const r = await api.getGenerationConfig();
              if (r.ok) {
                setRaw(r.raw);
                setDirty(false);
              }
            }}
          >
            撤销修改
          </Button>
          <Button type="primary" size="small" icon={<SaveOutlined />} disabled={!dirty} onClick={save}>
            保存
          </Button>
        </Space>
      }
    >
      <Alert
        type="info"
        showIcon
        style={{ marginBottom: 8 }}
        message="支持 JSON5 语法（含注释）。task 字段为 translate=翻译 / transcribe=转录；任务页的参数会以命令行覆盖优先。"
      />
      <Typography.Text type="secondary" style={{ fontSize: 12 }}>
        常用调优：声音过小导致漏翻 → 调低 vad_parameters.threshold；字幕长时间不消失 → 调低
        segment_merge.max_duration_ms；出现幻听 → 调高 threshold。
      </Typography.Text>
      <textarea
        value={raw}
        spellCheck={false}
        onChange={(e) => {
          setRaw(e.target.value);
          setDirty(true);
        }}
        style={{
          width: "100%",
          height: 480,
          marginTop: 8,
          fontFamily: "Consolas, monospace",
          fontSize: 13,
          lineHeight: "20px",
          padding: 10,
          borderRadius: 6,
          border: "1px solid #444",
          background: "#141414",
          color: "#d9d9d9",
          resize: "vertical",
        }}
      />
    </Card>
  );
}
