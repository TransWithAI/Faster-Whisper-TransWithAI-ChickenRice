import React, { useEffect, useRef, useState } from "react";
import { Button, Space } from "antd";
import { onEvent, AppEvent } from "../api";

interface LogPanelProps {
  /** 订阅的事件类型 */
  types: AppEvent["type"][];
  /** 事件到一行文本的映射 */
  toLine?: (e: AppEvent) => string | null;
  /** 外部触发的清空 key 变化时清空 */
  clearSignal?: number;
  height?: number | string;
}

/** 通用滚动日志面板：订阅指定事件并展示为终端风格文本 */
export default function LogPanel({ types, toLine, clearSignal, height = 240 }: LogPanelProps) {
  const [lines, setLines] = useState<string[]>([]);
  const boxRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const off = onEvent((e) => {
      if (!types.includes(e.type)) return;
      const line = toLine ? toLine(e) : JSON.stringify(e);
      if (line === null) return;
      setLines((prev) => {
        const next = [...prev, line];
        return next.length > 2000 ? next.slice(next.length - 2000) : next;
      });
    });
    return off;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [types.join(","), toLine]);

  useEffect(() => {
    if (clearSignal !== undefined) setLines([]);
  }, [clearSignal]);

  useEffect(() => {
    if (boxRef.current) boxRef.current.scrollTop = boxRef.current.scrollHeight;
  }, [lines]);

  return (
    <div
      ref={boxRef}
      style={{
        height,
        overflow: "auto",
        background: "#141414",
        color: "#d9d9d9",
        fontFamily: "Consolas, monospace",
        fontSize: 12,
        lineHeight: "18px",
        padding: 8,
        borderRadius: 6,
        whiteSpace: "pre-wrap",
        wordBreak: "break-all",
      }}
    >
      {lines.length === 0 ? "" : lines.join("\n")}
    </div>
  );
}

/** 一键复制日志的辅助按钮组（可选使用） */
export function LogActions({ getText }: { getText: () => string }) {
  return (
    <Space style={{ marginBottom: 8 }}>
      <Button
        size="small"
        onClick={() => {
          navigator.clipboard.writeText(getText());
        }}
      >
        复制日志
      </Button>
    </Space>
  );
}
