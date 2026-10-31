import React, { useState } from "react";
import { ConfigProvider, Tabs, Button, Space, Typography } from "antd";
import { useI18n } from "./i18n";
import TaskPage from "./pages/TaskPage";
import ModelsPage from "./pages/ModelsPage";
import ModalPage from "./pages/ModalPage";
import ConfigPage from "./pages/ConfigPage";
import EnvPage from "./pages/EnvPage";
import AboutPage from "./pages/AboutPage";

export default function App() {
  const { t, toggle } = useI18n();
  const [active, setActive] = useState("task");

  const items = [
    { key: "task", label: t("tabTask"), children: <TaskPage /> },
    { key: "models", label: t("tabModels"), children: <ModelsPage /> },
    { key: "modal", label: t("tabModal"), children: <ModalPage /> },
    { key: "config", label: t("tabConfig"), children: <ConfigPage /> },
    { key: "env", label: t("tabEnv"), children: <EnvPage /> },
    { key: "about", label: t("tabAbout"), children: <AboutPage /> },
  ];

  return (
    <div style={{ padding: 16, height: "100vh", display: "flex", flexDirection: "column" }}>
      <div style={{ display: "flex", alignItems: "center", marginBottom: 8 }}>
        <Typography.Title level={4} style={{ margin: 0, flex: 1 }}>
          🎙️ {t("appName")}
        </Typography.Title>
        <Button size="small" onClick={toggle}>
          {t("langLabel")}
        </Button>
      </div>
      <div style={{ flex: 1, minHeight: 0 }}>
        <Tabs
          activeKey={active}
          onChange={setActive}
          items={items}
          style={{ height: "100%" }}
        />
      </div>
    </div>
  );
}
