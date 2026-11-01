import React from "react";
import ReactDOM from "react-dom/client";
import { ConfigProvider, theme } from "antd";
import zhCN from "antd/locale/zh_CN";
import enUS from "antd/locale/en_US";
import App from "./App";
import { I18nProvider, useI18n } from "./i18n";

function Root() {
  const { lang } = useI18n();
  return (
    <ConfigProvider
      locale={lang === "zh-CN" ? zhCN : enUS}
      theme={{ algorithm: theme.defaultAlgorithm }}
    >
      <App />
    </ConfigProvider>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <I18nProvider>
      <Root />
    </I18nProvider>
  </React.StrictMode>
);
