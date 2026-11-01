import React, { createContext, useContext, useState, useCallback } from "react";

export type Lang = "zh-CN" | "en-US";

const dict = {
  "zh-CN": {
    appName: "Faster Whisper TransWithAI",
    tabTask: "转录 / 翻译",
    tabModels: "模型管理",
    tabModal: "云端推理 (Modal)",
    tabConfig: "生成参数",
    tabEnv: "环境诊断",
    tabAbout: "关于",
    langLabel: "English",
  },
  "en-US": {
    appName: "Faster Whisper TransWithAI",
    tabTask: "Transcribe / Translate",
    tabModels: "Models",
    tabModal: "Cloud (Modal)",
    tabConfig: "Generation Config",
    tabEnv: "Diagnostics",
    tabAbout: "About",
    langLabel: "中文",
  },
} as const;

export type DictKey = keyof (typeof dict)["zh-CN"];

const I18nContext = createContext<{
  lang: Lang;
  t: (k: DictKey) => string;
  toggle: () => void;
}>({ lang: "zh-CN", t: (k) => dict["zh-CN"][k], toggle: () => {} });

export function I18nProvider({ children }: { children: React.ReactNode }) {
  const [lang, setLang] = useState<Lang>("zh-CN");
  const t = useCallback((k: DictKey) => dict[lang][k] ?? k, [lang]);
  const toggle = useCallback(
    () => setLang((l) => (l === "zh-CN" ? "en-US" : "zh-CN")),
    []
  );
  return (
    <I18nContext.Provider value={{ lang, t, toggle }}>
      {children}
    </I18nContext.Provider>
  );
}

export function useI18n() {
  return useContext(I18nContext);
}
