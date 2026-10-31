import { createRoot } from 'react-dom/client';
const MARK = '__rc_react_root__';

// ========================== Render ==========================

export function render(node, container) {
  const root = container[MARK] || createRoot(container);
  root.render(node);
  container[MARK] = root;
}

// ========================= Unmount ==========================
export async function unmount(container) {
  // Delay to unmount to avoid React 18 sync warning
  return Promise.resolve().then(() => {
    container[MARK]?.unmount();
    delete container[MARK];
  });
}