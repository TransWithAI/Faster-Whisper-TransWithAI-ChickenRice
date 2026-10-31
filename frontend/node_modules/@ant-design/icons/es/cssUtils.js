const APPEND_ORDER = 'data-rc-order';
const APPEND_PRIORITY = 'data-rc-priority';
const MARK_KEY = 'rc-util-key';
const containerCache = new Map();
function canUseDom() {
  return !!(typeof window !== 'undefined' && window.document && window.document.createElement);
}
function contains(root, node) {
  if (!root || !node) {
    return false;
  }
  if (root.contains) {
    return root.contains(node);
  }
  let current = node;
  while (current) {
    if (current === root) {
      return true;
    }
    current = current.parentNode;
  }
  return false;
}
function getMark({
  mark
} = {}) {
  if (mark) {
    return mark.startsWith('data-') ? mark : `data-${mark}`;
  }
  return MARK_KEY;
}
function getContainer(option) {
  if (option.attachTo) {
    return option.attachTo;
  }
  const head = document.querySelector('head');
  return head || document.body;
}
function getOrder(prepend) {
  if (prepend === 'queue') {
    return 'prependQueue';
  }
  return prepend ? 'prepend' : 'append';
}
function findStyles(container) {
  return Array.from((containerCache.get(container) || container).children).filter(node => node.tagName === 'STYLE');
}
function injectCSS(css, option = {}) {
  if (!canUseDom()) {
    return null;
  }
  const {
    csp,
    prepend,
    priority = 0
  } = option;
  const mergedOrder = getOrder(prepend);
  const isPrependQueue = mergedOrder === 'prependQueue';
  const styleNode = document.createElement('style');
  styleNode.setAttribute(APPEND_ORDER, mergedOrder);
  if (isPrependQueue && priority) {
    styleNode.setAttribute(APPEND_PRIORITY, `${priority}`);
  }
  if (csp?.nonce) {
    styleNode.nonce = csp.nonce;
  }
  styleNode.innerHTML = css;
  const container = getContainer(option);
  const {
    firstChild
  } = container;
  if (prepend) {
    if (isPrependQueue) {
      const existStyle = (option.styles || findStyles(container)).filter(node => {
        if (!['prepend', 'prependQueue'].includes(node.getAttribute(APPEND_ORDER))) {
          return false;
        }
        const nodePriority = Number(node.getAttribute(APPEND_PRIORITY) || 0);
        return priority >= nodePriority;
      });
      if (existStyle.length) {
        container.insertBefore(styleNode, existStyle[existStyle.length - 1].nextSibling);
        return styleNode;
      }
    }
    container.insertBefore(styleNode, firstChild);
  } else {
    container.appendChild(styleNode);
  }
  return styleNode;
}
function findExistNode(key, option = {}) {
  let {
    styles
  } = option;
  styles ||= findStyles(getContainer(option));
  return styles.find(node => node.getAttribute(getMark(option)) === key);
}
function syncRealContainer(container, option) {
  const cachedRealContainer = containerCache.get(container);
  if (!cachedRealContainer || !contains(document, cachedRealContainer)) {
    const placeholderStyle = injectCSS('', option);
    if (!placeholderStyle) {
      return;
    }
    const {
      parentNode
    } = placeholderStyle;
    containerCache.set(container, parentNode);
    container.removeChild(placeholderStyle);
  }
}
export function updateCSS(css, key, originOption = {}) {
  if (!canUseDom()) {
    return null;
  }
  const container = getContainer(originOption);
  const styles = findStyles(container);
  const option = {
    ...originOption,
    styles
  };
  syncRealContainer(container, option);
  const existNode = findExistNode(key, option);
  if (existNode) {
    if (option.csp?.nonce && existNode.nonce !== option.csp.nonce) {
      existNode.nonce = option.csp.nonce;
    }
    if (existNode.innerHTML !== css) {
      existNode.innerHTML = css;
    }
    return existNode;
  }
  const newNode = injectCSS(css, option);
  newNode?.setAttribute(getMark(option), key);
  return newNode;
}
function getRoot(ele) {
  return ele?.getRootNode?.();
}
export function getShadowRoot(ele) {
  const root = getRoot(ele);
  return typeof ShadowRoot !== 'undefined' && root instanceof ShadowRoot ? root : null;
}
const warned = {};
export function warningOnce(valid, message) {
  if (valid || warned[message]) {
    return;
  }
  if (process.env.NODE_ENV !== 'production' && typeof console !== 'undefined') {
    console.error(`Warning: ${message}`);
  }
  warned[message] = true;
}