function getRoot(ele) {
  return ele?.getRootNode?.();
}

/**
 * Check if is in shadowRoot
 */
export function inShadow(ele) {
  return getRoot(ele) instanceof ShadowRoot;
}

/**
 * Return shadowRoot if possible
 */
export function getShadowRoot(ele) {
  return inShadow(ele) ? getRoot(ele) : null;
}