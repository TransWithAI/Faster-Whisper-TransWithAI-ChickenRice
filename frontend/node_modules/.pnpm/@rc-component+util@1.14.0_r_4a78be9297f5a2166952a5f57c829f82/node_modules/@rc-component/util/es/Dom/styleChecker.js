import canUseDom from "./canUseDom";
const isStyleNameSupport = styleName => {
  if (canUseDom() && window.document.documentElement) {
    const styleNameList = Array.isArray(styleName) ? styleName : [styleName];
    const {
      documentElement
    } = window.document;
    return styleNameList.some(name => name in documentElement.style);
  }
  return false;
};
const isStyleValueSupport = (styleName, value) => {
  if (!isStyleNameSupport(styleName)) {
    return false;
  }
  const ele = document.createElement('div');
  const origin = ele.style[styleName];
  ele.style[styleName] = value;
  return ele.style[styleName] !== origin;
};
export function isStyleSupport(styleName, styleValue) {
  if (!Array.isArray(styleName) && styleValue !== undefined) {
    return isStyleValueSupport(styleName, styleValue);
  }
  return isStyleNameSupport(styleName);
}