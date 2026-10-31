/* eslint-disable no-param-reassign */
import { removeCSS, updateCSS } from "./Dom/dynamicCSS";
let cached;
function measureScrollbarSize(ele) {
  const randomId = `rc-scrollbar-measure-${Math.random().toString(36).substring(7)}`;
  const measureEle = document.createElement('div');
  measureEle.id = randomId;

  // Create Style
  const measureStyle = measureEle.style;
  measureStyle.position = 'absolute';
  measureStyle.left = '0';
  measureStyle.top = '0';
  measureStyle.width = '100px';
  measureStyle.height = '100px';
  measureStyle.overflow = 'scroll';

  // Clone Style if needed
  let fallbackWidth;
  let fallbackHeight;
  if (ele) {
    const targetStyle = getComputedStyle(ele);
    measureStyle.scrollbarColor = targetStyle.scrollbarColor;
    measureStyle.scrollbarWidth = targetStyle.scrollbarWidth;

    // Set Webkit style
    const webkitScrollbarStyle = getComputedStyle(ele, '::-webkit-scrollbar');
    const width = parseInt(webkitScrollbarStyle.width, 10);
    const height = parseInt(webkitScrollbarStyle.height, 10);

    // Try wrap to handle CSP case
    try {
      const widthStyle = width ? `width: ${webkitScrollbarStyle.width};` : '';
      const heightStyle = height ? `height: ${webkitScrollbarStyle.height};` : '';
      updateCSS(`
#${randomId}::-webkit-scrollbar {
${widthStyle}
${heightStyle}
}`, randomId);
    } catch (e) {
      // Can't wrap, just log error
      console.error(e);

      // Get from style directly
      fallbackWidth = width;
      fallbackHeight = height;
    }
  }
  document.body.appendChild(measureEle);

  // Measure. Get fallback style if provided
  const scrollWidth = ele && fallbackWidth && !Number.isNaN(fallbackWidth) ? fallbackWidth : measureEle.offsetWidth - measureEle.clientWidth;
  const scrollHeight = ele && fallbackHeight && !Number.isNaN(fallbackHeight) ? fallbackHeight : measureEle.offsetHeight - measureEle.clientHeight;

  // Clean up
  document.body.removeChild(measureEle);
  removeCSS(randomId);
  return {
    width: scrollWidth,
    height: scrollHeight
  };
}
export default function getScrollBarSize(fresh) {
  if (typeof document === 'undefined') {
    return 0;
  }
  if (fresh || cached === undefined) {
    cached = measureScrollbarSize();
  }
  return cached.width;
}
export function getTargetScrollBarSize(target) {
  if (typeof document === 'undefined' || !target || !(target instanceof Element)) {
    return {
      width: 0,
      height: 0
    };
  }
  return measureScrollbarSize(target);
}