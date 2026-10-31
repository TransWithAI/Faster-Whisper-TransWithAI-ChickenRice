import { useEffect, useRef, useState } from 'react';
import isVisible from "./isVisible";
import useId from "../hooks/useId";
function focusable(node, includePositive = false) {
  if (isVisible(node)) {
    const nodeName = node.nodeName.toLowerCase();
    const isFocusableElement =
    // Focusable element
    ['input', 'select', 'textarea', 'button'].includes(nodeName) ||
    // Editable element
    node.isContentEditable ||
    // Anchor with href element
    nodeName === 'a' && !!node.getAttribute('href');

    // Get tabIndex
    const tabIndexAttr = node.getAttribute('tabindex');
    const tabIndexNum = Number(tabIndexAttr);

    // Parse as number if validate
    let tabIndex = null;
    if (tabIndexAttr && !Number.isNaN(tabIndexNum)) {
      tabIndex = tabIndexNum;
    } else if (isFocusableElement && tabIndex === null) {
      tabIndex = 0;
    }

    // Block focusable if disabled
    if (isFocusableElement && node.disabled) {
      tabIndex = null;
    }
    return tabIndex !== null && (tabIndex >= 0 || includePositive && tabIndex < 0);
  }
  return false;
}
export function getFocusNodeList(node, includePositive = false) {
  const res = [...node.querySelectorAll('*')].filter(child => {
    return focusable(child, includePositive);
  });
  if (focusable(node, includePositive)) {
    res.unshift(node);
  }
  return res;
}
// Used for `rc-input` `rc-textarea` `rc-input-number`
/**
 * Focus element and set cursor position for input/textarea elements.
 */
export function triggerFocus(element, option) {
  if (!element) return;
  element.focus(option);

  // Selection content
  const {
    cursor
  } = option || {};
  if (cursor && (element instanceof HTMLInputElement || element instanceof HTMLTextAreaElement)) {
    const len = element.value.length;
    switch (cursor) {
      case 'start':
        element.setSelectionRange(0, 0);
        break;
      case 'end':
        element.setSelectionRange(len, len);
        break;
      default:
        element.setSelectionRange(0, len);
    }
  }
}

// ======================================================
// ==                    Lock Focus                    ==
// ======================================================
let lastFocusElement = null;
let focusElements = [];
// Map stable ID to lock element
const idToElementMap = new Map();
// Map stable ID to ignored element
const ignoredElementMap = new Map();
function getLastElement() {
  return focusElements[focusElements.length - 1];
}
function isIgnoredElement(element) {
  const lastElement = getLastElement();
  if (element && lastElement) {
    // Find the ID that maps to the last element
    let lockId;
    for (const [id, ele] of idToElementMap.entries()) {
      if (ele === lastElement) {
        lockId = id;
        break;
      }
    }
    const ignoredEle = ignoredElementMap.get(lockId);
    return !!ignoredEle && (ignoredEle === element || ignoredEle.contains(element));
  }
  return false;
}
function hasFocus(element) {
  const {
    activeElement
  } = document;
  return element === activeElement || element.contains(activeElement);
}
function syncFocus() {
  const lastElement = getLastElement();
  const {
    activeElement
  } = document;

  // If current focus is on an ignored element, don't force it back
  if (isIgnoredElement(activeElement)) {
    return;
  }
  if (lastElement && !hasFocus(lastElement)) {
    const focusableList = getFocusNodeList(lastElement);
    const matchElement = focusableList.includes(lastFocusElement) ? lastFocusElement : focusableList[0];
    matchElement?.focus({
      preventScroll: true
    });
  } else {
    lastFocusElement = activeElement;
  }
}
function onWindowKeyDown(e) {
  if (e.key === 'Tab') {
    const {
      activeElement
    } = document;
    const lastElement = getLastElement();
    const focusableList = getFocusNodeList(lastElement);
    const last = focusableList[focusableList.length - 1];
    if (e.shiftKey && activeElement === focusableList[0]) {
      // Tab backward on first focusable element
      lastFocusElement = last;
    } else if (!e.shiftKey && activeElement === last) {
      // Tab forward on last focusable element
      lastFocusElement = focusableList[0];
    }
  }
}

/**
 * Lock focus in the element.
 * It will force back to the first focusable element when focus leaves the element.
 * @param id - A stable ID for this lock instance
 */
export function lockFocus(element, id) {
  if (element) {
    // Store the mapping between ID and element
    idToElementMap.set(id, element);

    // Refresh focus elements
    focusElements = focusElements.filter(ele => ele !== element);
    focusElements.push(element);

    // Just add event since it will de-duplicate
    window.addEventListener('focusin', syncFocus);
    window.addEventListener('keydown', onWindowKeyDown, true);
    syncFocus();
  }

  // Always return unregister function
  return () => {
    lastFocusElement = null;
    focusElements = focusElements.filter(ele => ele !== element);
    idToElementMap.delete(id);
    ignoredElementMap.delete(id);
    if (focusElements.length === 0) {
      window.removeEventListener('focusin', syncFocus);
      window.removeEventListener('keydown', onWindowKeyDown, true);
    }
  };
}

/**
 * Retry an effect until it reports ready.
 * When `ready` is `false`, it will schedule one more effect cycle and call `func` again
 * with the next `retryTimes`.
 */

function useRetryEffect(func, deps) {
  /* eslint-disable react-hooks/exhaustive-deps */
  const retryTimesRef = useRef(0);
  const [retryMark, setRetryMark] = useState(0);
  useEffect(() => {
    retryTimesRef.current = 0;
  }, deps);
  useEffect(() => {
    const [clearFn, ready] = func(retryTimesRef.current);
    if (!ready) {
      retryTimesRef.current += 1;
      setRetryMark(count => count + 1);
    }
    return clearFn;
  }, [...deps, retryMark]);
  /* eslint-enable react-hooks/exhaustive-deps */
}

/**
 * Lock focus within an element.
 * When locked, focus will be restricted to focusable elements within the specified element.
 * If multiple elements are locked, only the last locked element will be effective.
 * @returns A function to mark an element as ignored, which will temporarily allow focus on that element even if it's outside the locked area.
 */
export function useLockFocus(lock, getElement) {
  const id = useId();
  const getElementRef = useRef(getElement);
  getElementRef.current = getElement;
  const lockEffect = retryTimes => {
    if (!lock) {
      return [undefined, true];
    }
    const element = getElementRef.current();
    if (element) {
      return [lockFocus(element, id), true];
    }
    return [undefined, retryTimes >= 1];
  };
  useRetryEffect(lockEffect, [id, lock]);
  const ignoreElement = ele => {
    if (ele) {
      // Set the ignored element using stable ID
      ignoredElementMap.set(id, ele);
    }
  };
  return [ignoreElement];
}