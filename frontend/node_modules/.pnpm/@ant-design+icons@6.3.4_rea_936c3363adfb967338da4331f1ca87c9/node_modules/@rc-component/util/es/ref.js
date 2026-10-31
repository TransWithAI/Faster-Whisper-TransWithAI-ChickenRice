import { isValidElement, version } from 'react';
import { ForwardRef, isMemo } from 'react-is';
import useMemo from "./hooks/useMemo";
import isFragment from "./React/isFragment";
const ReactMajorVersion = Number(version?.split('.')[0]);
export const fillRef = (ref, node) => {
  if (typeof ref === 'function') {
    ref(node);
  } else if (typeof ref === 'object' && ref && 'current' in ref) {
    ref.current = node;
  }
};

/**
 * Merge refs into one ref function to support ref passing.
 */
export const composeRef = (...refs) => {
  const refList = refs.filter(Boolean);
  if (refList.length <= 1) {
    return refList[0];
  }
  return node => {
    refs.forEach(ref => {
      fillRef(ref, node);
    });
  };
};
export const useComposeRef = (...refs) => {
  return useMemo(() => composeRef(...refs),
  // eslint-disable-next-line react-hooks/exhaustive-deps
  refs, (prev, next) => prev.length !== next.length || prev.every((ref, i) => ref !== next[i]));
};
export const supportRef = nodeOrComponent => {
  if (!nodeOrComponent) {
    return false;
  }

  // React 19 no need `forwardRef` anymore. So just pass if is a React element.
  if (isReactElement(nodeOrComponent) && ReactMajorVersion >= 19) {
    return true;
  }
  const type = isMemo(nodeOrComponent) ? nodeOrComponent.type.type : nodeOrComponent.type;

  // Function component node
  if (typeof type === 'function' && !type.prototype?.render && type.$$typeof !== ForwardRef) {
    return false;
  }

  // Class component
  if (typeof nodeOrComponent === 'function' && !nodeOrComponent.prototype?.render && nodeOrComponent.$$typeof !== ForwardRef) {
    return false;
  }
  return true;
};
function isReactElement(node) {
  return /*#__PURE__*/isValidElement(node) && !isFragment(node);
}
export const supportNodeRef = node => {
  return isReactElement(node) && supportRef(node);
};

/**
 * In React 19. `ref` is not a property from node.
 * But a property from `props.ref`.
 * To check if `props.ref` exist or fallback to `ref`.
 */
export const getNodeRef = node => {
  if (node && isReactElement(node)) {
    const ele = node;

    // Source from:
    // https://github.com/mui/material-ui/blob/master/packages/mui-utils/src/getReactNodeRef/getReactNodeRef.ts
    return ele.props.propertyIsEnumerable('ref') ? ele.props.ref : ele.ref;
  }
  return null;
};