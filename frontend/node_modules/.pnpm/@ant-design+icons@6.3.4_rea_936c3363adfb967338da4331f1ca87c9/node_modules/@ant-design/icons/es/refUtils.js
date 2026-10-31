import * as React from 'react';
function fillRef(ref, node) {
  if (typeof ref === 'function') {
    ref(node);
  } else if (typeof ref === 'object' && ref && 'current' in ref) {
    ref.current = node;
  }
}
function composeRef(...refs) {
  const refList = refs.filter(Boolean);
  if (refList.length <= 1) {
    return refList[0];
  }
  return node => {
    refs.forEach(ref => {
      fillRef(ref, node);
    });
  };
}
export function useComposeRef(...refs) {
  return React.useMemo(() => composeRef(...refs), refs);
}