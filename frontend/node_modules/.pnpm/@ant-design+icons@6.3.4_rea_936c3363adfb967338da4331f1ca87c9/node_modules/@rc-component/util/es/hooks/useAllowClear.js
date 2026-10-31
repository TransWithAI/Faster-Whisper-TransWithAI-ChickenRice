import * as React from 'react';
/** Normalize clear configuration. Components should apply their defaults first. */
export default function useAllowClear(allowClear) {
  return React.useMemo(() => allowClear && typeof allowClear === 'object' ? {
    ...allowClear,
    disabled: allowClear.disabled ?? false
  } : {
    disabled: !allowClear
  }, [allowClear]);
}