import * as React from 'react';
import useEvent from "./useEvent";
/**
 * Same as React.useState but will always get latest state.
 * This is useful when React merge multiple state updates into one.
 * e.g. onTransitionEnd trigger multiple event at once will be merged state update in React.
 */
function useSyncState(defaultValue) {
  const [, forceUpdate] = React.useReducer(x => x + 1, 0);
  const currentValueRef = React.useRef(defaultValue);
  const getValue = useEvent(() => {
    return currentValueRef.current;
  });
  const setValue = useEvent(updater => {
    currentValueRef.current = typeof updater === 'function' ? updater(currentValueRef.current) : updater;
    forceUpdate();
  });
  return [getValue, setValue];
}
export default useSyncState;