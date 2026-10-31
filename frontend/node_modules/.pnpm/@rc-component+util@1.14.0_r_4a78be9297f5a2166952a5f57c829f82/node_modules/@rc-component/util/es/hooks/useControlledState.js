import { useState } from 'react';
import useLayoutEffect from "./useLayoutEffect";
/**
 * Similar to `useState` but will use props value if provided.
 * From React 18, we do not need safe `useState` since it will not throw for unmounted update.
 * This hooks remove the `onChange` & `postState` logic since we only need basic merged state logic.
 */
export default function useControlledState(defaultStateValue, value) {
  const [innerValue, setInnerValue] = useState(defaultStateValue);
  const mergedValue = value !== undefined ? value : innerValue;
  useLayoutEffect(mount => {
    if (!mount) {
      setInnerValue(value);
    }
  }, [value]);
  return [
  // Value
  mergedValue,
  // Update function
  setInnerValue];
}