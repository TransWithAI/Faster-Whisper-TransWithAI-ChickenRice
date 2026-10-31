import * as React from 'react';
import raf from "../raf";
import useEvent from "./useEvent";
/**
 * Similar to `useState`, but updates on the next frame by default.
 * Pending updates are always replaced by the latest one.
 */
export default function useDelayState(defaultValue) {
  const [value, setValue] = React.useState(defaultValue);
  const delayRef = React.useRef(null);
  const cancelPending = useEvent(() => {
    if (delayRef.current) {
      const [isRaf, delay] = delayRef.current;
      if (isRaf) {
        raf.cancel(delay);
      } else {
        clearTimeout(delay);
      }
      delayRef.current = null;
    }
  });
  React.useEffect(() => cancelPending, [cancelPending]);
  const setDelayValue = useEvent((nextValue, immediatelyOrDelay) => {
    const delayConfig = immediatelyOrDelay || {
      frame: 1
    };
    cancelPending();
    if (delayConfig === true) {
      setValue(nextValue);
    } else if ('ms' in delayConfig) {
      delayRef.current = [false, setTimeout(() => setValue(nextValue), delayConfig.ms)];
    } else {
      delayRef.current = [true, raf(() => setValue(nextValue), delayConfig.frame)];
    }
  });
  return [value, setDelayValue];
}