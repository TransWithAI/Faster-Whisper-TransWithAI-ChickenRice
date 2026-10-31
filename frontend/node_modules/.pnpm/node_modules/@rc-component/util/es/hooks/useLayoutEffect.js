import * as React from 'react';
import canUseDom from "../Dom/canUseDom";

/**
 * Wrap `React.useLayoutEffect` which will not throw warning message in test env
 */
const useInternalLayoutEffect = process.env.NODE_ENV !== 'test' && canUseDom() ? React.useLayoutEffect : React.useEffect;
const useLayoutEffect = (callback, deps) => {
  const firstMountRef = React.useRef(true);
  useInternalLayoutEffect(() => {
    return callback(firstMountRef.current);
  }, deps);

  // We tell react that first mount has passed
  useInternalLayoutEffect(() => {
    firstMountRef.current = false;
    return () => {
      firstMountRef.current = true;
    };
  }, []);
};
export const useLayoutUpdateEffect = (callback, deps) => {
  useLayoutEffect(firstMount => {
    if (!firstMount) {
      return callback();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
};
export default useLayoutEffect;