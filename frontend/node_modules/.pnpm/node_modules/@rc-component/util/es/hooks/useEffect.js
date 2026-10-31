import * as React from 'react';

/** As `React.useEffect` but pass origin value in callback and not need care deps length change. */
function useEffect(callback, deps) {
  const prevRef = React.useRef(deps);
  React.useEffect(() => {
    if (deps.length !== prevRef.current.length || deps.some((dep, index) => dep !== prevRef.current[index])) {
      callback(prevRef.current);
    }
    prevRef.current = deps;
  });
}
export default useEffect;