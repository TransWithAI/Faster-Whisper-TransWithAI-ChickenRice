import * as React from 'react';
/**
 * Same as React.useState but `setState` accept `ignoreDestroy` param to not to setState after destroyed.
 * We do not make this auto is to avoid real memory leak.
 * Developer should confirm it's safe to ignore themselves.
 */
const useSafeState = defaultValue => {
  const destroyRef = React.useRef(false);
  const [value, setValue] = React.useState(defaultValue);
  React.useEffect(() => {
    destroyRef.current = false;
    return () => {
      destroyRef.current = true;
    };
  }, []);
  function safeSetState(updater, ignoreDestroy) {
    if (ignoreDestroy && destroyRef.current) {
      return;
    }
    setValue(updater);
  }
  return [value, safeSetState];
};
export default useSafeState;