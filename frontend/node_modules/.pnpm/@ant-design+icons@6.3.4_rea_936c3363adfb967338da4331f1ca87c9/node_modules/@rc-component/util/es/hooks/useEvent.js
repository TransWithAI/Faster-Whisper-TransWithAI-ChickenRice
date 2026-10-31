import * as React from 'react';
const useEvent = callback => {
  const fnRef = React.useRef(callback);
  fnRef.current = callback;
  const memoFn = React.useCallback((...args) => fnRef.current?.(...args), []);
  return memoFn;
};
export default useEvent;