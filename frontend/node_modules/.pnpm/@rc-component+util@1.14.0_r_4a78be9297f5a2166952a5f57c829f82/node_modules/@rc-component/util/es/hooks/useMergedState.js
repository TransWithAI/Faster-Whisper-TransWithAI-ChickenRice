import useEvent from "./useEvent";
import { useLayoutUpdateEffect } from "./useLayoutEffect";
import useState from "./useState";
/** We only think `undefined` is empty */
function hasValue(value) {
  return value !== undefined;
}

/**
 * @deprecated Please use `useControlledState` instead if not need support < React 18.
 * Similar to `useState` but will use props value if provided.
 * Note that internal use rc-util `useState` hook.
 */
export default function useMergedState(defaultStateValue, option) {
  const {
    defaultValue,
    value,
    onChange,
    postState
  } = option || {};

  // ======================= Init =======================
  const [innerValue, setInnerValue] = useState(() => {
    if (hasValue(value)) {
      return value;
    } else if (hasValue(defaultValue)) {
      return typeof defaultValue === 'function' ? defaultValue() : defaultValue;
    } else {
      return typeof defaultStateValue === 'function' ? defaultStateValue() : defaultStateValue;
    }
  });
  const mergedValue = value !== undefined ? value : innerValue;
  const postMergedValue = postState ? postState(mergedValue) : mergedValue;

  // ====================== Change ======================
  const onChangeFn = useEvent(onChange);
  const [prevValue, setPrevValue] = useState([mergedValue]);
  useLayoutUpdateEffect(() => {
    const prev = prevValue[0];
    if (innerValue !== prev) {
      onChangeFn(innerValue, prev);
    }
  }, [prevValue]);

  // Sync value back to `undefined` when it from control to un-control
  useLayoutUpdateEffect(() => {
    if (!hasValue(value)) {
      setInnerValue(value);
    }
  }, [value]);

  // ====================== Update ======================
  const triggerChange = useEvent((updater, ignoreDestroy) => {
    setInnerValue(updater, ignoreDestroy);
    setPrevValue([mergedValue], ignoreDestroy);
  });
  return [postMergedValue, triggerChange];
}