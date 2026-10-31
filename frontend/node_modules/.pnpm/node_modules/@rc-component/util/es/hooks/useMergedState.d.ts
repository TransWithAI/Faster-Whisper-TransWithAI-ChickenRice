type Updater<T> = (updater: T | ((origin: T) => T), ignoreDestroy?: boolean) => void;
/**
 * @deprecated Please use `useControlledState` instead if not need support < React 18.
 * Similar to `useState` but will use props value if provided.
 * Note that internal use rc-util `useState` hook.
 */
export default function useMergedState<T, R = T>(defaultStateValue: T | (() => T), option?: {
    defaultValue?: T | (() => T);
    value?: T;
    onChange?: (value: T, prevValue: T) => void;
    postState?: (value: T) => T;
}): [R, Updater<T>];
export {};
