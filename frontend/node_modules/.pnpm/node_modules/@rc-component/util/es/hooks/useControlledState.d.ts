type Updater<T> = (updater: T | ((origin: T) => T)) => void;
/**
 * Similar to `useState` but will use props value if provided.
 * From React 18, we do not need safe `useState` since it will not throw for unmounted update.
 * This hooks remove the `onChange` & `postState` logic since we only need basic merged state logic.
 */
export default function useControlledState<T>(defaultStateValue: T | (() => T), value?: T): [T, Updater<T>];
export {};
