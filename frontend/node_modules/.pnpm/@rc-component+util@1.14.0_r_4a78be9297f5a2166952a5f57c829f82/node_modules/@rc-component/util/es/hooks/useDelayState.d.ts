import * as React from 'react';
export type DelayConfig = {
    frame: number;
    ms?: never;
} | {
    frame?: never;
    ms: number;
};
export type SetDelayState<T> = (nextValue: React.SetStateAction<T>, 
/** `true` updates immediately. `false` delays the update by one frame. */
immediatelyOrDelay?: boolean | DelayConfig) => void;
/**
 * Similar to `useState`, but updates on the next frame by default.
 * Pending updates are always replaced by the latest one.
 */
export default function useDelayState<T>(defaultValue: T | (() => T)): [T, SetDelayState<T>];
