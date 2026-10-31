/**
 * Merges multiple props objects into one. Unlike `Object.assign()` or `{ ...a, ...b }`, it skips
 * properties whose value is explicitly set to `undefined`.
 */
declare function mergeProps<A, B>(a: A, b: B): B & A;
declare function mergeProps<A, B, C>(a: A, b: B, c: C): C & B & A;
declare function mergeProps<A, B, C, D>(a: A, b: B, c: C, d: D): D & C & B & A;
export default mergeProps;
