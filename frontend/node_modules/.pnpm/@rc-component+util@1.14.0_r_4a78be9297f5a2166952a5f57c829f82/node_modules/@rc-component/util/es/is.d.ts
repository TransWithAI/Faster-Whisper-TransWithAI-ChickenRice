/**
 * Check if a value is neither `null` nor `undefined`.
 */
export declare function isNonNullable<T>(value: T): value is NonNullable<T>;
/**
 * Check whether a value should be treated as renderable content.
 *
 * Returns `false` only for `null`, `undefined`, `false`, and `''`; all other
 * values, including `0` and `true`, are treated as renderable.
 *
 * This is a compatibility-oriented presence check, not a complete React node
 * validator.
 */
export declare function isReactRenderable<T>(value: T): value is Exclude<NonNullable<T>, false | ''>;
