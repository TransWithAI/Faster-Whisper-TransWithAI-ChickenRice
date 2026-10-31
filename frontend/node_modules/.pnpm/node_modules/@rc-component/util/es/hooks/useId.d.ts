import * as React from 'react';
/** @private Note only worked in develop env. Not work in production. */
export declare function resetUuid(): void;
/**
 * Generate a valid HTML id from prefix and key.
 * Sanitizes the key by replacing invalid characters with hyphens.
 * @param prefix - The prefix for the id
 * @param key - The key from React element, may contain spaces or invalid characters
 * @returns A valid HTML id string
 */
export declare function getId(prefix: string, key: React.Key): string;
declare const _default: (id?: string) => string;
export default _default;
