import * as React from 'react';
function getUseId() {
  // We need fully clone React function here to avoid webpack warning React 17 do not export `useId`
  const fullClone = {
    ...React
  };
  return fullClone.useId;
}
let uuid = 0;

/** @private Note only worked in develop env. Not work in production. */
export function resetUuid() {
  if (process.env.NODE_ENV !== 'production') {
    uuid = 0;
  }
}

/**
 * Generate a valid HTML id from prefix and key.
 * Sanitizes the key by replacing invalid characters with hyphens.
 * @param prefix - The prefix for the id
 * @param key - The key from React element, may contain spaces or invalid characters
 * @returns A valid HTML id string
 */
export function getId(prefix, key) {
  // React.Key can be string | number, convert to string first
  const keyStr = String(key);

  // Valid id characters: letters, digits, hyphen, underscore, colon, period
  // Replace all invalid characters (including spaces) with hyphens to preserve length
  const sanitizedKey = keyStr.replace(/[^a-zA-Z0-9_.:-]/g, '-');
  return `${prefix}-${sanitizedKey}`;
}
const useOriginId = getUseId();
export default useOriginId ?
// Use React `useId`
function useId(id) {
  const reactId = useOriginId();

  // Developer passed id is single source of truth
  if (id) {
    return id;
  }

  // Test env always return mock id
  if (process.env.NODE_ENV === 'test') {
    return 'test-id';
  }
  return reactId;
} :
// Use compatible of `useId`
function useCompatId(id) {
  // Inner id for accessibility usage. Only work in client side
  const [innerId, setInnerId] = React.useState('ssr-id');
  React.useEffect(() => {
    const nextId = uuid;
    uuid += 1;
    setInnerId(`rc_unique_${nextId}`);
  }, []);

  // Developer passed id is single source of truth
  if (id) {
    return id;
  }

  // Test env always return mock id
  if (process.env.NODE_ENV === 'test') {
    return 'test-id';
  }

  // Return react native id or inner id
  return innerId;
};