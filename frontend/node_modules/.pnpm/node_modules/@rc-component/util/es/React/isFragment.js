const REACT_ELEMENT_TYPE_18 = Symbol.for('react.element');
const REACT_ELEMENT_TYPE_19 = Symbol.for('react.transitional.element');
const REACT_FRAGMENT_TYPE = Symbol.for('react.fragment');

/**
 * Compatible with React 18 or 19 to check if node is a Fragment.
 */
export default function isFragment(object) {
  return (
    // Base object type
    object && typeof object === 'object' && (
    // React Element type
    object.$$typeof === REACT_ELEMENT_TYPE_18 || object.$$typeof === REACT_ELEMENT_TYPE_19) &&
    // React Fragment type
    object.type === REACT_FRAGMENT_TYPE
  );
}