function composeProps(originProps, patchProps, isAll) {
  const composedProps = {
    ...originProps,
    ...(isAll ? patchProps : {})
  };
  Object.keys(patchProps).forEach(key => {
    const func = patchProps[key];
    if (typeof func === 'function') {
      composedProps[key] = (...args) => {
        func(...args);
        return originProps[key]?.(...args);
      };
    }
  });
  return composedProps;
}
export default composeProps;