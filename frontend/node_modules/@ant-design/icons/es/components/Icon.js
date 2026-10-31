function _extends() { _extends = Object.assign ? Object.assign.bind() : function (target) { for (var i = 1; i < arguments.length; i++) { var source = arguments[i]; for (var key in source) { if (Object.prototype.hasOwnProperty.call(source, key)) { target[key] = source[key]; } } } return target; }; return _extends.apply(this, arguments); }
// Seems this is used for iconFont
import * as React from 'react';
import { clsx } from 'clsx';
import Context from "./Context";
import { svgBaseProps, warning, useInsertStyles } from "../renderUtils";
import { useComposeRef } from "../refUtils";
const Icon = /*#__PURE__*/React.forwardRef((props, ref) => {
  const {
    // affect outter <i>...</i>
    className,
    // affect inner <svg>...</svg>
    component: Component,
    color,
    viewBox,
    spin,
    rotate,
    tabIndex,
    onClick,
    // children
    children,
    ...restProps
  } = props;
  const iconRef = React.useRef(null);
  const mergedRef = useComposeRef(iconRef, ref);
  warning(Boolean(Component || children), 'Should have `component` prop or `children`.');
  useInsertStyles(iconRef);
  const {
    prefixCls = 'anticon',
    rootClassName
  } = React.useContext(Context);
  const classString = clsx(rootClassName, prefixCls, {
    [`${prefixCls}-spin`]: !!spin && !!Component
  }, className);
  const svgClassString = clsx({
    [`${prefixCls}-spin`]: !!spin
  });
  const svgStyle = rotate ? {
    msTransform: `rotate(${rotate}deg)`,
    transform: `rotate(${rotate}deg)`
  } : undefined;
  const innerSvgProps = {
    ...svgBaseProps,
    color,
    className: svgClassString,
    style: svgStyle,
    viewBox
  };
  if (!viewBox) {
    delete innerSvgProps.viewBox;
  }

  // component > children
  const renderInnerNode = () => {
    if (Component) {
      return /*#__PURE__*/React.createElement(Component, innerSvgProps, children);
    }
    if (children) {
      warning(Boolean(viewBox) || React.Children.count(children) === 1 && /*#__PURE__*/React.isValidElement(children) && React.Children.only(children).type === 'use', 'Make sure that you provide correct `viewBox`' + ' prop (default `0 0 1024 1024`) to the icon.');
      return /*#__PURE__*/React.createElement("svg", _extends({}, innerSvgProps, {
        viewBox: viewBox
      }), children);
    }
    return null;
  };
  let iconTabIndex = tabIndex;
  if (iconTabIndex === undefined && onClick) {
    iconTabIndex = -1;
  }
  return /*#__PURE__*/React.createElement("span", _extends({
    role: "img"
  }, restProps, {
    ref: mergedRef,
    tabIndex: iconTabIndex,
    onClick: onClick,
    className: classString
  }), renderInnerNode());
});
if (process.env.NODE_ENV !== 'production') {
  Icon.displayName = 'AntdIcon';
}
export default Icon;