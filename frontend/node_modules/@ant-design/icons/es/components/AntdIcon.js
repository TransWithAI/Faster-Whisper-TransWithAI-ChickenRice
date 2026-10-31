'use client';

function _extends() { _extends = Object.assign ? Object.assign.bind() : function (target) { for (var i = 1; i < arguments.length; i++) { var source = arguments[i]; for (var key in source) { if (Object.prototype.hasOwnProperty.call(source, key)) { target[key] = source[key]; } } } return target; }; return _extends.apply(this, arguments); }
import * as React from 'react';
import { clsx } from 'clsx';
import Context from "./Context";
import ReactIcon from "./IconBaseTwoTone";
import { getTwoToneColor, setTwoToneColor } from "./twoTonePrimaryColor";
import { DEFAULT_TWOTONE_COLOR } from "../colorUtils";
import { isIconDefinition, warning } from "../renderUtils";
import { normalizeTwoToneColors } from "../utils";
// Initial setting
// should move it to antd main repo?
setTwoToneColor(DEFAULT_TWOTONE_COLOR);

// https://github.com/DefinitelyTyped/DefinitelyTyped/issues/34757#issuecomment-488848720

const Icon = /*#__PURE__*/React.forwardRef((props, ref) => {
  const {
    // affect outter <i>...</i>
    className,
    // affect inner <svg>...</svg>
    icon,
    spin,
    rotate,
    tabIndex,
    onClick,
    // other
    twoToneColor,
    ...restProps
  } = props;
  const {
    prefixCls = 'anticon',
    rootClassName
  } = React.useContext(Context);
  warning(isIconDefinition(icon), `icon should be icon definiton, but got ${typeof icon}`);
  if (!isIconDefinition(icon)) {
    return null;
  }
  const classString = clsx(rootClassName, prefixCls, {
    [`${prefixCls}-${icon.name}`]: !!icon.name,
    [`${prefixCls}-spin`]: !!spin || icon.name === 'loading'
  }, className);
  let iconTabIndex = tabIndex;
  if (iconTabIndex === undefined && onClick) {
    iconTabIndex = -1;
  }
  const svgStyle = rotate ? {
    msTransform: `rotate(${rotate}deg)`,
    transform: `rotate(${rotate}deg)`
  } : undefined;
  const [primaryColor, secondaryColor] = normalizeTwoToneColors(twoToneColor);
  return /*#__PURE__*/React.createElement("span", _extends({
    role: "img",
    "aria-label": icon.name
  }, restProps, {
    ref: ref,
    tabIndex: iconTabIndex,
    onClick: onClick,
    className: classString
  }), /*#__PURE__*/React.createElement(ReactIcon, {
    icon: icon,
    primaryColor: primaryColor,
    secondaryColor: secondaryColor,
    style: svgStyle
  }));
});
Icon.getTwoToneColor = getTwoToneColor;
Icon.setTwoToneColor = setTwoToneColor;
if (process.env.NODE_ENV !== 'production') {
  Icon.displayName = 'AntdIcon';
}
export default Icon;