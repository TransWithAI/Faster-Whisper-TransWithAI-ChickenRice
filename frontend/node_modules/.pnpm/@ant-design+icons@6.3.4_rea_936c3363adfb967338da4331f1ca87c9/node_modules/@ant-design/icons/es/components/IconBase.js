import * as React from 'react';
import { generate, isIconDefinition, warning, useInsertStyles } from "../renderUtils";
const IconBase = props => {
  const {
    icon,
    className,
    onClick,
    style,
    primaryColor: _primaryColor,
    secondaryColor: _secondaryColor,
    ...restProps
  } = props;
  const svgRef = React.useRef(null);
  useInsertStyles(svgRef);
  warning(isIconDefinition(icon), `icon should be icon definiton, but got ${typeof icon}`);
  if (!isIconDefinition(icon)) {
    return null;
  }
  const target = icon;
  return generate(target.icon, `svg-${target.name}`, {
    className,
    onClick,
    style,
    'data-icon': target.name,
    width: '1em',
    height: '1em',
    fill: 'currentColor',
    'aria-hidden': 'true',
    ...restProps,
    ref: svgRef
  });
};
IconBase.displayName = 'IconReact';
export default IconBase;