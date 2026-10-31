import * as React from 'react';
import { getSecondaryColor } from "../colorUtils";
import { generate, isIconDefinition, warning, useInsertStyles } from "../renderUtils";
const twoToneColorPalette = {
  primaryColor: '#333',
  secondaryColor: '#E6E6E6',
  calculated: false
};
function setTwoToneColors({
  primaryColor,
  secondaryColor
}) {
  twoToneColorPalette.primaryColor = primaryColor;
  twoToneColorPalette.secondaryColor = secondaryColor || getSecondaryColor(primaryColor);
  twoToneColorPalette.calculated = !!secondaryColor;
}
function getTwoToneColors() {
  return {
    ...twoToneColorPalette
  };
}
const IconBaseTwoTone = props => {
  const {
    icon,
    className,
    onClick,
    style,
    primaryColor,
    secondaryColor,
    ...restProps
  } = props;
  const svgRef = React.useRef(null);
  let colors = twoToneColorPalette;
  if (primaryColor) {
    colors = {
      primaryColor,
      secondaryColor: secondaryColor || getSecondaryColor(primaryColor)
    };
  }
  useInsertStyles(svgRef);
  warning(isIconDefinition(icon), `icon should be icon definiton, but got ${typeof icon}`);
  if (!isIconDefinition(icon)) {
    return null;
  }
  let target = icon;
  if (target && typeof target.icon === 'function') {
    target = {
      ...target,
      icon: target.icon(colors.primaryColor, colors.secondaryColor)
    };
  }
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
IconBaseTwoTone.displayName = 'IconReact';
IconBaseTwoTone.getTwoToneColors = getTwoToneColors;
IconBaseTwoTone.setTwoToneColors = setTwoToneColors;
export default IconBaseTwoTone;