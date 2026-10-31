import ReactIcon from "./IconBaseTwoTone";
import { normalizeTwoToneColors } from "../utils";
export function setTwoToneColor(twoToneColor) {
  const [primaryColor, secondaryColor] = normalizeTwoToneColors(twoToneColor);
  return ReactIcon.setTwoToneColors({
    primaryColor,
    secondaryColor
  });
}
export function getTwoToneColor() {
  const colors = ReactIcon.getTwoToneColors();
  if (!colors.calculated) {
    return colors.primaryColor;
  }
  return [colors.primaryColor, colors.secondaryColor];
}