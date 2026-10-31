export { getSecondaryColor } from "./colorUtils";
export { generate, iconStyles, isIconDefinition, normalizeAttrs, svgBaseProps, useInsertStyles, warning } from "./renderUtils";
export function normalizeTwoToneColors(twoToneColor) {
  if (!twoToneColor) {
    return [];
  }
  return Array.isArray(twoToneColor) ? twoToneColor : [twoToneColor];
}