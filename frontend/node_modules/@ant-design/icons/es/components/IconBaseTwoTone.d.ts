import * as React from 'react';
import type { IconProps } from './IconBase';
export interface TwoToneColorPaletteSetter {
    primaryColor: string;
    secondaryColor?: string;
}
export interface TwoToneColorPalette extends TwoToneColorPaletteSetter {
    calculated?: boolean;
}
declare function setTwoToneColors({ primaryColor, secondaryColor }: TwoToneColorPaletteSetter): void;
declare function getTwoToneColors(): TwoToneColorPalette;
interface IconBaseTwoToneComponent<P> extends React.FC<P> {
    getTwoToneColors: typeof getTwoToneColors;
    setTwoToneColors: typeof setTwoToneColors;
}
declare const IconBaseTwoTone: IconBaseTwoToneComponent<IconProps>;
export default IconBaseTwoTone;
