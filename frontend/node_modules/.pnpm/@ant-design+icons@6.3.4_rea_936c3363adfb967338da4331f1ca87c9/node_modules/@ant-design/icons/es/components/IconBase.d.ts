import * as React from 'react';
import type { IconDefinition } from '@ant-design/icons-svg/lib/types';
export interface IconProps {
    icon: IconDefinition;
    className?: string;
    onClick?: React.MouseEventHandler<SVGSVGElement>;
    style?: React.CSSProperties;
    primaryColor?: string;
    secondaryColor?: string;
    focusable?: string;
}
declare const IconBase: React.FC<IconProps>;
export default IconBase;
