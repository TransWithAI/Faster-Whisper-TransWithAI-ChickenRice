import * as React from 'react';
import type { IconDefinition } from '@ant-design/icons-svg/lib/types';
import type { IconBaseProps } from './Icon';
import type { TwoToneColor } from './twoTonePrimaryColor';
export interface AntdIconProps extends IconBaseProps {
    twoToneColor?: TwoToneColor;
}
export interface IconComponentProps extends AntdIconProps {
    icon: IconDefinition;
}
declare const Icon: React.ForwardRefExoticComponent<Omit<IconComponentProps, "ref"> & React.RefAttributes<HTMLSpanElement>>;
export default Icon;
