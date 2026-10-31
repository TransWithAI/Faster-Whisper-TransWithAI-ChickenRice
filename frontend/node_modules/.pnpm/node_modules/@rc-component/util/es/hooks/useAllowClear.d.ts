import * as React from 'react';
export interface AllowClearConfig {
    clearIcon?: React.ReactNode;
    /** Disable the clear entry without rendering a button or reserving space. */
    disabled?: boolean;
    tabIndex?: number;
    /** Accessible name for the clear button. */
    label?: string;
}
/** Normalize clear configuration. Components should apply their defaults first. */
export default function useAllowClear(allowClear?: boolean | AllowClearConfig): AllowClearConfig;
