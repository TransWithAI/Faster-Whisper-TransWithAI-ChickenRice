import type * as React from 'react';
import type { Root } from 'react-dom/client';
declare const MARK = "__rc_react_root__";
type ContainerType = (Element | DocumentFragment) & {
    [MARK]?: Root;
};
export declare function render(node: React.ReactElement, container: ContainerType): void;
export declare function unmount(container: ContainerType): Promise<void>;
export {};
