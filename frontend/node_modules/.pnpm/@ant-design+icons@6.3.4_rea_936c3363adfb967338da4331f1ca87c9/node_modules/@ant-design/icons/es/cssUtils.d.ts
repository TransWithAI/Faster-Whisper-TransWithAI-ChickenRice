export declare function updateCSS(css: string, key: string, originOption?: {
    attachTo?: Element | ShadowRoot;
    csp?: {
        nonce?: string;
    };
    mark?: string;
    prepend?: boolean | 'queue';
    priority?: number;
}): HTMLStyleElement;
export declare function getShadowRoot(ele: HTMLElement | null): ShadowRoot;
export declare function warningOnce(valid: boolean, message: string): void;
