import * as React from 'react';
type Ref<T> = React.Ref<T> | undefined;
export declare function useComposeRef<T>(...refs: Ref<T>[]): React.Ref<T>;
export {};
