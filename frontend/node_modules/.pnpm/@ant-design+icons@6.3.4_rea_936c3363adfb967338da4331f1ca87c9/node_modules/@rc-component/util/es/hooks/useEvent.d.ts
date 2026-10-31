declare const useEvent: <T extends (...args: any[]) => any>(callback: T) => undefined extends T ? (...args: Parameters<NonNullable<T>>) => ReturnType<NonNullable<T>> | undefined : T;
export default useEvent;
