export type Path = (string | number | symbol)[];
export default function set<Entity = any, Output = Entity, Value = any>(entity: Entity, paths: Path, value: Value, removeIfUndefined?: boolean): Output;
export type MergeFn = (current: any, next: any) => any;
/**
 * Merge multiple objects. Support custom merge logic.
 * @param sources object sources
 * @param config.prepareArray Customize array prepare function.
 * It will return empty [] by default.
 * So when match array, it will auto be override with next array in sources.
 */
export declare function mergeWith<T extends object>(sources: T[], config?: {
    prepareArray?: MergeFn;
}): T;
/**
 * Merge multiple objects into a new single object.
 * Arrays will be replaced by default.
 */
export declare function merge<T extends object>(...sources: T[]): T;
