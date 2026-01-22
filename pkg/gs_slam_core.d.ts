/* tslint:disable */
/* eslint-disable */

export class WasmViewer {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    compute_super_resolution(factor: number): void;
    export_ply(): Uint8Array;
    load_data(data: Uint8Array): void;
    static new(canvas_id: string): Promise<WasmViewer>;
    render(): void;
    resize(width: number, height: number): void;
    set_display_mode(mode: number): void;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_wasmviewer_free: (a: number, b: number) => void;
    readonly wasmviewer_compute_super_resolution: (a: number, b: number) => void;
    readonly wasmviewer_export_ply: (a: number) => [number, number];
    readonly wasmviewer_load_data: (a: number, b: number, c: number) => void;
    readonly wasmviewer_new: (a: number, b: number) => any;
    readonly wasmviewer_render: (a: number) => void;
    readonly wasmviewer_resize: (a: number, b: number, c: number) => void;
    readonly wasmviewer_set_display_mode: (a: number, b: number) => void;
    readonly wasm_bindgen__closure__destroy__hdea67971afca1db4: (a: number, b: number) => void;
    readonly wasm_bindgen__closure__destroy__haa8d35baf68946ad: (a: number, b: number) => void;
    readonly wasm_bindgen__closure__destroy__hc7cbe808f92bd2a2: (a: number, b: number) => void;
    readonly wasm_bindgen__convert__closures_____invoke__hc27386a376e4be9b: (a: number, b: number, c: any, d: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h4b9f10a45f608ac8: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h108e3b58c6d79455: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h986b81874b025409: (a: number, b: number, c: any) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
