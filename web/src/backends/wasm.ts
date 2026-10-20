// WASM module loading and feature detection.
//
// Emscripten output lives in public/wasm/ (built by wasm/Makefile) and is loaded at
// runtime rather than bundled: Vite then serves/copies it verbatim, so the loader's
// own `new URL('<name>.wasm', import.meta.url)` and pthread `new Worker(...)` calls
// resolve against /wasm/ exactly as Emscripten generated them.
import { simd, threads } from 'wasm-feature-detect';

export type WasmVariant = 'scalar' | 'simd' | 'threads';
export const WASM_VARIANTS: readonly WasmVariant[] = ['scalar', 'simd', 'threads'];

export interface WasmFeatures {
  simd: boolean;
  /** Shared memory + atomics validate and SharedArrayBuffer can be posted (needs cross-origin isolation). */
  threads: boolean;
}

export async function detectWasmFeatures(): Promise<WasmFeatures> {
  const [hasSimd, hasThreads] = await Promise.all([simd(), threads()]);
  return { simd: hasSimd, threads: hasThreads };
}

export function variantSupported(variant: WasmVariant, features: WasmFeatures): boolean {
  switch (variant) {
    case 'scalar':
      return true;
    case 'simd':
      return features.simd;
    case 'threads':
      return features.simd && features.threads; // threads build also uses -msimd128
  }
}

/**
 * The parts of an Emscripten module object the harness relies on; must match
 * EXPORTED_RUNTIME_METHODS / EXPORTED_FUNCTIONS in wasm/Makefile. With memory
 * growth (scalar/simd builds) the HEAP* views are replaced on growth, so re-read
 * them from the module after any allocation instead of caching them.
 */
export interface EmscriptenModule {
  HEAPU8: Uint8Array;
  HEAP32: Int32Array;
  HEAPU32: Uint32Array;
  HEAPF32: Float32Array;
  UTF8ToString(ptr: number): string;
  _malloc(bytes: number): number;
  _free(ptr: number): void;
}

export function wasmModuleUrl(name: string, variant: WasmVariant): string {
  return new URL(`${import.meta.env.BASE_URL}wasm/${name}-${variant}.mjs`, self.location.origin).href;
}

/** Imports and instantiates `<name>-<variant>.mjs`. For the threads build this also waits for the pthread pool. */
export async function loadWasm<M extends EmscriptenModule>(name: string, variant: WasmVariant): Promise<M> {
  const { default: factory } = (await import(/* @vite-ignore */ wasmModuleUrl(name, variant))) as {
    default: () => Promise<M>;
  };
  return factory();
}

/** Worker count for the threads build; must equal THREADS_POOL_SIZE in wasm/Makefile. */
export function wasmThreadCount(): number {
  return Math.min(navigator.hardwareConcurrency, 8);
}
