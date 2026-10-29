# Builds every wasm/src/<name>.cpp into three ES-module variants:
#   web/public/wasm/<name>-scalar.{mjs,wasm}
#   web/public/wasm/<name>-simd.{mjs,wasm}
#   web/public/wasm/<name>-threads.{mjs,wasm}
#
#   make            build all variants
#   make check-simd confirm v128 instructions appear in -simd builds and not in -scalar
#                   (workloads in NO_SIMD have no vector form: only the scalar half is checked)
#   make clean

ROOT     := $(abspath ..)
OUT      := $(ROOT)/web/public/wasm
SOURCES  := $(wildcard src/*.cpp)
NAMES    := $(basename $(notdir $(SOURCES)))
VARIANTS := scalar simd threads

# Prefer the repo-local pinned SDK (scripts/setup-emsdk.sh), else whatever is on PATH.
EMSDK_DIR := $(ROOT)/.tools/emsdk
ifneq ($(wildcard $(EMSDK_DIR)/upstream/emscripten/emcc),)
  EMXX    ?= $(EMSDK_DIR)/upstream/emscripten/em++
  OBJDUMP ?= $(EMSDK_DIR)/upstream/bin/llvm-objdump
else
  EMXX    ?= em++
  OBJDUMP ?= llvm-objdump
endif

# No -ffast-math anywhere: keep JS-like float semantics (PLAN.md §5).
COMMON := -std=c++20 -O3 -Iinclude \
  -sMODULARIZE -sEXPORT_ES6 -sENVIRONMENT=web,worker \
  -sEXPORTED_FUNCTIONS=_malloc,_free \
  -sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAP32,HEAPU32,HEAPF32,UTF8ToString \
  -sFILESYSTEM=0 -sASSERTIONS=0

# Threads: fixed memory, no growth (growth + pthreads slows every heap access).
THREADS_INITIAL_MEMORY ?= 512MB
THREADS_POOL_SIZE      ?= Math.min(navigator.hardwareConcurrency,8)

FLAGS_scalar  := -sALLOW_MEMORY_GROWTH=1
FLAGS_simd    := -msimd128 -sALLOW_MEMORY_GROWTH=1
FLAGS_threads := -msimd128 -pthread -sPTHREAD_POOL_SIZE='$(THREADS_POOL_SIZE)' \
  -sINITIAL_MEMORY=$(THREADS_INITIAL_MEMORY) -sALLOW_MEMORY_GROWTH=0

# Kernels with no simd128 form; their -simd build is the same code with -msimd128.
NO_SIMD := sort

TARGETS := $(foreach n,$(NAMES),$(foreach v,$(VARIANTS),$(OUT)/$(n)-$(v).mjs))

.PHONY: all clean check-simd
all: $(TARGETS)

define variant_rule
$(OUT)/%-$(1).mjs: src/%.cpp $(wildcard include/*.hpp) Makefile | $(OUT)
	$(EMXX) $(COMMON) $(FLAGS_$(1)) -DWASM_VARIANT=$(1) $$< -o $$@
endef
$(foreach v,$(VARIANTS),$(eval $(call variant_rule,$(v))))

$(OUT):
	mkdir -p $@

check-simd: all
	@set -e; for n in $(NAMES); do \
	  s=$$($(OBJDUMP) -d $(OUT)/$$n-scalar.wasm | grep -c 'v128\|f32x4\|i32x4' || true); \
	  v=$$($(OBJDUMP) -d $(OUT)/$$n-simd.wasm   | grep -c 'v128\|f32x4\|i32x4' || true); \
	  t=$$($(OBJDUMP) -d $(OUT)/$$n-threads.wasm | grep -c 'v128\|f32x4\|i32x4' || true); \
	  echo "$$n: simd-instructions scalar=$$s simd=$$v threads=$$t"; \
	  if [ "$$s" -ne 0 ]; then echo "  FAIL"; exit 1; fi; \
	  case " $(NO_SIMD) " in *" $$n "*) echo "  (no vector form expected)"; continue;; esac; \
	  if [ "$$v" -eq 0 ] || [ "$$t" -eq 0 ]; then echo "  FAIL"; exit 1; fi; \
	done; echo "check-simd OK"

clean:
	rm -rf $(OUT)
