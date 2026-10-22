# Native "ceiling" builds of wasm/src/*.cpp (PLAN.md §6, optional), for sanity-checking
# the WASM numbers. Same source and flags as the WASM variants where it matters:
#   scalar   auto-vectorization off
#   simd     auto-vectorized (NEON / SSE: 4 × f32, like wasm simd128)
#   threads  simd + persistent pool of min(hardware threads, 8)
# -ffp-contract=off keeps a*b+c unfused, matching WASM (which has no FMA).
#
#   make        build bin/matmul-{scalar,simd,threads}
#   make run    build and print time + GFLOP/s per size for each variant

CXX      ?= clang++
SRC      := ../wasm/src
COMMON   := -std=c++20 -O3 -ffp-contract=off -I../wasm/include
VARIANTS := scalar simd threads

FLAGS_scalar  := -fno-vectorize -fno-slp-vectorize
FLAGS_simd    :=
FLAGS_threads := -pthread -DMATMUL_NATIVE_THREADS

.PHONY: all run clean
all: $(foreach v,$(VARIANTS),bin/matmul-$(v))

bin/matmul-%: $(SRC)/matmul.cpp bench_matmul.cpp $(wildcard ../wasm/include/*.hpp) Makefile | bin
	$(CXX) $(COMMON) $(FLAGS_$*) $(SRC)/matmul.cpp bench_matmul.cpp -o $@

bin:
	mkdir -p $@

run: all
	@for v in $(VARIANTS); do echo "== matmul-$$v"; bin/matmul-$$v; done

clean:
	rm -rf bin
