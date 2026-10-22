# Native "ceiling" builds of wasm/src/*.cpp (PLAN.md §6, optional), for sanity-checking
# the WASM numbers. Same source and flags as the WASM variants where it matters:
#   scalar   auto-vectorization off
#   simd     auto-vectorized (NEON / SSE: 4 × f32, like wasm simd128); NATIVE_SIMD
#            selects hand-written SIMD kernels (reduction) in place of wasm intrinsics
#   threads  simd + persistent pool of min(hardware threads, 8)
# -ffp-contract=off keeps a*b+c unfused, matching WASM (which has no FMA).
#
#   make        build bin/<workload>-{scalar,simd,threads} for every bench_<workload>.cpp
#   make run    build and print timings per size for each workload and variant

CXX       ?= clang++
SRC       := ../wasm/src
COMMON    := -std=c++20 -O3 -ffp-contract=off -I../wasm/include
VARIANTS  := scalar simd threads
WORKLOADS := $(patsubst bench_%.cpp,%,$(wildcard bench_*.cpp))

FLAGS_scalar  := -fno-vectorize -fno-slp-vectorize
FLAGS_simd    := -DNATIVE_SIMD
FLAGS_threads := -DNATIVE_SIMD -pthread -DNATIVE_THREADS

BINS := $(foreach w,$(WORKLOADS),$(foreach v,$(VARIANTS),bin/$(w)-$(v)))

.PHONY: all run clean
all: $(BINS)

define variant_rule
bin/%-$(1): $(SRC)/%.cpp bench_%.cpp $(wildcard ../wasm/include/*.hpp) Makefile | bin
	$(CXX) $(COMMON) $(FLAGS_$(1)) $(SRC)/$$*.cpp bench_$$*.cpp -o $$@
endef
$(foreach v,$(VARIANTS),$(eval $(call variant_rule,$(v))))

bin:
	mkdir -p $@

run: all
	@for b in $(BINS); do echo "== $${b#bin/}"; $$b; done

clean:
	rm -rf bin
