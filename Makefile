# ORACLE — build minimal (macOS / Linux). Windows : voir README (CMake).
#
#   make              # build oracle, oracle_bench, micro_gemm (avec OpenMP)
#   make OPENMP=0     # sans OpenMP (attention mono-thread ; zéro dépendance)
#   make clean
#
# Aucune dépendance externe : juste un compilateur C++17.

CXX      ?= c++
UNAME_M  := $(shell uname -m)
UNAME_S  := $(shell uname -s)

ifeq ($(UNAME_M),arm64)
  ARCH := -mcpu=native
else ifeq ($(UNAME_M),aarch64)
  ARCH := -mcpu=native
else
  ARCH := -mavx2 -mfma -march=native
endif

OPENMP ?= 1
ifeq ($(OPENMP),1)
  OMP := -fopenmp
else
  OMP :=
endif

CXXFLAGS := -std=c++17 -O3 -DNDEBUG $(ARCH) $(OMP) -I.
LDFLAGS  := $(OMP)
ifneq ($(UNAME_S),Darwin)
  LDLIBS := -ldl -lpthread
else
  LDLIBS :=
endif

BENCH_DEFS := -DBENCH_COMPILER="\"$(shell $(CXX) --version | head -n1)\"" \
              -DBENCH_CXX_FLAGS="\"$(ARCH) $(OMP)\""

ENGINE_SRC := core/engine.cpp core/model.cpp core/tokenizer.cpp core/sampler.cpp \
              core/ops.cpp core/gguf_parser.cpp core/platform.cpp \
              core/machine_profile.cpp bug/logger.cpp
ENGINE_OBJ := $(ENGINE_SRC:.cpp=.o)

all: oracle oracle_bench micro_gemm

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

oracle: apps/oracle_cli.o $(ENGINE_OBJ)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDLIBS)

oracle_bench: benchmark/bench_engine.o $(ENGINE_OBJ)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDLIBS)

# bench_engine a besoin des defines compilo/flags
benchmark/bench_engine.o: benchmark/bench_engine.cpp
	$(CXX) $(CXXFLAGS) $(BENCH_DEFS) -c $< -o $@

micro_gemm: benchmark/micro_gemm.o $(ENGINE_OBJ)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDLIBS)

clean:
	rm -f oracle oracle_bench micro_gemm \
	      apps/*.o benchmark/*.o $(ENGINE_OBJ)

.PHONY: all clean
