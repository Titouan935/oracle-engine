cmake_minimum_required(VERSION 3.14)
project(oracle LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Optimisation selon l'architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -mcpu=native")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -mavx2 -mfma -march=native")
endif()

find_package(OpenMP QUIET)

# Sources du moteur (zéro dépendance externe)
set(ENGINE_SRC
    core/engine.cpp
    core/model.cpp
    core/tokenizer.cpp
    core/sampler.cpp
    core/ops.cpp
    core/gguf_parser.cpp
    core/platform.cpp
    core/machine_profile.cpp
    bug/logger.cpp
)

add_library(oracle_engine STATIC ${ENGINE_SRC})
target_include_directories(oracle_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
if(WIN32)
    target_compile_definitions(oracle_engine PUBLIC _WIN32_WINNT=0x0A00)
    target_link_libraries(oracle_engine PUBLIC -lwinhttp -lws2_32 -lole32 -lbcrypt -lgdi32 -luser32)
else()
    target_link_libraries(oracle_engine PUBLIC dl pthread)
endif()
if(OpenMP_CXX_FOUND)
    target_link_libraries(oracle_engine PUBLIC OpenMP::OpenMP_CXX)
    target_compile_options(oracle_engine PUBLIC -fopenmp)
endif()

# CLI : oracle -m model.gguf -p "prompt"
add_executable(oracle apps/oracle_cli.cpp)
target_link_libraries(oracle PRIVATE oracle_engine)
if(WIN32)
    target_link_options(oracle PRIVATE -mconsole)
endif()

# Benchmark reproductible
add_executable(oracle_bench benchmark/bench_engine.cpp)
target_link_libraries(oracle_bench PRIVATE oracle_engine)
target_compile_definitions(oracle_bench PRIVATE
    BENCH_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
    BENCH_CXX_FLAGS="${CMAKE_CXX_FLAGS_RELEASE}")
if(WIN32)
    target_link_libraries(oracle_bench PRIVATE -lpsapi)
    target_link_options(oracle_bench PRIVATE -mconsole)
endif()

# Micro-benchmark isolé du kernel gemm_q
add_executable(micro_gemm benchmark/micro_gemm.cpp)
target_link_libraries(micro_gemm PRIVATE oracle_engine)
if(WIN32)
    target_link_options(micro_gemm PRIVATE -mconsole)
endif()
