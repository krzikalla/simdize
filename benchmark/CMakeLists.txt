add_executable(
  simd_access_benchmark
  compute_bm.cpp
  loop_bm.cpp
  universal_bm.cpp
)
target_link_libraries(
  simd_access_benchmark
  benchmark::benchmark
  benchmark::benchmark_main
#  -fsanitize=address,undefined
)

#include(googlebenchmark)
