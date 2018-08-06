#include "q1_params.hpp"

#include <unordered_map>
#include <string>
#include <cuda/api_wrappers.h>

extern const std::unordered_map<std::string, cuda::device_function_t> plain_kernels;
extern const std::unordered_map<std::string, cuda::grid_block_dimension_t> fixed_threads_per_block;
extern const std::unordered_map<std::string, cuda::grid_block_dimension_t> max_threads_per_block;

q1_params_t parse_command_line(int argc, char** argv);
