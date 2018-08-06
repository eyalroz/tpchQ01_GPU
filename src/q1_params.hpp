#pragma once
#ifndef Q1_PARAMS_H_
#define Q1_PARAMS_H_

#include "data_types.h"
#include "constants.hpp"

#include <iostream>
#include <cuda/api_wrappers.h>

struct q1_params_t {

    cuda::device::id_t cuda_device_id    { cuda::device::default_device_id };
    double scale_factor                  { defaults::scale_factor };
    std::string kernel_variant           { defaults::kernel_variant };
    bool should_print_results            { defaults::should_print_results };
    bool use_filter_pushdown             { false };
    bool apply_compression               { defaults::apply_compression };
    int num_gpu_streams                  { defaults::num_gpu_streams };
    cuda::grid_block_dimension_t num_threads_per_block
                                         { defaults::num_threads_per_block };
    cardinality_t num_tuples_per_thread  { defaults::num_tuples_per_thread };
    cardinality_t num_tuples_per_kernel_launch
                                         { defaults::num_tuples_per_kernel_launch };
        // Make sure it's a multiple of num_threads_per_block and of warp_size, or bad things may happen

    // This is the number of times we run the actual query execution - the part that we time;
    // it will not include initialization/allocations that are not necessary when the DBMS
    // is brought up. Note the allocation vs sub-allocation issue (see further comments below)
    int num_query_execution_runs         { defaults::num_query_execution_runs };

    bool use_coprocessing                { false };

    // The fraction of the total table (i.e. total number of tuples) which the CPU, rather than
    // the GPU, will undertake to process; this ignores any filter precomputation work.
    double cpu_processing_fraction       { defaults::cpu_coprocessing_fraction };
//    bool user_set_num_threads_per_block  { false };
};

inline std::ostream& operator<<(std::ostream& os, const q1_params_t& p)
{
    os << "SF = " << p.scale_factor << " | "
       << "kernel = " << p.kernel_variant << " | "
       << (p.use_filter_pushdown ? "filter precomp" : "") << " | "
       << (p.apply_compression ? "compressed" : "uncompressed" ) << " | "
       << "streams = " << p.num_gpu_streams << " | "
       << "block size = " << p.num_threads_per_block << " | "
       << "tuples per thread = " << p.num_tuples_per_thread << " | "
       << "batch size " << p.num_tuples_per_kernel_launch;
    return os;
}


#endif // Q1_PARAMS_H_
