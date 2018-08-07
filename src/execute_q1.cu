#include "execute_q1.hpp"
#include "data_types.h"
#include "constants.hpp"
#include "kernels/ht_in_global_mem.cuh"
#include "kernels/ht_in_registers.cuh"
#include "kernels/ht_in_registers_per_thread.cuh"
#include "kernels/ht_in_local_mem.cuh"
#include "kernels/ht_in_shared_mem_per_thread.cuh"
#include "kernels/ht_in_shared_mem_per_bank.cuh"
#include "kernels/ht_in_shared_mem_per_block.cuh"
#include "cpu/common.hpp"
#include "cpu.h"

#include "util/helper.hpp"
#include "util/extra_pointer_traits.hpp"
#include "util/bit_operations.hpp"

#include <iostream>
#include <chrono>
#include <unordered_map>
#include <numeric>
#include <sstream>

#ifndef GPU
#error The GPU preprocessor directive must be defined (ask Tim for the reason)
#endif

using std::tie;
using std::make_pair;
using std::make_unique;
using std::unique_ptr;
using std::cout;
using std::cerr;
using std::endl;
using std::flush;
using std::string;
using cuda::warp_size;


const std::unordered_map<string, cuda::device_function_t> plain_kernels = {
    { "local_mem",               kernels::local_mem::one_table_per_thread::tpch_query_01         },
    { "in_registers",            kernels::in_registers::several_tables_per_warp::tpch_query_01   },
    { "in_registers_per_thread", kernels::in_registers::one_table_per_thread::tpch_query_01      },
    { "shared_mem_per_thread",   kernels::shared_mem::one_table_per_thread::tpch_query_01<>      },
    { "shared_mem_per_block",    kernels::shared_mem::one_table_per_block::tpch_query_01         },
    { "shared_mem_per_bank",     kernels::shared_mem::one_table_per_bank::tpch_query_01          },
    { "global",                  kernels::global_mem::single_table::tpch_query_01                },
};

const std::unordered_map<string, cuda::device_function_t> kernels_compressed = {
    { "local_mem",               kernels::local_mem::one_table_per_thread::tpch_query_01_compressed       },
    { "in_registers",            kernels::in_registers::several_tables_per_warp::tpch_query_01_compressed },
    { "in_registers_per_thread", kernels::in_registers::one_table_per_thread::tpch_query_01_compressed    },
    { "shared_mem_per_thread",   kernels::shared_mem::one_table_per_thread::tpch_query_01_compressed<>    },
    { "shared_mem_per_block",    kernels::shared_mem::one_table_per_block::tpch_query_01_compressed       },
    { "shared_mem_per_bank",     kernels::shared_mem::one_table_per_bank::tpch_query_01_compressed        },
    { "global",                  kernels::global_mem::single_table::tpch_query_01_compressed              },
};

const std::unordered_map<string, cuda::device_function_t> kernels_filter_pushdown = {
    { "local_mem",               kernels::local_mem::one_table_per_thread::tpch_query_01_compressed_precomputed_filter       },
    { "in_registers",            kernels::in_registers::several_tables_per_warp::tpch_query_01_compressed_precomputed_filter },
    { "in_registers_per_thread", kernels::in_registers::one_table_per_thread::tpch_query_01_compressed_precomputed_filter    },
    { "global",                  kernels::global_mem::single_table::tpch_query_01_compressed_precomputed_filter              },
    { "shared_mem_per_block",    kernels::shared_mem::one_table_per_block::tpch_query_01_compressed_precomputed_filter       },
    { "shared_mem_per_bank",     kernels::shared_mem::one_table_per_bank::tpch_query_01_compressed_precomputed_filter        },
    { "shared_mem_per_thread",   kernels::shared_mem::one_table_per_thread::tpch_query_01_compressed_precomputed_filter<>    },
};

// Some kernel variants cannot support as many threads per block as the hardware allows generally,
// and for these we use a fixed number for now
const std::unordered_map<string, cuda::grid_block_dimension_t> fixed_threads_per_block = {
    { "in_registers",           kernels::in_registers::several_tables_per_warp::fixed_threads_per_block },
};

const std::unordered_map<string, cuda::grid_block_dimension_t> max_threads_per_block = {
    { "shared_mem_per_thread", kernels::shared_mem::one_table_per_thread::max_threads_per_block },
};



void execute_query_1_once(
    const q1_params_t&              __restrict__  params,
    cuda::device_t<>                              cuda_device,
    int                                           run_index,
    cardinality_t                                 cardinality,
    std::vector<cuda::stream_t<>>&  __restrict__  streams,
    host_aggregates_t&              __restrict__  aggregates_on_host,
    device_aggregates_t&            __restrict__  aggregates_on_device,
    stream_input_buffer_sets&       __restrict__  stream_input_buffer_sets,
    input_buffer_set<plain_ptr, is_not_compressed>&
                                    __restrict__  uncompressed, // on host
    input_buffer_set<cuda::memory::host::unique_ptr, is_compressed>&
                                    __restrict__  compressed // on host
)
{
    if (params.use_coprocessing || params.use_filter_pushdown) {
         cpu_coprocessor->Clear();
    }
    auto gpu_end_offset = cardinality;
    if (params.use_coprocessing || params.use_filter_pushdown) {
        cardinality_t cpu_start_offset = cardinality;
        if (params.use_coprocessing) {
            // Split the work between the CPU and the GPU; GPU part
            // starts at the beginning, CPU part starts at some offset
            cpu_start_offset -= cardinality * params.cpu_processing_fraction;

            // To allow
            // for nice assumed alignment for the CPU-processed data,
            // we only let the CPU start at full batch intervals
            // (which is not theoretically necessary but simpler for us)
            cpu_start_offset -= cpu_start_offset % params.num_tuples_per_kernel_launch;
        }
        else {
            cpu_start_offset = cardinality;
        }

        auto num_records_for_cpu_to_process = cardinality - cpu_start_offset;

        precomp_filter = compressed.precomputed_filter.get();
        compr_shipdate = compressed.ship_date.get();

        if (params.use_filter_pushdown) {
            // Process everything on the CPU, but don't aggregate anything before cpu_start_offset
            (*cpu_coprocessor)(0, cardinality, cpu_start_offset);
        }
        else {
            // Process  num_records_for_cpu_to_process starting at cpu_start_offset,
            // and don't precompute the filter for any part of the table
            (*cpu_coprocessor)(cpu_start_offset, num_records_for_cpu_to_process, 0);
        }

         gpu_end_offset = cpu_start_offset;
    }

    // Initialize the aggregates; perhaps we should do this in a single kernel? ... probably not worth it
    streams[0].enqueue.memset(aggregates_on_device.sum_quantity.get(),         0, num_potential_groups * sizeof(sum_quantity_t));
    streams[0].enqueue.memset(aggregates_on_device.sum_base_price.get(),       0, num_potential_groups * sizeof(sum_base_price_t));
    streams[0].enqueue.memset(aggregates_on_device.sum_discounted_price.get(), 0, num_potential_groups * sizeof(sum_discounted_price_t));
    streams[0].enqueue.memset(aggregates_on_device.sum_charge.get(),           0, num_potential_groups * sizeof(sum_charge_t));
    streams[0].enqueue.memset(aggregates_on_device.sum_discount.get(),         0, num_potential_groups * sizeof(sum_discount_t));
    streams[0].enqueue.memset(aggregates_on_device.record_count.get(),         0, num_potential_groups * sizeof(cardinality_t));

    cuda::event_t aggregates_initialized_event = streams[0].enqueue.event(
        cuda::event::sync_by_blocking, cuda::event::dont_record_timings, cuda::event::not_interprocess);
    for (int i = 1; i < params.num_gpu_streams; ++i) {
        streams[i].enqueue.wait(aggregates_initialized_event);
        // The other streams also require the aggregates to be initialized before doing any work
    }

    auto stream_index = 0;
    for (size_t offset_in_table2 = 0;
         offset_in_table2 < gpu_end_offset;
         offset_in_table2 += params.num_tuples_per_kernel_launch,
         stream_index = (stream_index+1) % params.num_gpu_streams)
    {
        size_t offset_in_table;
        cardinality_t num_tuples_for_this_launch;
        if (params.use_filter_pushdown) {
            FilterChunk chunk;
            precomp_filter_queue.wait_dequeue(chunk);

            offset_in_table = chunk.offset;
            num_tuples_for_this_launch = chunk.num;
        } else {
            offset_in_table = offset_in_table2;
            num_tuples_for_this_launch = std::min<cardinality_t>(params.num_tuples_per_kernel_launch, gpu_end_offset - offset_in_table);
        }

        auto num_return_flag_bit_containers_for_this_launch = div_rounding_up(num_tuples_for_this_launch, return_flag_values_per_container);
        auto num_line_status_bit_containers_for_this_launch = div_rounding_up(num_tuples_for_this_launch, line_status_values_per_container);

        // auto start_copy = timer::now();  // This can't work, since copying is asynchronous.
        auto& stream = streams[stream_index];

        if (params.apply_compression) {
            auto& stream_input_buffer_set = stream_input_buffer_sets.compressed[stream_index];
#if 0
            if (false && params.use_filter_pushdown) {
                cuda::profiling::scoped_range_marker("On-CPU filtering");
                precompute_filter_for_table_chunk(
                    compressed.ship_date.get() + offset_in_table,
                    compressed.precomputed_filter.get() + offset_in_table / bits_per_container,
                    num_tuples_for_this_launch);
            }
#endif
            stream.enqueue.copy(stream_input_buffer_set.discount.get()      , compressed.discount.get()       + offset_in_table, num_tuples_for_this_launch * sizeof(compressed::discount_t));
            stream.enqueue.copy(stream_input_buffer_set.extended_price.get(), compressed.extended_price.get() + offset_in_table, num_tuples_for_this_launch * sizeof(compressed::extended_price_t));
            stream.enqueue.copy(stream_input_buffer_set.tax.get()           , compressed.tax.get()            + offset_in_table, num_tuples_for_this_launch * sizeof(compressed::tax_t));
            stream.enqueue.copy(stream_input_buffer_set.quantity.get()      , compressed.quantity.get()       + offset_in_table, num_tuples_for_this_launch * sizeof(compressed::quantity_t));
            stream.enqueue.copy(stream_input_buffer_set.return_flag.get()   , compressed.return_flag.get()    + offset_in_table / return_flag_values_per_container, num_return_flag_bit_containers_for_this_launch * sizeof(bit_container_t));
            stream.enqueue.copy(stream_input_buffer_set.line_status.get()   , compressed.line_status.get()    + offset_in_table / line_status_values_per_container, num_line_status_bit_containers_for_this_launch * sizeof(bit_container_t));
            if (not params.use_filter_pushdown) {
                stream.enqueue.copy(stream_input_buffer_set.ship_date.get(), compressed.ship_date.get() + offset_in_table, num_tuples_for_this_launch * sizeof(compressed::ship_date_t));
            } else {
                auto num_bit_containers_for_this_launch = div_rounding_up(num_tuples_for_this_launch, bits_per_container);
                stream.enqueue.copy(stream_input_buffer_set.precomputed_filter.get(), compressed.precomputed_filter.get() + offset_in_table / bits_per_container, num_bit_containers_for_this_launch * sizeof(bit_container_t));
            }
        }
        else {
            auto& stream_input_buffer_set = stream_input_buffer_sets.uncompressed[stream_index];
            stream.enqueue.copy(stream_input_buffer_set.ship_date.get()     , uncompressed.ship_date      + offset_in_table, num_tuples_for_this_launch * sizeof(ship_date_t));
            stream.enqueue.copy(stream_input_buffer_set.discount.get()      , uncompressed.discount       + offset_in_table, num_tuples_for_this_launch * sizeof(discount_t));
            stream.enqueue.copy(stream_input_buffer_set.extended_price.get(), uncompressed.extended_price + offset_in_table, num_tuples_for_this_launch * sizeof(extended_price_t));
            stream.enqueue.copy(stream_input_buffer_set.tax.get()           , uncompressed.tax            + offset_in_table, num_tuples_for_this_launch * sizeof(tax_t));
            stream.enqueue.copy(stream_input_buffer_set.quantity.get()      , uncompressed.quantity       + offset_in_table, num_tuples_for_this_launch * sizeof(quantity_t));
            stream.enqueue.copy(stream_input_buffer_set.return_flag.get()   , uncompressed.return_flag    + offset_in_table, num_tuples_for_this_launch * sizeof(return_flag_t));
            stream.enqueue.copy(stream_input_buffer_set.line_status.get()   , uncompressed.line_status    + offset_in_table, num_tuples_for_this_launch * sizeof(line_status_t));
        }

        cuda::grid_block_dimension_t num_threads_per_block;
        cuda::grid_dimension_t       num_blocks;

        if (params.kernel_variant == "in_registers") {
            // Calculation is different here because more than one thread works on a tuple,
            // and some figuring is per-warp
            auto num_warps_per_block = params.num_threads_per_block / warp_size;
                // rounding down the number of threads per block!
            num_threads_per_block = num_warps_per_block * warp_size;
            auto num_tables_per_warp       = cuda::warp_size / num_potential_groups;
            auto num_tuples_handled_by_block = num_tables_per_warp * num_warps_per_block * params.num_tuples_per_thread;
            num_blocks = div_rounding_up(
                num_tuples_for_this_launch,
                num_tuples_handled_by_block);
        }
        else {
            num_blocks = div_rounding_up(
                    num_tuples_for_this_launch,
                    params.num_threads_per_block * params.num_tuples_per_thread);
            // NOTE: If the number of blocks drops below the number of GPU cores, this is definitely useless,
            // and to be on the safe side - twice as many.
            num_threads_per_block = params.num_threads_per_block;
        }
        auto launch_config = cuda::make_launch_config(num_blocks, num_threads_per_block);

//        cout << "Launch parameters:\n"
//             << "    num_tuples_for_this_launch  = " << num_tuples_for_this_launch << '\n'
//             << "    grid blocks                 = " << num_blocks << '\n'
//             << "    threads per block           = " << num_threads_per_block << '\n';

        if (params.use_filter_pushdown) {
            assert(params.apply_compression && "Filter pre-computation is only currently supported when compression is employed");
            auto& stream_input_buffer_set = stream_input_buffer_sets.compressed[stream_index];
            auto kernel = kernels_filter_pushdown.at(params.kernel_variant);
            stream.enqueue.kernel_launch(
                kernel,
                launch_config,
                aggregates_on_device.sum_quantity.get(),
                aggregates_on_device.sum_base_price.get(),
                aggregates_on_device.sum_discounted_price.get(),
                aggregates_on_device.sum_charge.get(),
                aggregates_on_device.sum_discount.get(),
                aggregates_on_device.record_count.get(),
                stream_input_buffer_set.precomputed_filter.get(),
                stream_input_buffer_set.discount.get(),
                stream_input_buffer_set.extended_price.get(),
                stream_input_buffer_set.tax.get(),
                stream_input_buffer_set.quantity.get(),
                stream_input_buffer_set.return_flag.get(),
                stream_input_buffer_set.line_status.get(),
                num_tuples_for_this_launch);
        } else if (params.apply_compression) {
            auto& stream_input_buffer_set = stream_input_buffer_sets.compressed[stream_index];
            auto kernel = kernels_compressed.at(params.kernel_variant);
            stream.enqueue.kernel_launch(
                kernel,
                launch_config,
                aggregates_on_device.sum_quantity.get(),
                aggregates_on_device.sum_base_price.get(),
                aggregates_on_device.sum_discounted_price.get(),
                aggregates_on_device.sum_charge.get(),
                aggregates_on_device.sum_discount.get(),
                aggregates_on_device.record_count.get(),
                stream_input_buffer_set.ship_date.get(),
                stream_input_buffer_set.discount.get(),
                stream_input_buffer_set.extended_price.get(),
                stream_input_buffer_set.tax.get(),
                stream_input_buffer_set.quantity.get(),
                stream_input_buffer_set.return_flag.get(),
                stream_input_buffer_set.line_status.get(),
                num_tuples_for_this_launch);
        } else {
            auto& stream_input_buffer_set = stream_input_buffer_sets.uncompressed[stream_index];
            auto kernel = plain_kernels.at(params.kernel_variant);
            stream.enqueue.kernel_launch(
                kernel,
                launch_config,
                aggregates_on_device.sum_quantity.get(),
                aggregates_on_device.sum_base_price.get(),
                aggregates_on_device.sum_discounted_price.get(),
                aggregates_on_device.sum_charge.get(),
                aggregates_on_device.sum_discount.get(),
                aggregates_on_device.record_count.get(),
                stream_input_buffer_set.ship_date.get(),
                stream_input_buffer_set.discount.get(),
                stream_input_buffer_set.extended_price.get(),
                stream_input_buffer_set.tax.get(),
                stream_input_buffer_set.quantity.get(),
                stream_input_buffer_set.return_flag.get(),
                stream_input_buffer_set.line_status.get(),
                num_tuples_for_this_launch);
        }
    }
    for(int i = 1; i < params.num_gpu_streams; i++) {
        auto ev = cuda_device.create_event(cuda::event::sync_by_blocking);
        streams[i].enqueue.event(ev);
        streams[0].enqueue.wait(ev);
    }

    streams[0].enqueue.copy(aggregates_on_host.sum_quantity.get(),         aggregates_on_device.sum_quantity.get(),         num_potential_groups * sizeof(sum_quantity_t));
    streams[0].enqueue.copy(aggregates_on_host.sum_base_price.get(),       aggregates_on_device.sum_base_price.get(),       num_potential_groups * sizeof(sum_base_price_t));
    streams[0].enqueue.copy(aggregates_on_host.sum_discounted_price.get(), aggregates_on_device.sum_discounted_price.get(), num_potential_groups * sizeof(sum_discounted_price_t));
    streams[0].enqueue.copy(aggregates_on_host.sum_charge.get(),           aggregates_on_device.sum_charge.get(),           num_potential_groups * sizeof(sum_charge_t));
    streams[0].enqueue.copy(aggregates_on_host.sum_discount.get(),         aggregates_on_device.sum_discount.get(),         num_potential_groups * sizeof(sum_discount_t));
    streams[0].enqueue.copy(aggregates_on_host.record_count.get(),         aggregates_on_device.record_count.get(),         num_potential_groups * sizeof(cardinality_t));

    if (cpu_coprocessor) { cpu_coprocessor->wait(); }

    streams[0].synchronize();
}
