#pragma once

#include "util/preprocessor_shorthands.h"
#include "util/atomics.cuh"
#include "constants.hpp"
#include "data_types.h"
#include "util/bit_operations.hpp"

namespace kernels {
namespace shared_mem {
namespace one_table_per_bank {

using cuda::warp_size;

enum {
    size_of_set_of_hash_tables =
        (sizeof(sum_quantity_t) + sizeof(sum_base_price_t) + sizeof(sum_discounted_price_t) +
        sizeof(sum_charge_t) + sizeof(sum_discount_t) + sizeof (cardinality_t)) * num_potential_groups,
    assumed_shared_memory_size = (64 * 1024), // Note: This will fail on Kepler and older GPUs
    // shared_memory_lane_width = 4,
    num_hash_table_sets = warp_size, // / (sizeof(sum_quantity_t) / shared_memory_lane_width),
    total_hash_tables_size = num_hash_table_sets * size_of_set_of_hash_tables
};

static_assert(assumed_shared_memory_size >= size_of_set_of_hash_tables, "Shared memory too small to fit all hashes");
	// In a JIT setting, this would be a runtime check, i.e. we won't JIT this kernel if the tables are too large;
	// In a Q1-like setting, the maximum number of groups when they're each in separate lanes is about 46. We could
	// also have one table spanning pair or triples of lanes etc., but that's not implemented here


__global__
void tpch_query_01(
    sum_quantity_t*          __restrict__ sum_quantity,
    sum_base_price_t*        __restrict__ sum_base_price,
    sum_discounted_price_t*  __restrict__ sum_discounted_price,
    sum_charge_t*            __restrict__ sum_charge,
    sum_discount_t*          __restrict__ sum_discount,
    cardinality_t*           __restrict__ record_count,
    const ship_date_t*       __restrict__ ship_date,
    const discount_t*        __restrict__ discount,
    const extended_price_t*  __restrict__ extended_price,
    const tax_t*             __restrict__ tax,
    const quantity_t*        __restrict__ quantity,
    const return_flag_t*     __restrict__ return_flag,
    const line_status_t*     __restrict__ line_status,
    cardinality_t                         num_tuples)
{
    __shared__ sum_quantity_t         sums_of_quantity         [num_potential_groups * num_hash_table_sets];
    __shared__ sum_base_price_t       sums_of_base_price       [num_potential_groups * num_hash_table_sets];
    __shared__ sum_discounted_price_t sums_of_discounted_price [num_potential_groups * num_hash_table_sets];
    __shared__ sum_charge_t           sums_of_charge           [num_potential_groups * num_hash_table_sets];
    __shared__ sum_discount_t         sums_of_discount         [num_potential_groups * num_hash_table_sets];
    __shared__ cardinality_t          record_counts            [num_potential_groups * num_hash_table_sets];
        // In each of these arrays, A single hash table is a strided sequence of elements (stride of
    	// warp size = number of banks = 32 elements; and that works both for the 64-bit and the 32-bit
    	// tables, apparently)

    cardinality_t input_stride = (blockDim.x * gridDim.x); //Grid-Stride
    auto warp_index = threadIdx.x / warp_size;
    auto warps_in_block = blockDim.x / warp_size;
    auto lane_index = threadIdx.x % warp_size;
    auto table_set_index = lane_index;

    auto lane_sum_of_quantity          = sums_of_quantity         + table_set_index;
    auto lane_sum_of_base_price        = sums_of_base_price       + table_set_index;
    auto lane_sum_of_charge            = sums_of_charge           + table_set_index;
    auto lane_sum_of_discounted_price  = sums_of_discounted_price + table_set_index;
    auto lane_sum_of_discount          = sums_of_discount         + table_set_index;
    auto lane_record_counts             = record_counts            + table_set_index;

    // all block threads which are to work on a certain hashtable will initialize that table

    for(int i = threadIdx.x; i < num_potential_groups * num_hash_table_sets; i++) {
        sums_of_quantity        [i] = 0;
        sums_of_base_price      [i] = 0;
        sums_of_charge          [i] = 0;
        sums_of_discounted_price[i] = 0;
        sums_of_discount        [i] = 0;
        record_counts           [i] = 0;
    }

    __syncthreads();

    auto global_thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    for(cardinality_t i = global_thread_index; i < num_tuples; i += input_stride) {
        if (true){  // (ship_date[i] <= threshold_ship_date)
            // TODO: Some of these calculations could work on uint32_t
            auto line_quantity         = quantity[i];
            auto line_discount         = discount[i];
            auto line_price            = extended_price[i];
            auto line_discount_factor  = Decimal64::ToValue(1, 0) - line_discount;
            auto line_discounted_price = Decimal64::Mul(line_discount_factor, line_price);
            auto line_tax_factor       = tax[i] + Decimal64::ToValue(1, 0);
            auto line_charge           = Decimal64::Mul(line_discounted_price, line_tax_factor);
            auto line_return_flag      = return_flag[i];
            auto line_status_          = line_status[i];

            int group_index =
                (encode_return_flag(line_return_flag) << line_status_bits) + encode_line_status(line_status_);

            atomicAdd( & lane_sum_of_quantity        [group_index * warp_size], line_quantity);
            atomicAdd( & lane_sum_of_base_price      [group_index * warp_size], line_price);
            atomicAdd( & lane_sum_of_charge          [group_index * warp_size], line_charge);
            atomicAdd( & lane_sum_of_discounted_price[group_index * warp_size], line_discounted_price);
            atomicAdd( & lane_sum_of_discount        [group_index * warp_size], line_discount);
            atomicAdd( & lane_record_counts          [group_index * warp_size], 1);
        }
    }

    // final aggregation

    for(int group_index = warp_index; group_index < num_potential_groups; group_index += warps_in_block) {
        atomicAdd( & sum_quantity        [group_index], lane_sum_of_quantity        [group_index * warp_size]);
        atomicAdd( & sum_base_price      [group_index], lane_sum_of_base_price      [group_index * warp_size]);
        atomicAdd( & sum_charge          [group_index], lane_sum_of_charge          [group_index * warp_size]);
        atomicAdd( & sum_discounted_price[group_index], lane_sum_of_discounted_price[group_index * warp_size]);
        atomicAdd( & sum_discount        [group_index], lane_sum_of_discount        [group_index * warp_size]);
        atomicAdd( & record_count        [group_index], lane_record_counts          [group_index * warp_size]);
    }
}

__global__
void tpch_query_01_compressed(
    sum_quantity_t*                      __restrict__ sum_quantity,
    sum_base_price_t*                    __restrict__ sum_base_price,
    sum_discounted_price_t*              __restrict__ sum_discounted_price,
    sum_charge_t*                        __restrict__ sum_charge,
    sum_discount_t*                      __restrict__ sum_discount,
    cardinality_t*                       __restrict__ record_count,
    const compressed::ship_date_t*       __restrict__ ship_date,
    const compressed::discount_t*        __restrict__ discount,
    const compressed::extended_price_t*  __restrict__ extended_price,
    const compressed::tax_t*             __restrict__ tax,
    const compressed::quantity_t*        __restrict__ quantity,
    const bit_container_t*               __restrict__ return_flag,
    const bit_container_t*               __restrict__ line_status,
    cardinality_t                                     num_tuples)
{
    __shared__ sum_quantity_t         sums_of_quantity         [num_potential_groups * num_hash_table_sets];
    __shared__ sum_base_price_t       sums_of_base_price       [num_potential_groups * num_hash_table_sets];
    __shared__ sum_discounted_price_t sums_of_discounted_price [num_potential_groups * num_hash_table_sets];
    __shared__ sum_charge_t           sums_of_charge           [num_potential_groups * num_hash_table_sets];
    __shared__ sum_discount_t         sums_of_discount         [num_potential_groups * num_hash_table_sets];
    __shared__ cardinality_t          record_counts            [num_potential_groups * num_hash_table_sets];
        // In each of these arrays, A single hash table is a strided sequence of elements (stride of
    	// warp size = number of banks = 32 elements; and that works both for the 64-bit and the 32-bit
    	// tables, apparently)

    auto warp_index = threadIdx.x / warp_size;
    auto warps_in_block = blockDim.x / warp_size;
    auto lane_index = threadIdx.x % warp_size;
    auto table_set_index = lane_index;

    auto lane_sum_of_quantity          = sums_of_quantity         + table_set_index;
    auto lane_sum_of_base_price        = sums_of_base_price       + table_set_index;
    auto lane_sum_of_charge            = sums_of_charge           + table_set_index;
    auto lane_sum_of_discounted_price  = sums_of_discounted_price + table_set_index;
    auto lane_sum_of_discount          = sums_of_discount         + table_set_index;
    auto lane_record_counts            = record_counts            + table_set_index;

    // all block threads which are to work on a certain hashtable will initialize that table

    for(int group_index = warp_index; group_index < num_potential_groups; group_index += warps_in_block) {
        lane_sum_of_quantity        [group_index * warp_size] = 0;
        lane_sum_of_base_price      [group_index * warp_size] = 0;
        lane_sum_of_charge          [group_index * warp_size] = 0;
        lane_sum_of_discounted_price[group_index * warp_size] = 0;
        lane_sum_of_discount        [group_index * warp_size] = 0;
        lane_record_counts          [group_index * warp_size] = 0;
    }

    __syncthreads();

    cardinality_t input_stride = (blockDim.x * gridDim.x); //Grid-Stride
    auto global_thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    for(cardinality_t i = global_thread_index; i < num_tuples; i += input_stride) {
        if (ship_date[i] <= compressed_threshold_ship_date) {
            // TODO: Some of these calculations could work on uint32_t
            auto line_quantity         = quantity[i];
            auto line_discount         = discount[i];
            auto line_price            = extended_price[i];
            auto line_discount_factor  = Decimal64::ToValue(1, 0) - line_discount;
            auto line_discounted_price = Decimal64::Mul(line_discount_factor, line_price);
            auto line_tax_factor       = tax[i] + Decimal64::ToValue(1, 0);
            auto line_charge           = Decimal64::Mul(line_discounted_price, line_tax_factor);
            auto line_return_flag      = get_bit_resolution_element<log_return_flag_bits, cardinality_t>(return_flag, i);
            auto line_status_          = get_bit_resolution_element<log_line_status_bits, cardinality_t>(line_status, i);

            int group_index = (line_return_flag << line_status_bits) + line_status_;

            atomicAdd( & lane_sum_of_quantity        [group_index * warp_size], line_quantity);
            atomicAdd( & lane_sum_of_base_price      [group_index * warp_size], line_price);
            atomicAdd( & lane_sum_of_charge          [group_index * warp_size], line_charge);
            atomicAdd( & lane_sum_of_discounted_price[group_index * warp_size], line_discounted_price);
            atomicAdd( & lane_sum_of_discount        [group_index * warp_size], line_discount);
            atomicAdd( & lane_record_counts          [group_index * warp_size], 1);
        }
    }

    // final aggregation

    for(int group_index = warp_index; group_index < num_potential_groups; group_index += warps_in_block) {
        atomicAdd( & sum_quantity        [group_index], lane_sum_of_quantity        [group_index * warp_size]);
        atomicAdd( & sum_base_price      [group_index], lane_sum_of_base_price      [group_index * warp_size]);
        atomicAdd( & sum_charge          [group_index], lane_sum_of_charge          [group_index * warp_size]);
        atomicAdd( & sum_discounted_price[group_index], lane_sum_of_discounted_price[group_index * warp_size]);
        atomicAdd( & sum_discount        [group_index], lane_sum_of_discount        [group_index * warp_size]);
        atomicAdd( & record_count        [group_index], lane_record_counts          [group_index * warp_size]);
    }
}

__global__
void tpch_query_01_compressed_precomputed_filter (
    sum_quantity_t*                      __restrict__ sum_quantity,
    sum_base_price_t*                    __restrict__ sum_base_price,
    sum_discounted_price_t*              __restrict__ sum_discounted_price,
    sum_charge_t*                        __restrict__ sum_charge,
    sum_discount_t*                      __restrict__ sum_discount,
    cardinality_t*                       __restrict__ record_count,
    const bit_container_t*               __restrict__ precomputed_filter,
    const compressed::discount_t*        __restrict__ discount,
    const compressed::extended_price_t*  __restrict__ extended_price,
    const compressed::tax_t*             __restrict__ tax,
    const compressed::quantity_t*        __restrict__ quantity,
    const bit_container_t*               __restrict__ return_flag,
    const bit_container_t*               __restrict__ line_status,
    cardinality_t                                     num_tuples)
 
{
    __shared__ sum_quantity_t         sums_of_quantity         [num_potential_groups * num_hash_table_sets];
    __shared__ sum_base_price_t       sums_of_base_price       [num_potential_groups * num_hash_table_sets];
    __shared__ sum_discounted_price_t sums_of_discounted_price [num_potential_groups * num_hash_table_sets];
    __shared__ sum_charge_t           sums_of_charge           [num_potential_groups * num_hash_table_sets];
    __shared__ sum_discount_t         sums_of_discount         [num_potential_groups * num_hash_table_sets];
    __shared__ cardinality_t          record_counts            [num_potential_groups * num_hash_table_sets];
        // In each of these arrays, A single hash table is a strided sequence of elements (stride of
    	// warp size = number of banks = 32 elements; and that works both for the 64-bit and the 32-bit
    	// tables, apparently)

    auto warp_index = threadIdx.x / warp_size;
    auto warps_in_block = blockDim.x / warp_size;
    auto lane_index = threadIdx.x % warp_size;
    auto table_set_index = lane_index;

    auto lane_sum_of_quantity          = sums_of_quantity         + table_set_index;
    auto lane_sum_of_base_price        = sums_of_base_price       + table_set_index;
    auto lane_sum_of_charge            = sums_of_charge           + table_set_index;
    auto lane_sum_of_discounted_price  = sums_of_discounted_price + table_set_index;
    auto lane_sum_of_discount          = sums_of_discount         + table_set_index;
    auto lane_record_counts            = record_counts            + table_set_index;

    // hash table initialization -
    // all block threads which are to work on a certain set of hash tables will initialize those tables

    for(int group_index = warp_index; group_index < num_potential_groups; group_index += warps_in_block) {
        lane_sum_of_quantity        [group_index * warp_size] = 0;
        lane_sum_of_base_price      [group_index * warp_size] = 0;
        lane_sum_of_charge          [group_index * warp_size] = 0;
        lane_sum_of_discounted_price[group_index * warp_size] = 0;
        lane_sum_of_discount        [group_index * warp_size] = 0;
        lane_record_counts          [group_index * warp_size] = 0;
    }

    __syncthreads();

    cardinality_t input_stride = (blockDim.x * gridDim.x); //Grid-Stride
    auto global_thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    for(cardinality_t i = global_thread_index; i < num_tuples; i += input_stride) {
        auto passes_filter = get_bit(precomputed_filter, i);
    	if (true) { // (passes_filter)
            // TODO: Some of these calculations could work on uint32_t
            auto line_quantity         = quantity[i];
            auto line_discount         = discount[i];
            auto line_price            = extended_price[i];
            auto line_discount_factor  = Decimal64::ToValue(1, 0) - line_discount;
            auto line_discounted_price = Decimal64::Mul(line_discount_factor, line_price);
            auto line_tax_factor       = tax[i] + Decimal64::ToValue(1, 0);
            auto line_charge           = Decimal64::Mul(line_discounted_price, line_tax_factor);
            auto line_return_flag      = get_bit_resolution_element<log_return_flag_bits, cardinality_t>(return_flag, i);
            auto line_status_          = get_bit_resolution_element<log_line_status_bits, cardinality_t>(line_status, i);

            int group_index = (line_return_flag << line_status_bits) + line_status_;

            atomicAdd( & lane_sum_of_quantity        [group_index * warp_size], line_quantity);
            atomicAdd( & lane_sum_of_base_price      [group_index * warp_size], line_price);
            atomicAdd( & lane_sum_of_charge          [group_index * warp_size], line_charge);
            atomicAdd( & lane_sum_of_discounted_price[group_index * warp_size], line_discounted_price);
            atomicAdd( & lane_sum_of_discount        [group_index * warp_size], line_discount);
            atomicAdd( & lane_record_counts          [group_index * warp_size], 1);
        }
    }

    __syncthreads();

    // final aggregation -
    // all block threads which have worked on a certain hash table set will collaboratively
    // update the global memory table with its contents

    for(int group_index = warp_index; group_index < num_potential_groups; group_index += warps_in_block) {
        atomicAdd( & sum_quantity        [group_index], lane_sum_of_quantity        [group_index * warp_size]);
        atomicAdd( & sum_base_price      [group_index], lane_sum_of_base_price      [group_index * warp_size]);
        atomicAdd( & sum_charge          [group_index], lane_sum_of_charge          [group_index * warp_size]);
        atomicAdd( & sum_discounted_price[group_index], lane_sum_of_discounted_price[group_index * warp_size]);
        atomicAdd( & sum_discount        [group_index], lane_sum_of_discount        [group_index * warp_size]);
        atomicAdd( & record_count        [group_index], lane_record_counts          [group_index * warp_size]);
    }
}

} // namespace one_table_per_bank
} // namespace shared_mem
} // namespace kernels
