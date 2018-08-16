#pragma once

#include "util/preprocessor_shorthands.hpp"
#include "util/atomics.cuh"
#include "constants.hpp"
#include "data_types.hpp"
#include "util/bit_operations.hpp"

namespace kernels {
namespace shared_mem {
namespace one_table_per_block {
	// i.e. NOT a single table per physical GPU core

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
    __shared__ sum_quantity_t         block_sum_of_quantity         [num_potential_groups];
    __shared__ sum_base_price_t       block_sum_of_base_price       [num_potential_groups];
    __shared__ sum_discounted_price_t block_sum_of_discounted_price [num_potential_groups];
    __shared__ sum_charge_t           block_sum_of_charge           [num_potential_groups];
    __shared__ sum_discount_t         block_sum_of_discount         [num_potential_groups];
    __shared__ cardinality_t          block_record_counts           [num_potential_groups];

    // The block's threads collaboratively initialize the table
    #pragma unroll
    for (int group_index = threadIdx.x; group_index < num_potential_groups; group_index += blockDim.x) {
        block_sum_of_quantity        [group_index] = 0;
        block_sum_of_base_price      [group_index] = 0;
        block_sum_of_charge          [group_index] = 0;
        block_sum_of_discounted_price[group_index] = 0;
        block_sum_of_discount        [group_index] = 0;
        block_record_counts          [group_index] = 0;
    }

    __syncthreads();

    cardinality_t input_stride = (blockDim.x * gridDim.x); //Grid-Stride
    auto global_thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    for(cardinality_t i = global_thread_index; i < num_tuples; i += input_stride) {

        if (ship_date[i] <= threshold_ship_date) {
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

            atomicAdd( & block_sum_of_quantity        [group_index], line_quantity);
            atomicAdd( & block_sum_of_base_price      [group_index], line_price);
            atomicAdd( & block_sum_of_charge          [group_index], line_charge);
            atomicAdd( & block_sum_of_discounted_price[group_index], line_discounted_price);
            atomicAdd( & block_sum_of_discount        [group_index], line_discount);
            atomicAdd( & block_record_counts          [group_index], 1);
        }
    }

    __syncthreads();

    // The block's threads collaboratively merge the (single) block table into the global one
    #pragma unroll
    for (int group_index = threadIdx.x; group_index < num_potential_groups; group_index += blockDim.x) {
        atomicAdd( & sum_quantity        [group_index], block_sum_of_quantity        [group_index]);
        atomicAdd( & sum_base_price      [group_index], block_sum_of_base_price      [group_index]);
        atomicAdd( & sum_charge          [group_index], block_sum_of_charge          [group_index]);
        atomicAdd( & sum_discounted_price[group_index], block_sum_of_discounted_price[group_index]);
        atomicAdd( & sum_discount        [group_index], block_sum_of_discount        [group_index]);
        atomicAdd( & record_count        [group_index], block_record_counts          [group_index]);
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
    __shared__ sum_quantity_t         block_sum_of_quantity         [num_potential_groups];
    __shared__ sum_base_price_t       block_sum_of_base_price       [num_potential_groups];
    __shared__ sum_discounted_price_t block_sum_of_discounted_price [num_potential_groups];
    __shared__ sum_charge_t           block_sum_of_charge           [num_potential_groups];
    __shared__ sum_discount_t         block_sum_of_discount         [num_potential_groups];
    __shared__ cardinality_t          block_record_counts           [num_potential_groups];

    // The block's threads collaboratively initialize the table
    #pragma unroll
    for (int group_index = threadIdx.x; group_index < num_potential_groups; group_index += blockDim.x) {
        block_sum_of_quantity        [group_index] = 0;
        block_sum_of_base_price      [group_index] = 0;
        block_sum_of_charge          [group_index] = 0;
        block_sum_of_discounted_price[group_index] = 0;
        block_sum_of_discount        [group_index] = 0;
        block_record_counts          [group_index] = 0;
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

            atomicAdd( & block_sum_of_quantity        [group_index], line_quantity);
            atomicAdd( & block_sum_of_base_price      [group_index], line_price);
            atomicAdd( & block_sum_of_charge          [group_index], line_charge);
            atomicAdd( & block_sum_of_discounted_price[group_index], line_discounted_price);
            atomicAdd( & block_sum_of_discount        [group_index], line_discount);
            atomicAdd( & block_record_counts          [group_index], 1);
        }
    }

    __syncthreads();

    // The block's threads collaboratively merge the (single) block table into the global one
    #pragma unroll
    for (int group_index = threadIdx.x; group_index < num_potential_groups; group_index += blockDim.x) {
        atomicAdd( & sum_quantity        [group_index], block_sum_of_quantity        [group_index]);
        atomicAdd( & sum_base_price      [group_index], block_sum_of_base_price      [group_index]);
        atomicAdd( & sum_charge          [group_index], block_sum_of_charge          [group_index]);
        atomicAdd( & sum_discounted_price[group_index], block_sum_of_discounted_price[group_index]);
        atomicAdd( & sum_discount        [group_index], block_sum_of_discount        [group_index]);
        atomicAdd( & record_count        [group_index], block_record_counts          [group_index]);
    }
}

__global__
void tpch_query_01_compressed_precomputed_filter(
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
    __shared__ sum_quantity_t         block_sum_of_quantity         [num_potential_groups];
    __shared__ sum_base_price_t       block_sum_of_base_price       [num_potential_groups];
    __shared__ sum_discounted_price_t block_sum_of_discounted_price [num_potential_groups];
    __shared__ sum_charge_t           block_sum_of_charge           [num_potential_groups];
    __shared__ sum_discount_t         block_sum_of_discount         [num_potential_groups];
    __shared__ cardinality_t          block_record_counts           [num_potential_groups];

    // The block's threads collaboratively initialize the table
    #pragma unroll
    for (int group_index = threadIdx.x; group_index < num_potential_groups; group_index += blockDim.x) {
        block_sum_of_quantity        [group_index] = 0;
        block_sum_of_base_price      [group_index] = 0;
        block_sum_of_charge          [group_index] = 0;
        block_sum_of_discounted_price[group_index] = 0;
        block_sum_of_discount        [group_index] = 0;
        block_record_counts          [group_index] = 0;
    }

    __syncthreads();

    cardinality_t input_stride = (blockDim.x * gridDim.x); //Grid-Stride
    auto global_thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    for(cardinality_t i = global_thread_index; i < num_tuples; i += input_stride) {
        auto passes_filter = get_bit(precomputed_filter, i);
    	if (passes_filter) {
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

            atomicAdd( & block_sum_of_quantity        [group_index], line_quantity);
            atomicAdd( & block_sum_of_base_price      [group_index], line_price);
            atomicAdd( & block_sum_of_charge          [group_index], line_charge);
            atomicAdd( & block_sum_of_discounted_price[group_index], line_discounted_price);
            atomicAdd( & block_sum_of_discount        [group_index], line_discount);
            atomicAdd( & block_record_counts          [group_index], 1);
        }
    }

    __syncthreads();

    // The block's threads collaboratively merge the (single) block table into the global one
    #pragma unroll
    for (int group_index = threadIdx.x; group_index < num_potential_groups; group_index += blockDim.x) {
        atomicAdd( & sum_quantity        [group_index], block_sum_of_quantity        [group_index]);
        atomicAdd( & sum_base_price      [group_index], block_sum_of_base_price      [group_index]);
        atomicAdd( & sum_charge          [group_index], block_sum_of_charge          [group_index]);
        atomicAdd( & sum_discounted_price[group_index], block_sum_of_discounted_price[group_index]);
        atomicAdd( & sum_discount        [group_index], block_sum_of_discount        [group_index]);
        atomicAdd( & record_count        [group_index], block_record_counts          [group_index]);
    }
}

} // namespace one_table_per_block
} // namespace shared_mem
} // namespace kernels
