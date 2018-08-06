#pragma once
#ifndef EXECUTE_Q1_HPP_
#define EXECUTE_Q1_HPP_

#include "common.hpp"
#include "util/helper.hpp"
#include "cpu.h"

#include <vector>
#include <cuda/api_wrappers.h>
#include <memory>

template <template <typename> class Ptr, bool Compressed>
struct input_buffer_set;

template <template <typename> class Ptr>
struct input_buffer_set<Ptr, is_compressed> {
    Ptr< compressed::ship_date_t[]      > ship_date;
    Ptr< compressed::discount_t[]       > discount;
    Ptr< compressed::extended_price_t[] > extended_price;
    Ptr< compressed::tax_t[]            > tax;
    Ptr< compressed::quantity_t[]       > quantity;
    Ptr< bit_container_t[]              > return_flag;
    Ptr< bit_container_t[]              > line_status;
    Ptr< bit_container_t[]              > precomputed_filter;
};

template <template <typename> class Ptr>
struct input_buffer_set<Ptr, is_not_compressed> {
    Ptr< ship_date_t[]      > ship_date;
    Ptr< discount_t[]       > discount;
    Ptr< extended_price_t[] > extended_price;
    Ptr< tax_t[]            > tax;
    Ptr< quantity_t[]       > quantity;
    Ptr< return_flag_t[]    > return_flag;
    Ptr< line_status_t[]    > line_status;
};


// Note: This should be a variant; or we could just templatize more.
struct stream_input_buffer_sets {
    std::vector<input_buffer_set<cuda::memory::device::unique_ptr, is_not_compressed > > uncompressed;
    std::vector<input_buffer_set<cuda::memory::device::unique_ptr, is_compressed     > > compressed;
};

template <template <typename> class UniquePtr>
struct aggregates_set {
    UniquePtr<sum_quantity_t[]        > sum_quantity;
    UniquePtr<sum_base_price_t[]      > sum_base_price;
    UniquePtr<sum_discounted_price_t[]> sum_discounted_price;
    UniquePtr<sum_charge_t[]          > sum_charge;
    UniquePtr<sum_discount_t[]        > sum_discount;
    UniquePtr<cardinality_t[]         > record_count;
};

using host_aggregates_t = aggregates_set<plugged_unique_ptr>;
using device_aggregates_t = aggregates_set<cuda::memory::device::unique_ptr>;

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
);

extern CoProc* cpu_coprocessor;


#endif // EXECUTE_Q1_HPP_
