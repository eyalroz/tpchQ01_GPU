#include "common.hpp"
#include "parse_cmdline.hpp"
#include "execute_q1.hpp"
#include "data_types.hpp"
#include "constants.hpp"
#include "expl_comp_strat/tpch_kit.hpp"
#include "cpu/common.hpp"
#include "cpu.hpp"

#include "util/helper.hpp"
#include "util/extra_pointer_traits.hpp"
#include "util/bit_operations.hpp"
#include "util/file_access.hpp"

#include <iostream>
#include <cuda/api_wrappers.h>
#include <vector>
#include <iomanip>
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

CoProc* cpu_coprocessor = nullptr;

using timer = std::chrono::high_resolution_clock;


template <typename T>
void print_results(const T& aggregates_on_host, cardinality_t cardinality) {
    cout << "+---------------------------------------------------- Results ------------------------------------------------------+\n";
    cout << "|  LS | RF |  sum_quantity        |  sum_base_price      |  sum_disc_price      |  sum_charge          | count      |\n";
    cout << "+-------------------------------------------------------------------------------------------------------------------+\n";
    auto print_dec = [] (auto s, auto x) { printf("%s%17ld.%02ld", s, Decimal64::GetInt(x), Decimal64::GetFrac(x)); };
    cardinality_t total_passing { 0 };

    for (int group=0; group<num_potential_groups; group++) {
        if (true) { // (aggregates_on_host.record_count[group] > 0) {
            char rf = decode_return_flag(group >> line_status_bits);
            char ls = decode_line_status(group & 0b1);
            int idx;

            if (rf == 'A' and ls == 'F') {
                idx = 393;
                if (cardinality == 6001215) {
                    assert(aggregates_on_host.sum_quantity[group] == 3773410700);
                    assert(aggregates_on_host.record_count[group] == 1478493);
                }
            } else if (rf == 'N' and ls == 'F') {
                idx = 406;
                if (cardinality == 6001215) {
                    assert(aggregates_on_host.sum_quantity[group] == 99141700);
                    assert(aggregates_on_host.record_count[group] == 38854);
                }
            } else if (rf == 'N' and ls == 'O') {
                idx = 415;
                rf = 'N';
                ls = 'O';
                if (cardinality == 6001215) {
                    assert(aggregates_on_host.sum_quantity[group] == 7447604000);
                    assert(aggregates_on_host.record_count[group] == 2920374);
                }
            } else if (rf == 'R' and ls == 'F') {
                idx = 410;
                if (cardinality == 6001215) {
                    assert(aggregates_on_host.sum_quantity[group]== 3771975300);
                    assert(aggregates_on_host.record_count[group]== 1478870);
                }
            } else {
                continue;
            }

            if (cpu_coprocessor) {
                auto t = &cpu_coprocessor->table[idx];
                auto& a = aggregates_on_host;

                a.sum_quantity[group] += t->sum_quantity;
                a.record_count[group] += t->count;
                
                a.sum_base_price[group] += t->sum_base_price;
                a.sum_discounted_price[group] += t->sum_disc_price;

                a.sum_charge[group] += t->sum_charge;
                a.sum_discount[group] += t->sum_disc;
            }
                
            printf("| # %c | %c ", rf, ls);
            print_dec(" | ",  aggregates_on_host.sum_quantity.get()[group]);
            print_dec(" | ",  aggregates_on_host.sum_base_price.get()[group]);
            print_dec(" | ",  aggregates_on_host.sum_discounted_price.get()[group]);
            print_dec(" | ",  aggregates_on_host.sum_charge.get()[group]);
            printf(" | %10u |\n", aggregates_on_host.record_count.get()[group]);
            total_passing += aggregates_on_host.record_count.get()[group];
        }
    }
    cout << "+-------------------------------------------------------------------------------------------------------------------+\n";
    cout
        << "Total number of elements tuples satisfying the WHERE clause: "
        << total_passing << " / " << cardinality << " = " << std::dec
        << (double) total_passing / cardinality << "\n";
}

void precompute_filter_for_table_chunk(
    const compressed::ship_date_t*  __restrict__  compressed_ship_date,
    bit_container_t*                __restrict__  precomputed_filter,
    cardinality_t                                 num_tuples)
{
    // Note: we assume ana aligned beginning, i.e. that the number of tuples per launch is
    // a multiple of bits_per_container
    cardinality_t end_offset = num_tuples;
    cardinality_t end_offset_in_full_containers = end_offset - end_offset % bits_per_container;
    for(cardinality_t i = 0; i < end_offset_in_full_containers; i += bits_per_container) {
        bit_container_t bit_container { 0 };
        for(int j = 0; j < bits_per_container; j++) {
            // Note this relies on the little-endianness of nVIDIA GPUs
            auto evaluated_where_clause = compressed_ship_date[i+j] <= compressed_threshold_ship_date;
            bit_container |= bit_container_t{evaluated_where_clause} << j;
        }
        precomputed_filter[i / bits_per_container] = bit_container;
    }
    if (end_offset > end_offset_in_full_containers) {
        bit_container_t bit_container { 0 };
        for(int j = 0; j + end_offset_in_full_containers < end_offset; j++) {
            auto evaluated_where_clause =
                compressed_ship_date[end_offset_in_full_containers+j] <= compressed_threshold_ship_date;
            bit_container |= bit_container_t{evaluated_where_clause} << j;
        }
        precomputed_filter[end_offset / bits_per_container] = bit_container;
    }
}

bool columns_are_cached(
    const q1_params_t&  params,
    bool                looking_for_compressed_columns)
{
    auto data_files_directory =
        filesystem::path(defaults::tpch_data_subdirectory) / std::to_string(params.scale_factor);
    auto some_cached_column_filename = data_files_directory /
        (std::string(looking_for_compressed_columns ? "compressed_" : "") + "shipdate.bin");
    return filesystem::exists(some_cached_column_filename);
    // TODO: We could check that _all_ cache files are present instead of just an arbirary one.
}

template <template <typename> class UniquePtr, bool Compressed>
cardinality_t load_cached_columns(
    const q1_params_t&                         params,
    input_buffer_set<UniquePtr, Compressed>&   buffer_set)
{
    auto data_files_directory =
        filesystem::path(defaults::tpch_data_subdirectory) / std::to_string(params.scale_factor);
    std::string filename_prefix = Compressed ? "compressed_" : "";
    auto shipdate_element_size = Compressed ? sizeof(compressed::ship_date_t) : sizeof(ship_date_t);
    auto cached_compressed_shipdate_filename = data_files_directory / (filename_prefix +  "shipdate.bin");
    cardinality_t cardinality = filesystem::file_size(cached_compressed_shipdate_filename) / shipdate_element_size;
    if (cardinality == cardinality_of_scale_factor_1) {
        // This handles sub-scale-factor-1 arguments, for which the actual table and column files are SF 1
        cardinality = ((double) cardinality) * params.scale_factor;
    }
    if (cardinality == 0) {
        throw std::runtime_error("The lineitem table column cardinality should not be 0");
    }

    cardinality_t return_flag_container_count =
        Compressed ? div_rounding_up(cardinality, return_flag_values_per_container) : cardinality;
    cardinality_t line_status_container_count =
        Compressed ? div_rounding_up(cardinality, line_status_values_per_container) : cardinality;

    buffer_set.ship_date      = extra_pointer_traits<decltype(buffer_set.ship_date      )>::make(cardinality);
    buffer_set.tax            = extra_pointer_traits<decltype(buffer_set.tax)            >::make(cardinality);
    buffer_set.discount       = extra_pointer_traits<decltype(buffer_set.discount)       >::make(cardinality);
    buffer_set.quantity       = extra_pointer_traits<decltype(buffer_set.quantity)       >::make(cardinality);
    buffer_set.extended_price = extra_pointer_traits<decltype(buffer_set.extended_price) >::make(cardinality);
    buffer_set.return_flag    = extra_pointer_traits<decltype(buffer_set.return_flag)    >::make(return_flag_container_count);
    buffer_set.line_status    = extra_pointer_traits<decltype(buffer_set.line_status)    >::make(line_status_container_count);
    load_column_from_binary_file(buffer_set.ship_date.get(),      cardinality, data_files_directory, filename_prefix + "shipdate"      + ".bin");
    load_column_from_binary_file(buffer_set.discount.get(),       cardinality, data_files_directory, filename_prefix + "discount"      + ".bin");
    load_column_from_binary_file(buffer_set.tax.get(),            cardinality, data_files_directory, filename_prefix + "tax"           + ".bin");
    load_column_from_binary_file(buffer_set.quantity.get(),       cardinality, data_files_directory, filename_prefix + "quantity"      + ".bin");
    load_column_from_binary_file(buffer_set.extended_price.get(), cardinality, data_files_directory, filename_prefix + "extendedprice" + ".bin");
    load_column_from_binary_file(buffer_set.return_flag.get(),    return_flag_container_count, data_files_directory, filename_prefix + "returnflag"    + ".bin");
    load_column_from_binary_file(buffer_set.line_status.get(),    line_status_container_count, data_files_directory, filename_prefix + "linestatus"    + ".bin");



/*
    for_each_argument(
        [&](auto tup){
            auto& ptr           = std::get<0>(tup);
            const auto& name    = std::get<1>(tup);
            cardinality_t count = std::get<2>(tup);
            using traits = typename extra_pointer_traits<decltype(ptr)>;
            ptr = traits::make(count);
            load_column_from_binary_file(
                ptr.get(), cardinality, data_files_directory, filename_prefix + name + ".bin");
        },
        tie(buffer_set.ship_date,      "shipdate",       cardinality                 ),
        tie(buffer_set.discount,       "discount",       cardinality                 ),
        tie(buffer_set.tax,            "tax",            cardinality                 ),
        tie(buffer_set.quantity,       "quantity",       cardinality                 ),
        tie(buffer_set.extended_price, "extendedprice",  cardinality                 ),
        tie(buffer_set.return_flag,    "returnflag",     return_flag_container_count ),
        tie(buffer_set.line_status,    "linestatus",     line_status_container_count )
    );
*/
    return cardinality;
}

template <template <typename> class Ptr, bool Compressed>
void write_columns_to_cache(
    q1_params_t                         params,
    input_buffer_set<Ptr, Compressed>&  buffer_set,
    cardinality_t                       cardinality)
{
    auto data_files_directory =
        filesystem::path(defaults::tpch_data_subdirectory) / std::to_string(params.scale_factor);
    std::string filename_prefix = Compressed ? "compressed_" : "";
    cardinality_t return_flag_container_count = Compressed ? div_rounding_up(cardinality, return_flag_values_per_container) : cardinality;
    cardinality_t line_status_container_count = Compressed ? div_rounding_up(cardinality, line_status_values_per_container) : cardinality;

    write_column_to_binary_file(&buffer_set.ship_date[0],      cardinality, data_files_directory, filename_prefix + "shipdate" + ".bin");
    write_column_to_binary_file(&buffer_set.discount[0],       cardinality, data_files_directory, filename_prefix + "discount" + ".bin");
    write_column_to_binary_file(&buffer_set.tax[0],            cardinality, data_files_directory, filename_prefix + "tax" + ".bin");
    write_column_to_binary_file(&buffer_set.quantity[0],       cardinality, data_files_directory, filename_prefix + "quantity" + ".bin");
    write_column_to_binary_file(&buffer_set.extended_price[0], cardinality, data_files_directory, filename_prefix + "extendedprice" + ".bin");
    write_column_to_binary_file(&buffer_set.return_flag[0],    return_flag_container_count, data_files_directory, filename_prefix + "returnflag" + ".bin");
    write_column_to_binary_file(&buffer_set.line_status[0],    line_status_container_count, data_files_directory, filename_prefix + "linestatus" + ".bin");

/*    for_each_argument(
        [&](auto tup){
            const auto& ptr     = std::get<0>(tup);
            auto raw_ptr        = // extra_pointer_traits<decltype(ptr)>::get();
                &ptr[0];
            const auto& name    = std::get<1>(tup);
            cardinality_t count = std::get<2>(tup);
            write_column_to_binary_file(
                raw_ptr, count, data_files_directory, filename_prefix + name + ".bin");
        },
        tie(buffer_set.ship_date,      "shipdate",      cardinality                 ),
        tie(buffer_set.discount,       "discount",      cardinality                 ),
        tie(buffer_set.tax,            "tax",           cardinality                 ),
        tie(buffer_set.quantity,       "quantity",      cardinality                 ),
        tie(buffer_set.extended_price, "extendedprice", cardinality                 ),
        tie(buffer_set.return_flag,    "returnflag",    return_flag_container_count ),
        tie(buffer_set.line_status,    "linestatus",    line_status_container_count )
    );
    */
}

cardinality_t parse_table_file_into_columns(
    const q1_params_t&      params,
    lineitem&               li)
{
    cardinality_t cardinality;

    auto data_files_directory =
        filesystem::path(defaults::tpch_data_subdirectory) / std::to_string(params.scale_factor);
    // TODO: Take this out into a script

    filesystem::create_directory(defaults::tpch_data_subdirectory);
    filesystem::create_directory(data_files_directory);
    auto table_file_path = data_files_directory / lineitem_table_file_name;
    cout << "Parsing the lineitem table in file " << table_file_path << endl;
    if (not filesystem::exists(table_file_path)) {
        throw std::runtime_error("Cannot locate table text file " + table_file_path.string());
        // Not generating it ourselves - that's: 1. Not healthy and 2. Not portable;
        // setup scripts are intended to do that
    }
    li.FromFile(table_file_path.c_str());
    cardinality = li.l_extendedprice.cardinality;
    if (cardinality == cardinality_of_scale_factor_1) {
        cardinality = ((double) cardinality) * params.scale_factor;
    }
    if (cardinality == 0) {
        throw std::runtime_error("The lineitem table column cardinality should not be 0");
    }
    cout << "CSV read & parsed; table length: " << cardinality << " records." << endl;
    return cardinality;
}

input_buffer_set<cuda::memory::host::unique_ptr, is_compressed> compress_columns(
    input_buffer_set<plain_ptr, is_not_compressed>  uncompressed,
    cardinality_t                                   cardinality)
{
    input_buffer_set<cuda::memory::host::unique_ptr, is_compressed> compressed = {
        cuda::memory::host::make_unique< compressed::ship_date_t[]      >(cardinality),
        cuda::memory::host::make_unique< compressed::discount_t[]       >(cardinality),
        cuda::memory::host::make_unique< compressed::extended_price_t[] >(cardinality),
        cuda::memory::host::make_unique< compressed::tax_t[]            >(cardinality),
        cuda::memory::host::make_unique< compressed::quantity_t[]       >(cardinality),
        cuda::memory::host::make_unique< bit_container_t[] >(div_rounding_up(cardinality, return_flag_values_per_container)),
        cuda::memory::host::make_unique< bit_container_t[] >(div_rounding_up(cardinality, line_status_values_per_container)),
        nullptr // precomputed filter - we don't create this here.
    };

    cout << "Compressing column data... " << flush;

    // Man, we really need to have a sub-byte-length-value container class
    std::memset(compressed.return_flag.get(), 0, div_rounding_up(cardinality, return_flag_values_per_container));
    std::memset(compressed.line_status.get(), 0, div_rounding_up(cardinality, line_status_values_per_container));
    for(cardinality_t i = 0; i < cardinality; i++) {
        compressed.ship_date[i]      = uncompressed.ship_date[i] - ship_date_frame_of_reference;
        compressed.discount[i]       = uncompressed.discount[i]; // we're keeping the factor 100 scaling
        compressed.extended_price[i] = uncompressed.extended_price[i];
        compressed.quantity[i]       = uncompressed.quantity[i] / 100;
        compressed.tax[i]            = uncompressed.tax[i]; // we're keeping the factor 100 scaling
        set_bit_resolution_element<log_return_flag_bits, cardinality_t>(
            compressed.return_flag.get(), i, encode_return_flag(uncompressed.return_flag[i]));
        set_bit_resolution_element<log_line_status_bits, cardinality_t>(
            compressed.line_status.get(), i, encode_line_status(uncompressed.line_status[i]));
        assert( (ship_date_t)      compressed.ship_date[i]      == uncompressed.ship_date[i] - ship_date_frame_of_reference);
        assert( (discount_t)       compressed.discount[i]       == uncompressed.discount[i]);
        assert( (extended_price_t) compressed.extended_price[i] == uncompressed.extended_price[i]);
        assert( (quantity_t)       compressed.quantity[i]       == uncompressed.quantity[i] / 100);
            // not keeping the scaling here since we know the data is all integral; you could call this a form
            // of compression
        assert( (tax_t)            compressed.tax[i]            == uncompressed.tax[i]);
    }
    for(cardinality_t i = 0; i < cardinality; i++) {
        assert(decode_return_flag(get_bit_resolution_element<log_return_flag_bits, cardinality_t>(compressed.return_flag.get(), i)) == uncompressed.return_flag[i]);
        assert(decode_line_status(get_bit_resolution_element<log_line_status_bits, cardinality_t>(compressed.line_status.get(), i)) == uncompressed.line_status[i]);
    }

    cout << "done." << endl;
    return compressed;
}

input_buffer_set<plain_ptr, is_not_compressed>
get_buffers_inside(lineitem& li)
{
    return {
        li.l_shipdate.get(),
        li.l_discount.get(),
        li.l_extendedprice.get(),
        li.l_tax.get(),
        li.l_quantity.get(),
        li.l_returnflag.get(),
        li.l_linestatus.get()
    };
};


/*
 * We have to do this, since the lineitem destructor releases... :-(
 */
void move_buffers_into_lineitem_object(
    input_buffer_set<plugged_unique_ptr, is_not_compressed>&  outside_buffer_set,
    lineitem&                                                 li,
    cardinality_t                                             cardinality)
{
    for_each_argument(
        [&](auto tup){
            std::get<0>(tup).cardinality = cardinality;
            std::get<0>(tup).m_ptr = std::get<1>(tup).release();
        },
        tie(li.l_shipdate,       outside_buffer_set.ship_date),
        tie(li.l_discount,       outside_buffer_set.discount),
        tie(li.l_tax,            outside_buffer_set.tax),
        tie(li.l_quantity,       outside_buffer_set.quantity),
        tie(li.l_extendedprice,  outside_buffer_set.extended_price),
        tie(li.l_returnflag,     outside_buffer_set.return_flag),
        tie(li.l_linestatus,     outside_buffer_set.line_status)
    );
}

void set_lineitem_cardinalities(
    lineitem&                                                 li,
    cardinality_t                                             cardinality)
{
    li.l_shipdate.cardinality = cardinality;
    li.l_discount.cardinality = cardinality;
    li.l_tax.cardinality = cardinality;
    li.l_quantity.cardinality = cardinality;
    li.l_extendedprice.cardinality = cardinality;
    li.l_returnflag.cardinality = cardinality;
    li.l_linestatus.cardinality = cardinality;
}

void allocate_non_input_resources(
    q1_params_t                     params,
    cuda::device_t<>                cuda_device,
    cardinality_t                   cardinality,
    device_aggregates_t&            aggregates_on_device,
    host_aggregates_t&              aggregates_on_host,
    stream_input_buffer_sets&       stream_input_buffer_sets
)
{
    aggregates_on_host = {
        std::make_unique< sum_quantity_t[]         >(num_potential_groups),
        std::make_unique< sum_base_price_t[]       >(num_potential_groups),
        std::make_unique< sum_discounted_price_t[] >(num_potential_groups),
        std::make_unique< sum_charge_t []          >(num_potential_groups),
        std::make_unique< sum_discount_t[]         >(num_potential_groups),
        std::make_unique< cardinality_t[]          >(num_potential_groups)
    };

    aggregates_on_device = {
        cuda::memory::device::make_unique< sum_quantity_t[]         >(cuda_device, num_potential_groups),
        cuda::memory::device::make_unique< sum_base_price_t[]       >(cuda_device, num_potential_groups),
        cuda::memory::device::make_unique< sum_discounted_price_t[] >(cuda_device, num_potential_groups),
        cuda::memory::device::make_unique< sum_charge_t []          >(cuda_device, num_potential_groups),
        cuda::memory::device::make_unique< sum_discount_t[]         >(cuda_device, num_potential_groups),
        cuda::memory::device::make_unique< cardinality_t[]          >(cuda_device, num_potential_groups)
    };

    if (params.apply_compression) {
        stream_input_buffer_sets.compressed.reserve(params.num_gpu_streams);
    } else {
        stream_input_buffer_sets.uncompressed.reserve(params.num_gpu_streams);
    }

    for (int i = 0; i < params.num_gpu_streams; ++i) {
        if (params.apply_compression) {
            auto stream_input_buffer_set = input_buffer_set<cuda::memory::device::unique_ptr, is_compressed>{
                cuda::memory::device::make_unique< compressed::ship_date_t[]      >(cuda_device, params.num_tuples_per_kernel_launch),
                cuda::memory::device::make_unique< compressed::discount_t[]       >(cuda_device, params.num_tuples_per_kernel_launch),
                cuda::memory::device::make_unique< compressed::extended_price_t[] >(cuda_device, params.num_tuples_per_kernel_launch),
                cuda::memory::device::make_unique< compressed::tax_t[]            >(cuda_device, params.num_tuples_per_kernel_launch),
                cuda::memory::device::make_unique< compressed::quantity_t[]       >(cuda_device, params.num_tuples_per_kernel_launch),
                cuda::memory::device::make_unique< bit_container_t[]              >(cuda_device, div_rounding_up(params.num_tuples_per_kernel_launch, return_flag_values_per_container)),
                cuda::memory::device::make_unique< bit_container_t[]              >(cuda_device, div_rounding_up(params.num_tuples_per_kernel_launch, line_status_values_per_container)),
                cuda::memory::device::make_unique< bit_container_t[]              >(cuda_device, div_rounding_up(params.num_tuples_per_kernel_launch, bits_per_container))
            };
            stream_input_buffer_sets.compressed.emplace_back(std::move(stream_input_buffer_set));
        }
        else {
            auto stream_input_buffer_set = input_buffer_set<cuda::memory::device::unique_ptr, is_not_compressed>{
                cuda::memory::device::make_unique< ship_date_t[]      >(cuda_device, params.num_tuples_per_kernel_launch),
                cuda::memory::device::make_unique< discount_t[]       >(cuda_device, params.num_tuples_per_kernel_launch),
                cuda::memory::device::make_unique< extended_price_t[] >(cuda_device, params.num_tuples_per_kernel_launch),
                cuda::memory::device::make_unique< tax_t[]            >(cuda_device, params.num_tuples_per_kernel_launch),
                cuda::memory::device::make_unique< quantity_t[]       >(cuda_device, params.num_tuples_per_kernel_launch),
                cuda::memory::device::make_unique< return_flag_t[]    >(cuda_device, params.num_tuples_per_kernel_launch),
                cuda::memory::device::make_unique< line_status_t[]    >(cuda_device, params.num_tuples_per_kernel_launch),
            };
            stream_input_buffer_sets.uncompressed.emplace_back(std::move(stream_input_buffer_set));
        }
    }
}

int main(int argc, char** argv) {
    make_sure_we_are_on_cpu_core_0();

    auto params = parse_command_line(argc, argv);
    morsel_size = params.num_tuples_per_kernel_launch;
    cardinality_t cardinality;

    lineitem li((size_t)(max_line_items_per_sf * std::max(params.scale_factor, 1.0)));
        // Note: lineitem should really not need this cap, it should just adjust
        // allocated space as the need arises (and start with an estimate based on
        // the file size
    input_buffer_set<plugged_unique_ptr, is_not_compressed> uncompressed_outside_li;
    input_buffer_set<plain_ptr, is_not_compressed> uncompressed;
        // If we parse, we have to go through an "lineitem" object which holds its
        // own pointers, while we need this class, which is the same template for
        // the compressed as uncompressed case. For this reason we have these two
        // objects, using the first, moving it into li, then using the second as
        // a sort of a facade for li.

    input_buffer_set<cuda::memory::host::unique_ptr, is_compressed> compressed;
        // Compressed columns are handled entirely independently of lineitem objects,
        // so we don't need the two input_buffer_set objects

    auto columns_to_process_are_cached = columns_are_cached(params, params.apply_compression);

    if (columns_to_process_are_cached) {
        if (params.apply_compression) {
            cardinality = load_cached_columns(params, compressed);
            set_lineitem_cardinalities(li, cardinality);
        }
        else {
            cardinality = load_cached_columns(params, uncompressed_outside_li);
            move_buffers_into_lineitem_object(uncompressed_outside_li, li, cardinality);
            uncompressed = get_buffers_inside(li);
        }
    }
    else {
        if (columns_are_cached(params, is_not_compressed)) {
            cardinality = load_cached_columns(params, uncompressed_outside_li);
            move_buffers_into_lineitem_object(uncompressed_outside_li, li, cardinality);
            uncompressed = get_buffers_inside(li);
        }
        else {
            cardinality = parse_table_file_into_columns(params, li);
            uncompressed = get_buffers_inside(li);
            write_columns_to_cache(params, uncompressed, cardinality);
                // We write the uncompressed columns to cache files
                // even if our interest is in the compressed ones
        }

        if (params.apply_compression) {
            compressed = compress_columns(uncompressed, cardinality);
            write_columns_to_cache(params, compressed, cardinality);
        }
    }

    if (params.use_filter_pushdown) {
        assert(params.apply_compression);
        compressed.precomputed_filter =
            cuda::memory::host::make_unique< bit_container_t[] >(div_rounding_up(cardinality, bits_per_container));
    }

    cpu_coprocessor = (params.use_coprocessing or params.use_filter_pushdown) ?  new CoProc(li, true) : nullptr;

    // We don't need li beyond this point. Actually, we should need it at all except dfor parsing perhaps


    // Note:
    // We are not timing the host-side allocations here. In a real DBMS, these will likely only be
    // a few sub-allocations, which would take very little time (dozens of clock cycles overall) -
    // no system calls.

    device_aggregates_t        aggregates_on_device;
    host_aggregates_t          aggregates_on_host;
    stream_input_buffer_sets   stream_input_buffer_sets;


    auto cuda_device = cuda::device::get(params.cuda_device_id);

    std::vector<cuda::stream_t<>>  streams;
    streams.reserve(params.num_gpu_streams);
        // We'll be scheduling (most of) our work in a round-robin fashion on all of
        // the streams, to prevent the GPU from idling.
    for(int i = 0; i < params.num_gpu_streams; i++) {
        auto stream = cuda_device.create_stream(cuda::stream::async);
        streams.emplace_back(std::move(stream));
    }

    // Note:
    // We are not timing the allocations here. In a real DBMS, actual CUDA allocations would
    // happen with the DBMS is brought up, and when a query is processed, it will only be
    // a few sub-allocations, which would take very little time (dozens of clock cycles overall) -
    // no CUDA API nor system calls. We _will_, however, time the initialization of the buffers.
    allocate_non_input_resources(
        params,
        cuda_device,
        cardinality,
        aggregates_on_device,
        aggregates_on_host,
        stream_input_buffer_sets);

    std::ofstream results_file;
    results_file.open("results.csv", std::ios::out);

    cuda::profiling::start();
    std::stringstream ss;
    ss 
       << "Binary: " << argv[0] << " | "
       << "Time: " << timestamp() << " | "
       << "Hostname: " << host_name() << " | "
       << "Parameters: " << params;
    cuda::profiling::mark::point(ss.str());

    for(int run_index = 0; run_index < params.num_query_execution_runs; run_index++) {
        cout << "Executing TPC-H Query 1, run " << run_index + 1 << " of " << params.num_query_execution_runs << "... " << flush;
        auto start = timer::now();
        execute_query_1_once(
            params, cuda_device, run_index, cardinality, streams,
            aggregates_on_host, aggregates_on_device, stream_input_buffer_sets,
            uncompressed, compressed);

        auto end = timer::now();

        std::chrono::duration<double> duration(end - start);
        cout << "done." << endl;
        results_file << duration.count() << '\n';
        if (params.should_print_results) {
            print_results(aggregates_on_host, cardinality);
        }
    }
    cuda::profiling::stop();
}
