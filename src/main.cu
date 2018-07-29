#include "data_types.h"
#include "util/helper.hpp"
#include "util/extra_pointer_traits.hpp"
#include "constants.hpp"
#include "util/bit_operations.hpp"
#include "kernels/ht_in_global_mem.hpp"
#include "kernels/ht_in_registers.cuh"
#include "kernels/ht_in_registers_per_thread.cuh"
#include "kernels/ht_in_local_mem.cuh"
#include "kernels/ht_in_shared_mem_per_thread.cuh"
#include "kernels/ht_in_shared_mem_per_block.cuh"
#include "expl_comp_strat/tpch_kit.hpp"
#include "expl_comp_strat/common.hpp"
#include "cpu/common.hpp"
#include "cpu.h"
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

static CoProc* cpu_coprocessor = nullptr;

using timer = std::chrono::high_resolution_clock;

template <template <typename> class Ptr, bool Compressed>
struct input_buffer_set;

enum : bool { is_compressed = true, is_not_compressed = false};

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

const std::unordered_map<string, cuda::device_function_t> plain_kernels = {
    { "local_mem",               kernels::local_mem::one_table_per_thread::tpch_query_01         },
    { "in_registers",            kernels::in_registers::several_tables_per_warp::tpch_query_01   },
    { "in_registers_per_thread", kernels::in_registers::one_table_per_thread::tpch_query_01      },
    { "shared_mem_per_thread",   kernels::shared_mem::one_table_per_thread::tpch_query_01<>      },
    { "shared_mem_per_block",    kernels::shared_mem::one_table_per_block::tpch_query_01         },
    { "global",                  kernels::global_mem::single_table::tpch_query_01                },
};

const std::unordered_map<string, cuda::device_function_t> kernels_compressed = {
    { "local_mem",               kernels::local_mem::one_table_per_thread::tpch_query_01_compressed       },
    { "in_registers",            kernels::in_registers::several_tables_per_warp::tpch_query_01_compressed },
    { "in_registers_per_thread", kernels::in_registers::one_table_per_thread::tpch_query_01_compressed    },
    { "shared_mem_per_thread",   kernels::shared_mem::one_table_per_thread::tpch_query_01_compressed<>    },
    { "shared_mem_per_block",    kernels::shared_mem::one_table_per_block::tpch_query_01_compressed       },
    { "global",                  kernels::global_mem::single_table::tpch_query_01_compressed              },
};

const std::unordered_map<string, cuda::device_function_t> kernels_filter_pushdown = {
    { "local_mem",               kernels::local_mem::one_table_per_thread::tpch_query_01_compressed_precomputed_filter       },
    { "in_registers",            kernels::in_registers::several_tables_per_warp::tpch_query_01_compressed_precomputed_filter },
    { "in_registers_per_thread", kernels::in_registers::one_table_per_thread::tpch_query_01_compressed_precomputed_filter    },
    { "global",                  kernels::global_mem::single_table::tpch_query_01_compressed_precomputed_filter              },
    { "shared_mem_per_block",    kernels::shared_mem::one_table_per_block::tpch_query_01_compressed_precomputed_filter       },
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

const std::unordered_map<string, unsigned> num_threads_to_handle_tuple = {
    { "local_mem",               1  },
    { "in_registers",            div_rounding_up(warp_size, warp_size / num_potential_groups)  },
    { "in_registers_per_thread", 1  },
    { "shared_mem_per_thread",   1  },
//  { "shared_mem",              1? },
    { "global",                  1  },
};

void print_help(int argc, char** argv) {
    fprintf(stderr, "Unrecognized command line option.\n");
    fprintf(stderr, "Usage: %s [args]\n", argv[0]);
    fprintf(stderr, "   --device=[default:%u] (number, e.g. 1)\n", cuda::device::default_device_id);
    fprintf(stderr, "   --apply-compression\n");
    fprintf(stderr, "   --print-results\n");
    fprintf(stderr, "   --use-filter-pushdown\n");
    fprintf(stderr, "   --use-coprocessing\n");
    fprintf(stderr, "   --hash-table-placement=[default:in_registers_per_thread]\n"
                    "     (one of: ");
    int remaining = plain_kernels.size();
    for (auto p : plain_kernels) {
        fprintf(stderr, "%s%s", p.first.c_str(), (--remaining == 0 ? ")\n" : ", "));
    }
    fprintf(stderr, "   --runs=[default:%u] (number, e.g. 1 - 100)\n", (unsigned) defaults::num_query_execution_runs);
    fprintf(stderr, "   --sf=[default:%f] (number, e.g. 0.01 - 100)\n", defaults::scale_factor);
    fprintf(stderr, "   --cpu-fraction=[default:%f] (number, e.g. 0.5 - 100)\n", defaults::cpu_coprocessing_fraction);
    fprintf(stderr, "   --streams=[default:%u] (number, e.g. 1 - 64)\n", defaults::num_gpu_streams);
    fprintf(stderr, "   --threads-per-block=[default:%u] (number, e.g. 32 - 1024)\n", defaults::num_threads_per_block);
    fprintf(stderr, "   --tuples_per_thread=[default:%u] (number, e.g. 1 - 1048576)\n", defaults::num_tuples_per_thread);
    fprintf(stderr, "   --tuples-per-kernel-launch=[default:%u] (number, e.g. 64 - 4194304)\n", defaults::num_tuples_per_kernel_launch);
}


struct q1_params_t {

    // Command-line-settable parameters

    cuda::device::id_t cuda_device_id    { cuda::device::default_device_id };
    double scale_factor                  { defaults::scale_factor };
    std::string kernel_variant           { defaults::kernel_variant };
    bool should_print_results            { defaults::should_print_results };
    bool use_filter_pushdown             { false };
    bool apply_compression               { defaults::apply_compression };
    int num_gpu_streams                  { defaults::num_gpu_streams };
    cuda::grid_block_dimension_t num_threads_per_block
                                         { defaults::num_threads_per_block };
    int num_tuples_per_thread            { defaults::num_tuples_per_thread };

    int num_tuples_per_kernel_launch     { defaults::num_tuples_per_kernel_launch };
        // Make sure it's a multiple of num_threads_per_block and of warp_size, or bad things may happen

    // This is the number of times we run the actual query execution - the part that we time;
    // it will not include initialization/allocations that are not necessary when the DBMS
    // is brought up. Note the allocation vs sub-allocation issue (see further comments below)
    int num_query_execution_runs         { defaults::num_query_execution_runs };

    bool use_coprocessing                { false };

    // The fraction of the total table (i.e. total number of tuples) which the CPU, rather than
    // the GPU, will undertake to process; this ignores any filter precomputation work.
    double cpu_processing_fraction       { defaults::cpu_coprocessing_fraction };
    bool user_set_num_threads_per_block  { false };
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

q1_params_t parse_command_line(int argc, char** argv)
{
    q1_params_t params;

    for(int i = 1; i < argc; i++) {
        auto arg = string(argv[i]);
        if (arg.substr(0,2) != "--") {
            print_help(argc, argv);
            exit(EXIT_FAILURE);
        }
        arg = arg.substr(2);
        if (arg == "list-devices") {
            get_device_properties();
            exit(1);
        } else if (arg == "use-coprocessing") {
            params.use_coprocessing = true;
        } else if (arg == "apply-compression") {
            params.apply_compression = true;
        } else if (arg == "use-filter-pushdown") {
            params.use_filter_pushdown = true;
        }  else if (arg == "print-results") {
            params.should_print_results = true;
        } else {
            // A  name=value argument
            auto p = split_once(arg, '=');
            auto& arg_name = p.first; auto& arg_value = p.second;
            if (arg_name == "scale-factor") {
                params.scale_factor = std::stod(arg_value);
                if (params.scale_factor - 0 < 0.001) {
                    cerr << "Invalid scale factor " + std::to_string(params.scale_factor) << endl;
                    exit(EXIT_FAILURE);
                }
            } else if (arg_name == "hash-table-placement") {
                params.kernel_variant = arg_value;
                if (plain_kernels.find(params.kernel_variant) == plain_kernels.end()) {
                    cerr << "No kernel variant named \"" + params.kernel_variant + "\" is available" << endl;
                    exit(EXIT_FAILURE);
                }
            } else if (arg_name == "device") {
                params.cuda_device_id = std::stod(arg_value);
                if ((params.cuda_device_id < 0) or (cuda::device::count() <= params.cuda_device_id)) {
                    cerr << "Invalid device number " << params.cuda_device_id << endl;
                    exit(EXIT_FAILURE);
                }
            } else if (arg_name == "streams") {
                params.num_gpu_streams = std::stoi(arg_value);
            } else if (arg_name == "tuples-per-thread") {
                params.num_tuples_per_thread = std::stoi(arg_value);
            } else if (arg_name == "threads-per-block") {
                params.num_threads_per_block = std::stoi(arg_value);
                params.user_set_num_threads_per_block = true;
                if (params.num_threads_per_block % cuda::warp_size != 0) {
                    cerr << "All kernels only support numbers of threads per grid block "
                         << "which are multiples of the warp size (" << cuda::warp_size << ")" << endl;
                    exit(EXIT_FAILURE);
                }
            } else if (arg_name == "tuples-per-kernel-launch") {
                params.num_tuples_per_kernel_launch = std::stoi(arg_value);
            } else if (arg_name == "runs") {
                try {
                    params.num_query_execution_runs = std::stoi(arg_value);
                    if (params.num_query_execution_runs <= 0) {
                        cerr << "Number of runs must be positive" << endl;
                        exit(EXIT_FAILURE);
                    }
                } catch(std::invalid_argument) {
                    cerr << "Cannot parse the number of runs passed after --runs=" << endl;
                    exit(EXIT_FAILURE);
                }
            } else if (arg_name == "cpu-fraction") {
                try {
                    params.cpu_processing_fraction = std::stod(arg_value);
                    if (params.cpu_processing_fraction < 0 or params.cpu_processing_fraction > 1.0) {
                        cerr << "The fraction of aggregation work be performed by the CPU must be in the range 0.0 - 1.0" << endl;
                        exit(EXIT_FAILURE);
                    }
                } catch(std::invalid_argument) {
                    cerr << "Cannot parse the fraction passed after --cpu-fraction=" << endl;
                    exit(EXIT_FAILURE);
                }
            } else {
                print_help(argc, argv);
                exit(EXIT_FAILURE);
            }
        }
    }
    if (params.use_filter_pushdown and not params.apply_compression) {
        cerr << "Filter precomputation is only currently supported when compression is applied; "
                "invoke with \"--apply-compression\"." << endl;
        exit(EXIT_FAILURE);
    }
    if (fixed_threads_per_block.find(params.kernel_variant) != fixed_threads_per_block.end()) {
        auto required_num_thread_per_block = fixed_threads_per_block.at(params.kernel_variant);
        if (params.user_set_num_threads_per_block and params.num_threads_per_block != required_num_thread_per_block) {
            throw std::invalid_argument("Invalid number of threads per block for kernel variant "
                + params.kernel_variant + " (it must be "
                + std::to_string(required_num_thread_per_block) + ")");
        }
        params.num_threads_per_block = fixed_threads_per_block.at(params.kernel_variant);
    }
    else if (max_threads_per_block.find(params.kernel_variant) != max_threads_per_block.end()) {
        auto max_threads_per_block_for_kernel_variant = max_threads_per_block.at(params.kernel_variant);
        if (params.user_set_num_threads_per_block and
            (max_threads_per_block_for_kernel_variant < params.num_threads_per_block)) {
            throw std::invalid_argument("Number of threads per block set for kernel variant "
                + params.kernel_variant + " exceeds the maximum possible value of "
                + std::to_string(max_threads_per_block_for_kernel_variant));
        }
        params.num_threads_per_block = max_threads_per_block_for_kernel_variant;
    }
    return params;
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
    input_buffer_set<UniquePtr, Compressed>&   buffer_set,
    lineitem&                                  li)
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
//             << "\tnum_tuples_for_this_launch  = " << num_tuples_for_this_launch << '\n'
//             << "\tgrid blocks                 = " << num_blocks << '\n'
//             << "\tthreads per block           = " << num_threads_per_block << '\n';

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
        cuda::memory::host::make_unique< bit_container_t[] >(div_rounding_up(cardinality, bits_per_container))
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

//template <template <typename> class Ptr>
//void test01(
//    input_buffer_set<Ptr, is_compressed>&   buffer_set)
//{
//    for_each_argument(
//        [&](auto tup){
//            auto& ptr           = std::get<0>(tup);
//            const auto& name    = std::get<1>(tup);
//            using traits = typename extra_pointer_traits<decltype(ptr)>;
//            ptr = traits::make(count);
//            load_column_from_binary_file(
//                ptr.get(), 100, "", name + ".bin");
//        },
//        tie(buffer_set.ship_date,      "shipdate"        ),
//        tie(buffer_set.discount,       "discount"        ),
//        tie(buffer_set.tax,            "tax"             ),
//        tie(buffer_set.quantity,       "quantity"        ),
//        tie(buffer_set.extended_price, "extendedprice"   ),
//        tie(buffer_set.return_flag,    "returnflag"      ),
//        tie(buffer_set.line_status,    "linestatus"      )
//    );
//}


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
            cardinality = load_cached_columns(params, compressed, li);
        }
        else {
            cardinality = load_cached_columns(params, uncompressed_outside_li, li);
            move_buffers_into_lineitem_object(uncompressed_outside_li, li, cardinality);
            uncompressed = get_buffers_inside(li);
        }
    }
    else {
        if (columns_are_cached(params, is_not_compressed)) {
            cardinality = load_cached_columns(params, uncompressed_outside_li, li);
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
