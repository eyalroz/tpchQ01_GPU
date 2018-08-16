#include "parse_cmdline.hpp"

#include "data_types.hpp"
#include "util/helper.hpp"
#include "constants.hpp"
#include "cpu/common.hpp"

#include <boost/program_options.hpp>

#include <iostream>
#include <cuda/api_wrappers.h>
#include <iomanip>
#include <sstream>
#include <unordered_map>

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

template <typename V>
V& update_with(V& updatee, const std::string& key, const boost::program_options::variables_map& vm)
{
    auto it = vm.find(key);
    if (it != vm.end()) {
        updatee = vm.at(key).template as<V>();
    }
    return updatee;
}

q1_params_t make_params(const boost::program_options::variables_map& vm)
{
    q1_params_t params;

    // TODO: Use verifiers instead of verifying manually

    params.use_coprocessing     = (vm.find("use-coprocessing"   ) != vm.end());
    params.apply_compression    = (vm.find("apply-compression"  ) != vm.end());
    params.use_filter_pushdown  = (vm.find("use-filter-pushdown") != vm.end());
    params.should_print_results = (vm.find("print-results"      ) != vm.end());

    update_with(params.scale_factor, "scale-factor", vm);
    if (params.scale_factor - 0 < 0.001) {
        cerr << "Invalid scale factor " + std::to_string(params.scale_factor) << endl;
        exit(EXIT_FAILURE);
    }
    update_with(params.kernel_variant, "hash-table-placement", vm);
    if (plain_kernels.find(params.kernel_variant) == plain_kernels.end()) {
        cerr << "No kernel variant named \"" + params.kernel_variant + "\" is available" << endl;
        exit(EXIT_FAILURE);
    }
    update_with(params.cuda_device_id, "device", vm);
    if ((params.cuda_device_id < 0) or (cuda::device::count() <= params.cuda_device_id)) {
        cerr << "Invalid device id " << params.cuda_device_id << endl;
        exit(EXIT_FAILURE);
    }
    update_with(params.num_gpu_streams, "streams", vm);
    update_with(params.num_tuples_per_thread, "tuples-per-thread", vm);
    update_with(params.num_threads_per_block, "threads-per-block", vm);
    update_with(params.num_gpu_streams, "streams", vm);
    if (params.num_threads_per_block % cuda::warp_size != 0) {
        cerr << "All kernels only support numbers of threads per grid block "
             << "which are multiples of the warp size (" << cuda::warp_size << ")" << endl;
        exit(EXIT_FAILURE);
    }
    update_with(params.num_tuples_per_kernel_launch, "tuples-per-kernel-launch", vm);
    update_with(params.num_query_execution_runs, "runs", vm);
    if (params.num_query_execution_runs <= 0) {
        cerr << "Number of runs must be positive" << endl;
        exit(EXIT_FAILURE);
    }
    update_with(params.cpu_processing_fraction, "cpu-fraction", vm);
    if (params.cpu_processing_fraction < 0 or params.cpu_processing_fraction > 1.0) {
        cerr << "The fraction of aggregation work be performed by the CPU must be in the range 0.0 - 1.0" << endl;
        exit(EXIT_FAILURE);
    }
    if (params.use_filter_pushdown and not params.apply_compression) {
        cerr << "Filter precomputation is only currently supported when compression is applied; "
                "invoke with \"--apply-compression\"." << endl;
        exit(EXIT_FAILURE);
    }
    auto user_set_num_threads_per_block = (vm.find("threads-per-block") != vm.end());
    if (fixed_threads_per_block.find(params.kernel_variant) != fixed_threads_per_block.end()) {
        auto required_num_thread_per_block = fixed_threads_per_block.at(params.kernel_variant);
        if (user_set_num_threads_per_block and params.num_threads_per_block != required_num_thread_per_block)
        {
            throw std::invalid_argument("Invalid number of threads per block for kernel variant "
                + params.kernel_variant + " (it must be "
                + std::to_string(required_num_thread_per_block) + ")");
        }
        params.num_threads_per_block = fixed_threads_per_block.at(params.kernel_variant);
    }
    else if (max_threads_per_block.find(params.kernel_variant) != max_threads_per_block.end()) {
        auto max_threads_per_block_for_kernel_variant = max_threads_per_block.at(params.kernel_variant);
        if (user_set_num_threads_per_block and (max_threads_per_block_for_kernel_variant < params.num_threads_per_block)) {
            throw std::invalid_argument("Number of threads per block set for kernel variant "
                + params.kernel_variant + " exceeds the maximum possible value of "
                + std::to_string(max_threads_per_block_for_kernel_variant));
        }
        params.num_threads_per_block = max_threads_per_block_for_kernel_variant;
    }
    return params;
}

void print_help(int argc, char** argv, const boost::program_options::options_description& desc)
{
    std::cout
        << argv[0] << ": TPC-H Q1 GPU (and CPU-GPU) implementation benchmark utility\n"
        << desc << '\n';
}

q1_params_t parse_command_line(int argc, char** argv)
{
    namespace po = boost::program_options;
    using std::string;

    std::stringstream ss;
    ss << "Implementation variant; one of: (";
    int remaining = plain_kernels.size();
    for (auto p : plain_kernels) {
        ss << p.first << (--remaining == 0 ? ")\n" : ", ");
    }
    std::string kernel_variant_names_argument { ss.str() };


    po::options_description cmdline_options("command-line options");
    cmdline_options.add_options()
        ("help",                                                                                                        "Print usage information")
        ("print-results",                                                                                               "Print a table of the query (or aggregation) results")
        ("scale-factor",             po::value<double       >()->default_value(defaults::scale_factor),                 "TPC-H data set size to use")
        ("device",                   po::value<int          >()->default_value(cuda::device::default_device_id),        "Index of GPU device to use")
        ("streams",                  po::value<int          >()->default_value(defaults::num_gpu_streams),              "Use this many CUDA streams (=queues) for scheduling GPU work")
        ("runs",                     po::value<int          >()->default_value(defaults::num_query_execution_runs),     "Number of times to execute the query")
        ("list-devices",                                                                                                "List CUDA devices on this system")
        ("use-coprocessing",                                                                                            "Use the both a CPU socket and a GPU to process Q1")
        ("apply-compression",                                                                                           "Use compressed input columns")
        ("use-filter-pushdown",                                                                                         "Precompute the Q1 WHERE clause on the CPU")
        ("cpu-fraction",             po::value<double       >()->default_value(defaults::cpu_coprocessing_fraction),    "Fraction of data to be processed by the CPU, when co-processing")
        ("hash-table-placement",     po::value<string       >()->default_value(defaults::kernel_variant),               kernel_variant_names_argument.c_str())
        ("tuples-per-thread",        po::value<cardinality_t>()->default_value(defaults::num_tuples_per_thread),        "Process this many LINEITEM tuples with each GPU kernel thread")
        ("threads-per-block",        po::value<cuda::grid_block_dimension_t
                                                            >()->default_value(defaults::num_threads_per_block),        "Use this many threads per GPU kernel grid block")
        ("tuples-per-kernel-launch", po::value<cardinality_t>()->default_value(defaults::num_tuples_per_kernel_launch), "Launch a kernel for the data of this many LINEITEM table tuples")
    ;

    po::variables_map variables_map;

    try {
        auto parse_result =
            po::basic_command_line_parser<char>(argc, argv)
                .options(cmdline_options)
                .run();
        po::store(parse_result, variables_map);
        po::notify(variables_map); // Remember this throws on error
    }
    catch (boost::program_options::required_option& e) {
        std::cerr << "Invocation error.\n";
        print_help(argc, argv, cmdline_options);
        exit(EXIT_FAILURE);
    }
    catch (boost::program_options::error& e) {
        std::cerr << "Invocation error.\n";
        print_help(argc, argv, cmdline_options);
        exit(EXIT_FAILURE);
    }

    if (variables_map.find("help") != variables_map.end()) {
        print_help(argc, argv, cmdline_options);
        exit(EXIT_SUCCESS);
    }
    if (variables_map.find("list-devices") != variables_map.end()) {
        list_device_properties();
        exit(EXIT_SUCCESS);
    }

    return make_params(variables_map);
}
