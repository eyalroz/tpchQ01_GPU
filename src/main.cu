#include <iostream>
#include <cuda/api_wrappers.h>
#include <vector>
#include <iomanip>

#include "kernel.hpp"
#include "../expl_comp_strat/tpch_kit.hpp"

using timer = std::chrono::high_resolution_clock;

inline bool file_exists (const std::string& name) {
  struct stat buffer;
  return (stat (name.c_str(), &buffer) == 0);
}

int main(){
    if (!file_exists("lineitem.tbl")) {
        fprintf(stderr, "lineitem.tbl not found!\n");
        exit(1);
    }
    std::cout << "TPC-H Query 1" << '\n';
    get_device_properties();
    /* load data */
    lineitem li(7000000ull);
    li.FromFile("lineitem.tbl");
    kernel_prologue();
    const size_t data_length = cardinality;
    const int nStreams = static_cast<int>((data_length + TUPLES_PER_STREAM - 1) / TUPLES_PER_STREAM);
    clear_tables();

    /* Allocate memory on device */
    auto current_device = cuda::device::current::get();

    auto d_shipdate      = cuda::memory::device::make_unique< int[]            >(current_device, data_length);
    auto d_discount      = cuda::memory::device::make_unique< int64_t[]        >(current_device, data_length);
    auto d_extendedprice = cuda::memory::device::make_unique< int64_t[]        >(current_device, data_length);
    auto d_tax           = cuda::memory::device::make_unique< int64_t[]        >(current_device, data_length);
    auto d_quantity      = cuda::memory::device::make_unique< int64_t[]        >(current_device, data_length);
    auto d_returnflag    = cuda::memory::device::make_unique< char[]           >(current_device, data_length);
    auto d_linestatus    = cuda::memory::device::make_unique< char[]           >(current_device, data_length);
    auto d_aggregations  = cuda::memory::device::make_unique< AggrHashTable[]  >(current_device, MAX_GROUPS);

    /* Transfer data to device */
    cudaStream_t streams[nStreams];
    for (int i = 0; i < nStreams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    cuda_check_error();
    auto start = timer::now();
    for (int i = 0; i < nStreams; ++i) {
        size_t offset = i * TUPLES_PER_STREAM;
        size_t size = std::min((size_t) TUPLES_PER_STREAM, (size_t) (data_length - offset));

        //std::cout << "Stream " << i << ": " << "[" << offset << " - " << offset + size << "]" << std::endl;

        cuda::memory::async::copy(d_shipdate.get()      + offset, shipdate      + offset, size * sizeof(int),     streams[i]);
        cuda::memory::async::copy(d_discount.get()      + offset, discount      + offset, size * sizeof(int64_t),     streams[i]);
        cuda::memory::async::copy(d_extendedprice.get() + offset, extendedprice + offset, size * sizeof(int64_t),     streams[i]);
        cuda::memory::async::copy(d_tax.get()           + offset, tax           + offset, size * sizeof(int64_t),     streams[i]);
        cuda::memory::async::copy(d_quantity.get()      + offset, quantity      + offset, size * sizeof(int64_t),     streams[i]);
        cuda::memory::async::copy(d_returnflag.get()    + offset, returnflag    + offset, size * sizeof(char), streams[i]);
        cuda::memory::async::copy(d_linestatus.get()    + offset, linestatus    + offset, size * sizeof(char), streams[i]);
    }

    for (int i = 0; i < nStreams; ++i) {
        size_t offset = i * TUPLES_PER_STREAM;
        size_t size = std::min((size_t) TUPLES_PER_STREAM, (size_t) (data_length - offset));;
        size_t amount_of_blocks = TUPLES_PER_STREAM / (VALUES_PER_THREAD * THREADS_PER_BLOCK);

        //std::cout << "Execution <<<" << amount_of_blocks << "," << THREADS_PER_BLOCK << ">>>" << std::endl;

        cuda::thread_local_tpchQ01<<<amount_of_blocks, THREADS_PER_BLOCK, 0, streams[i]>>>(
            d_shipdate.get() + offset,
            d_discount.get() + offset,
            d_extendedprice.get() + offset,
            d_tax.get() + offset,
            d_returnflag.get() + offset,
            d_linestatus.get() + offset,
            d_quantity.get() + offset,
            d_aggregations.get() + offset,
            (u64_t) size);
    }
    cudaDeviceSynchronize();
    auto end = timer::now();
    cuda_check_error();
    for (int i = 0; i < nStreams; ++i) {
        //size_t offset = i * TUPLES_PER_STREAM;
        //cudaMemcpyAsync(&a[offset], &d_a[offset], 
                          //streamBytes, cudaMemcpyDeviceToHost, streams[i]);
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    auto print_dec = [] (auto s, auto x) { printf("%s%ld.%ld", s, Decimal64::GetInt(x), Decimal64::GetFrac(x)); };
    for (size_t group=0; group<MAX_GROUPS; group++) {
        if (aggrs0[group].count > 0) {
            char rf = group >> 8;
            char ls = group & std::numeric_limits<unsigned char>::max();

            size_t i = group;

            printf("# %c|%c", rf, ls);
            print_dec(" | ", aggrs0[i].sum_quantity);
            print_dec(" | ", aggrs0[i].sum_base_price);
            print_dec(" | ", aggrs0[i].sum_disc_price);
            print_dec(" | ", aggrs0[i].sum_charge);
            printf("|%ld\n", aggrs0[i].count);
        }
    }

    double sf = li.l_returnflag.cardinality / 6001215;

    std::chrono::duration<double> duration(end - start);
    uint64_t tuples_per_second = static_cast<uint64_t>(data_length / duration.count());
    auto size_per_tuple = sizeof(int) + sizeof(int64_t) * 4 + sizeof(char) * 2;
    double effective_memory_throughput = tuples_per_second * size_per_tuple / 1024.0 / 1024.0 / 1024.0;
    double effective_memory_throughput_read = tuples_per_second * sizeof(int) / 1024.0 / 1024.0 / 1024.0;
    double effective_memory_throughput_write_output = tuples_per_second / 8.0 / 1024.0 / 1024.0 / 1024.0;
    std::cout << "\n+-------------------------------- Statistics -----------------------------------+\n";
    std::cout << "| TPC-H Q01 performance               : " << std::fixed << tuples_per_second << " [tuples/sec]" << std::endl;
    uint64_t cache_line_size = 128; // bytes

    

    std::cout << "| Time taken                          : ~" << std::setprecision(2)
              << duration.count() << " [s]" << std::endl;
    std::cout << "| Estimated time for TPC-H SF100      : ~" << std::setprecision(2)
              << duration.count() * (100 / sf) << " [s]" << std::endl;
    std::cout << "| Effective memory throughput (query) : ~" << std::setprecision(2)
              << effective_memory_throughput << " [GB/s]" << std::endl;
    std::cout << "| Estimated memory throughput (query) : ~" << std::setprecision(2)
              << (tuples_per_second * cache_line_size / 1024.0 / 1024.0 / 1024.0) << " [GB/s]" << std::endl;
    std::cout << "| Effective memory throughput (read)  : ~" << std::setprecision(2)
              << effective_memory_throughput_read << " [GB/s]" << std::endl;
    std::cout << "| Memory throughput (write)           : ~" << std::setprecision(2)
              << effective_memory_throughput_write_output << " [GB/s]" << std::endl;
    std::cout << "| Theoretical Bandwidth [GB/s]        : " << (5505 * 10e06 * (352 / 8) * 2) / 10e09 << std::endl;
    std::cout << "| Effective Bandwidth [GB/s]          : " << data_length * 25 * 13 / duration.count() << std::endl;
    std::cout << "+-------------------------------------------------------------------------------+\n";
    /* Test */
    /*std::cout << std::endl;
    std::vector<int> in(64, 1);
    std::vector<int> res(64, 0);  
    int  *d_in, *d_out, *d_res;
    cudaMalloc(&d_in,       64 * sizeof(int));
    cudaMalloc(&d_out,      64 * sizeof(int));
    cudaMalloc(&d_res,      64 * sizeof(int));
    cudaMemcpy(d_in,      &in[0],      64 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_res,      &res[0],      64 * sizeof(int), cudaMemcpyHostToDevice);
    cuda::filter_k<<<2,32>>>(d_out,d_res, d_in, 64);
    cudaMemcpy(     &res[0], d_res,      64 * sizeof(int), cudaMemcpyDeviceToHost);
    for(auto &i : res)
            printf(" %d ", i);
    //cuda::deviceReduceKernel<<<1,32>>>(d_out, d_out, 2);
    cudaFree(d_in);
    cudaFree(d_out);*/

}