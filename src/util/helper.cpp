#include "helper.hpp"

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

void list_device_properties(){
    int32_t device_cnt = 0;
    cudaGetDeviceCount(&device_cnt);
    cudaDeviceProp device_prop;

    for (int i = 0; i < device_cnt; i++) {
        cudaGetDeviceProperties(&device_prop, i);
        std::cout << "+---------------------------------------------------------------------------------------------------------------+\n";
        printf("| Device id: %d\t", i);
        printf("  Device name: %s\t",                   device_prop.name);
        printf("  Compute capability: %d.%d\n",         device_prop.major, device_prop.minor);
        std::cout << std::endl;
        printf("| Memory Clock Rate [KHz]: %d\n",       device_prop.memoryClockRate);
        printf("| Memory Bus Width [bits]: %d\n",       device_prop.memoryBusWidth);
        printf("| Peak Memory Bandwidth [GB/s]: %f\n",  2.0*device_prop.memoryClockRate*(device_prop.memoryBusWidth/8)/1.0e6);
        printf("| L2 size [KB]: %d\n",                  device_prop.l2CacheSize/1024);
        printf("| Shared Memory per Block [KB]: %lu\n", device_prop.sharedMemPerBlock/1024);
        std::cout << std::endl;
        printf("| Number of SMs: %d\n",                 device_prop.multiProcessorCount);
        printf("| Max. number of threads per SM: %d\n", device_prop.maxThreadsPerMultiProcessor);
        printf("| Concurrent kernels: %d\n",            device_prop.concurrentKernels);
        printf("| WarpSize: %d\n",                      device_prop.warpSize);
        printf("| MaxThreadsPerBlock: %d\n",            device_prop.maxThreadsPerBlock);
        printf("| MaxThreadsDim[0]: %d\n",              device_prop.maxThreadsDim[0]);
        printf("| MaxGridSize[0]: %d\n",                device_prop.maxGridSize[0]);
        printf("| PageableMemoryAccess: %d\n",          device_prop.pageableMemoryAccess);
        printf("| ConcurrentManagedAccess: %d\n",       device_prop.concurrentManagedAccess);
        printf("| Number of async. engines: %d\n",      device_prop.asyncEngineCount);
        std::cout << "+---------------------------------------------------------------------------------------------------------------+\n";
    }
}

void make_sure_we_are_on_cpu_core_0()
{
#if 0
    // CPU affinities are devil's work
    // Make sure we are on core 0
    // TODO: Why not in a function?
    cpu_set_t cpuset;

    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    sched_setaffinity(0, sizeof(cpuset), &cpuset);
#endif
}

std::pair<std::string,std::string> split_once(std::string delimited, char delimiter) {
    auto pos = delimited.find_first_of(delimiter);
    return { delimited.substr(0, pos), delimited.substr(pos+1) };
}


std::string host_name()
{
    enum { max_len = 1023 };
    char buffer[max_len + 1];
    gethostname(buffer, max_len);
    buffer[max_len] = '\0'; // to be on the safe side
    return { buffer };
}

std::string timestamp()
{
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_c), "%F %T");
    return ss.str();
}
