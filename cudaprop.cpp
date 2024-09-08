#include<iostream>
#include<cuda_runtime.h>

#define DEVICE 0


int main()
{

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, DEVICE);
    
    std::cout << "Device name: " << prop.name << std::endl;
    std::cout << "Number of asynchronous engines: " << prop.asyncEngineCount << std::endl;
    std::cout << "Memory Clock Rate (MHz): " << prop.memoryClockRate/1024 << std::endl;
    std::cout << "Number of multiprocessors on device: " << prop.multiProcessorCount << std::endl;
    std::cout << "Shared memory per block (Kb): " << prop.sharedMemPerBlock/1024 << std::endl;
    std::cout << "Shared memory per multiprocessor (Kb): " << prop.sharedMemPerMultiprocessor/1024 << std::endl;
    std::cout << "Maxim number of resident blocks per multiprocessor: " << prop.maxBlocksPerMultiProcessor << std::endl;
    std::cout << "Global memory available on device (Mb): " << prop.totalGlobalMem/(1024*1024) << std::endl;
    std::cout << "Device is a Tesla device using TCC driver: " << (prop.tccDriver ? "yes" : "no") << std::endl;
    std::cout << "Constant memory available on device (Kb): " << prop.totalConstMem << std::endl;
    std::cout << "Ratio of single precision performance to double precision performance: " << prop.singleToDoublePrecisionPerfRatio << std::endl;
    std::cout << "Size of L2 cache (Kb): " << prop.l2CacheSize/1024 << std::endl;
    std::cout << "Clock rate (MHz): " << prop.clockRate/1024 << std::endl;
    std::cout << "Warp size: " << prop.warpSize << std::endl;
    std::cout << "32-bit registers available per multiprocessor: " << prop.regsPerMultiprocessor << std::endl;
    std::cout << "32-bit registers available per block: " << prop.regsPerBlock << std::endl;
    std::cout << "Major compute capability: " << prop.major << std::endl;
    std::cout << "Maximum number of threads per block: " << prop.maxThreadsPerBlock  << std::endl;

    return 0;
}