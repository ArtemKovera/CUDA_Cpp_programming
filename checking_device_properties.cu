#include<iostream>
#include<cuda_runtime.h>


int main()
{
    int dev = 0;
    cudaDeviceProp deviceProp;

    cudaError_t error = cudaGetDeviceProperties(&deviceProp, dev);
    if(error != cudaSuccess)
    {
        fprintf(stderr, "Failure in calling cudaGetDeviceProperties()\n");
        cudaGetErrorString(error);
        return 1;
    }
    
    printf("GPU Device                                                      %s\n", deviceProp.name);
    printf("Total amount of global memory                                   %lu\n", deviceProp.totalGlobalMem);
    printf("Maximum amount of shared memory available to a thread block     %lu\n", deviceProp.sharedMemPerBlock);
    printf("Maximum number of 32-bit registers available to a thread block  %d\n", deviceProp.regsPerBlock);  
    printf("Warp size in threads                                            %d\n", deviceProp.warpSize);  
    printf("Maximum number of threads per block                             %d\n", deviceProp.maxThreadsPerBlock);
    printf("Maximum size of X dimension of a block                          %d\n", deviceProp.maxThreadsDim[0]);
    printf("Maximum size of Y dimension of a block                          %d\n", deviceProp.maxThreadsDim[1]);
    printf("Maximum size of Z dimension of a block                          %d\n", deviceProp.maxThreadsDim[2]);
    printf("Maximum size of X dimension of a grid                           %d\n", deviceProp.maxGridSize[0]);
    printf("Maximum size of Y dimension of a grid                           %d\n", deviceProp.maxGridSize[1]);
    printf("Maximum size of Z dimension of a grid                           %d\n", deviceProp.maxGridSize[2]);
    printf("Clock frequency in kilohertz                                    %d\n", deviceProp.clockRate);
    printf("Single precision to double precision performance (in flops)     %d\n", deviceProp.singleToDoublePrecisionPerfRatio);

    return 0;
}