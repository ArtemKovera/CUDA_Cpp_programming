#include<cuda_runtime.h>
#include<iostream>


__global__ void checkDimension()
{
    printf("threadIdx: (%d, %d, %d) | blockIdx: (%d, %d, %d) | blockDim: (%d, %d, %d) | gridDim (%d, %d, %d)\n",
           threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, 
           blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
}


int main()
{
    dim3 block(2, 2, 2);
    dim3 grid(2);
    
    std::cout << "\nPrinting from host:" << std::endl;
    printf("grid.x=%d grid.y=%d grid.z=%d\n", grid.x, grid.y, grid.z);
    printf("block.x=%d block.y=%d block.z=%d\n\n", block.x, block.y, block.z);
    
    std::cout << "Printing from the device:" << std::endl;
    checkDimension<<<grid, block>>>();

    cudaDeviceReset();
    return 0;
}