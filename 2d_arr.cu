#include<iostream>
#include<cuda_runtime.h>

__global__ void kernel (const float arr[][4])
{
   int indexX = (blockIdx.x*blockDim.x + threadIdx.x);
   int indexY = (blockIdx.y*blockDim.y + threadIdx.y);
   int indexZ = (blockIdx.z*blockDim.z + threadIdx.z);
   printf("arr[%d][%d] = %f; IndexX = %d; IndexY = %d; IndexZ = %d \n Block.x = %d; Block.y = %d; Block.x = %d; Thread.x = %d; Thread.y = %d; Thread.z = %d; \n Blocks.x = %d; Blocks.y = %d; Blocks.z = %d; Threads.x = %d; Threads.y = %d; Threads.z = %d \n ---------------------------------------------- \n\n", indexX, indexY, arr[indexX][indexY], indexX, indexY, indexZ, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z );
} 

int main()
{
   float arr [6][4] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24};
   dim3 blocks (2, 2, 1);
   dim3 threads (3, 2, 1);
   float * d_arr; 

   cudaError_t error = cudaMalloc((void **)&d_arr, 24 * sizeof(int));   
   if(error != cudaSuccess)
   {
       std::cerr << "Failure of GPU memory allocation for d_arr" << std::endl;
       std::cerr << cudaGetErrorString(error) << std::endl;
       return 1;
   }   

   error=cudaMemcpy(d_arr, arr, 24 * sizeof(float), cudaMemcpyHostToDevice); 

   if(error != cudaSuccess)
   {
       std::cerr << "Failure of transforing memory from host to GPU for d_arr" << std::endl;
       std::cerr << cudaGetErrorString(error) << std::endl;
       return 1;
   }     

   kernel<<<blocks, threads>>>( (const float (*)[4]) d_arr);
   error = cudaGetLastError();
   if ( error != cudaSuccess )
   {
       std::cerr << cudaGetErrorString(error) << std::endl;
      return 1;      
   }   

   cudaDeviceSynchronize();
   cudaFree(d_arr);
   return 0;
}
