#include<iostream>
#include<stdlib.h>
#include<cuda_runtime.h>

#define SIZE 1048576
#define NUMBER 8           
#define THREADS_PER_BLOCK 512
#define BLOCKS (SIZE/THREADS_PER_BLOCK)


__global__ void count (const int * arr, int * arr2, const int val)
{

    __shared__ int sh;
    sh = 0;
    __syncthreads();

    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(arr[index] == val)
    {
        atomicAdd(&sh, 1);
    }
        
    
    __syncthreads();
    
    //only one thread per block writes to arr2
    if (threadIdx.x==0)
    {
        arr2[blockIdx.x] = sh;
    }
       
}


int main()
{

    int * arr = new(std::nothrow) int [SIZE];

    size_t total = 0;

    if(arr)
    {
        for(int i = 0; i < SIZE; ++i)
            arr[i] = rand()%10;
    }
    else
    {
        std::cerr << "Failure to allocate memmory for arr" << std::endl; 
        return 1;
    }

    for(int i = 0; i < SIZE; ++i)
    {
        if(arr[i] == NUMBER)
            ++total;
    }
    
    std::cout << "Total number computed on host = " << total << std::endl;
    

    int * ptr = new(std::nothrow) int [BLOCKS] {0};

    if(ptr == nullptr)
    {
        std::cerr << "Failure to allocate memory for ptr" << std::endl;
        return 1;        
    }
       
    int * d_arr1  = nullptr;
    int * d_arr2  = nullptr;

    cudaError_t error = cudaMalloc((void**)&d_arr1, SIZE*sizeof(int));
    if(error != cudaSuccess)
    {
        std::cerr << "Failure of GPU memory allocation for d_arr1" << std::endl;
        cudaGetErrorString(error);
        return 1;
    }
    error = cudaMalloc((void**)&d_arr2, BLOCKS*sizeof(int));
    if(error != cudaSuccess)
    {
        std::cerr << "Failure of GPU memory allocation for d_arr2" << std::endl;
        cudaGetErrorString(error);
        return 1;
    }   
    

    error = cudaMemcpy(d_arr1, arr, SIZE*sizeof(int), cudaMemcpyHostToDevice);
    if(error != cudaSuccess)
    {
        std::cerr << "Failure of transforing memory from host to GPU for d_arr1" << std::endl;
        cudaGetErrorString(error);
        return 1;
    } 
    error = cudaMemcpy(d_arr2, ptr, BLOCKS*sizeof(int), cudaMemcpyHostToDevice);
    if(error != cudaSuccess)
    {
        std::cerr << "Failure of transforing memory from host to GPU for d_arr2" << std::endl;
        cudaGetErrorString(error);
        return 1;
    }       

    count<<<BLOCKS, THREADS_PER_BLOCK>>>(d_arr1, d_arr2, NUMBER);
    error = cudaGetLastError();
    if ( error != cudaSuccess )
    {
       printf("CUDA Error: %s\n", cudaGetErrorString(error)); 
       return 1;      
    }   
    
    error = cudaMemcpy(ptr, d_arr2, BLOCKS*sizeof(int), cudaMemcpyDeviceToHost);
    if(error != cudaSuccess)
    {
        std::cerr << "Failure of transforing memory from GPU to host for d_arr2" << std::endl;
        cudaGetErrorString(error);
        return 1;
    } 
    
    total = 0;
    for(int i = 0; i < BLOCKS; ++i)
    {
        total += ptr[i];
    }

    std::cout << "Total number computed on GPU = " << total << std::endl;

    cudaFree(d_arr1);
    cudaFree(d_arr2);
    delete [] arr;
    delete [] ptr;
    arr = nullptr;
    ptr = nullptr;
    d_arr1 = nullptr;
    d_arr2 = nullptr;

    return 0;
}