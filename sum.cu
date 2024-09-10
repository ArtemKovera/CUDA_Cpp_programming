#include<iostream>
#include<cuda_runtime.h>

     
#define THREADS 1024
#define SIZE (THREADS*THREADS)

__global__ void computeSum (unsigned int * d_arr, unsigned int * d_total)
{
    __shared__ unsigned int sum;
    sum = 0;
    __syncthreads();

    unsigned int index = threadIdx.x * THREADS;

    for(int i = 0; i < THREADS; ++i )
        atomicAdd(&sum, d_arr[index+i]);
    
    __syncthreads();
    
    //only one thread writes to d_total;
    if(threadIdx.x==0)
       *d_total = sum;

}

int main()
{
    unsigned int * h_arr = new(std::nothrow) unsigned int [SIZE];

    unsigned int h_total{0};
    unsigned int d_total{0};


    if(h_arr)
    {
        for(int i = 0; i < SIZE; ++i)
            h_arr[i] = rand()%10;
    }
    else
    {
        std::cerr << "Failure to allocate memmory for h_arr" << std::endl; 
        return 1;
    }

    for(int i = 0; i < SIZE; ++i)
        h_total += h_arr[i];
        
    std::cout << "Sum computed on host: " << h_total << std::endl;

    unsigned int * d_arr{nullptr};
    unsigned int * d_total_p{nullptr};

    cudaError_t error = cudaMalloc((void**)&d_arr, SIZE*sizeof(unsigned int));
    if(error != cudaSuccess)
    {
        std::cerr << "Failure of GPU memory allocation for d_arr" << std::endl;
        cudaGetErrorString(error);
        return 1;
    }
    error = cudaMalloc((void**)&d_total_p, sizeof(unsigned int));
    if(error != cudaSuccess)
    {
        std::cerr << "Failure of GPU memory allocation for d_total_p" << std::endl;
        cudaGetErrorString(error);
        return 1;
    } 
    
    error = cudaMemcpy(d_arr, h_arr, SIZE*sizeof(unsigned int), cudaMemcpyHostToDevice);
    if(error != cudaSuccess)
    {
        std::cerr << "Failure of transforing memory from host to GPU for d_arr" << std::endl;
        cudaGetErrorString(error);
        return 1;
    } 
    error = cudaMemcpy(d_total_p, &d_total, sizeof(unsigned int), cudaMemcpyHostToDevice);
    
    if(error != cudaSuccess)
    {
        std::cerr << "Failure of transforing memory from host to GPU for d_total_p" << std::endl;
        cudaGetErrorString(error);
        return 1;
    } 
    
    dim3 blocksPerGrid(1, 1, 1);
    dim3 threadsPerBlock(THREADS, 1, 1); 
    
    computeSum<<<blocksPerGrid, threadsPerBlock>>>(d_arr, d_total_p);

    error = cudaMemcpy(&d_total, d_total_p, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if(error != cudaSuccess)
    {
        std::cerr << "Failure of transforing memory from GPU to host for d_total" << std::endl;
        cudaGetErrorString(error);
        return 1;
    }     

    std::cout << "Sum computed on GPU: " << d_total << std::endl;


    delete [] h_arr;
    cudaFree(d_arr);
    cudaFree(d_total_p);


    return 0;
}