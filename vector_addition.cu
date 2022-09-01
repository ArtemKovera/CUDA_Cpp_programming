#include<iostream>
#include<cuda_runtime.h>

#define ARRAY_SIZE 1048576 
#define VALUE_1 3.3
#define VALUE_2 2.2
#define BLOCK_SIZE 1024
#define CHECK_FROM 1048560 
#define CHECK_TO   1048576

__global__ void sumVectors (const float * vec1, const float * vec2, float * result)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    result[index] = vec1[index] + vec2[index];
}


int main()
{
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

    float * h_vec1   = new float [ARRAY_SIZE];
    float * h_vec2   = new float [ARRAY_SIZE]; 
    float * h_result = new float [ARRAY_SIZE]; 
    
    for(int i = 0; i < ARRAY_SIZE; i++)
    {
        h_vec1[i] = VALUE_1;
        h_vec2[i] = VALUE_2;
    }

    float * d_vec1   = nullptr;
    float * d_vec2   = nullptr;
    float * d_result = nullptr;

    cudaError_t error = cudaMalloc((void**)&d_vec1, ARRAY_BYTES);
    if(error != cudaSuccess)
    {
        fprintf(stderr, "Failure of GPU memory allocation for d_vec1");
        cudaGetErrorString(error);
        return 1;
    }

    error = cudaMalloc((void**)&d_vec2, ARRAY_BYTES);
    if(error != cudaSuccess)
    {
        fprintf(stderr, "Failure of GPU memory allocation for d_vec2");
        cudaGetErrorString(error);
        return 1;
    }   

    error = cudaMalloc((void**)&d_result, ARRAY_BYTES);
    if(error != cudaSuccess)
    {
        fprintf(stderr, "Failure of GPU memory allocation for d_result");
        cudaGetErrorString(error);
        return 1;
    }  

    error = cudaMemcpy(d_vec1, h_vec1, ARRAY_BYTES, cudaMemcpyHostToDevice);
    if(error != cudaSuccess)
    {
        fprintf(stderr, "Failure of transforing memory from host to GPU for d_vec1");
        cudaGetErrorString(error);
        return 1;
    }   

    error = cudaMemcpy(d_vec2, h_vec2, ARRAY_BYTES, cudaMemcpyHostToDevice);
    if(error != cudaSuccess)
    {
        fprintf(stderr, "Failure of transforing memory from host to GPU for d_vec2");
        cudaGetErrorString(error);
        return 1;
    }    

    dim3 threads = dim3(BLOCK_SIZE, 1, 1);
    dim3 blocks = dim3(ARRAY_SIZE / BLOCK_SIZE, 1, 1);
    
    //launching kernel
    sumVectors<<<blocks, threads>>>(d_vec1, d_vec2, d_result);
    cudaDeviceSynchronize();
    
    error = cudaMemcpy(h_result, d_result, ARRAY_BYTES, cudaMemcpyDeviceToHost);
    if(error != cudaSuccess)
    {
        fprintf(stderr, "Failure of transforing memory from GPU to host for d_result");
        cudaGetErrorString(error);
        return 1;
    }  
    
    //checking result
    for(int i = CHECK_FROM; i < CHECK_TO; i++)
    {
        std::cout << h_result[i] << std::endl;
    }  

    cudaFree(d_vec1);          
    cudaFree(d_vec2);          
    cudaFree(d_result);          

    delete [] h_vec1;
    delete [] h_vec2;
    delete [] h_result;

    cudaDeviceReset();

    return 0;
}