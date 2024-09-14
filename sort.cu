#include<iostream>
#include<cuda_runtime.h>

#define SIZE 16

//this kernel takes pointers to two arrays on the GPU of the same size: the sourse array and the destination array
//so, this algorithm uses extra GPU memory 
__global__ void mergeSortGPU(float * arraySource, float * arrayDestination, const unsigned int subArraySize)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x * subArraySize;
   
    float * leftS  = &arraySource[index];
    float * rightS = &arraySource[index + subArraySize/2];
    float * leftD  = &arrayDestination[index];
    float * endD   = &arrayDestination[index + subArraySize];   

    while(leftD != endD)
    {
        if(leftS==&arraySource[index + subArraySize/2])
        {
            *leftD = *rightS;
            ++leftD;
            ++rightS;  
            continue;          
        }

        if(rightS==&arraySource[index + subArraySize])
        {
            *leftD = *leftS;
            ++leftD;
            ++leftS; 
            continue;          
        }


        if(*leftS < *rightS)
        {
            *leftD = *leftS;
            ++leftD;
            ++leftS;
        }
        else
        {
            *leftD = *rightS;
            ++leftD;
            ++rightS;
        }
   
    }
   
}


int main()
{
    
    float arr [SIZE] = {7.8, -2.3, -4, 33, 7.1, 45, 98, -18.5, 9.7, 12.7, -8.9, 6.8, -12, 9.01, 68.2, 8.8};
    dim3 blocks (1, 1, 1);
    dim3 threads1 (8, 1, 1);
    dim3 threads2 (4, 1, 1);
    dim3 threads3 (2, 1, 1);
    dim3 threads4 (1, 1, 1);

    std::cout << "Original array:";

    for(int i = 0; i < SIZE; ++i)
    {
        std::cout << arr[i] << "    ";
    }

    std::cout << std::endl;

    float * d_arr = nullptr; 
    float * d_arr2 = nullptr;
    
    cudaError_t error = cudaMalloc((void **)&d_arr, SIZE * sizeof(float));   
    if(error != cudaSuccess)
    {
        std::cerr << "Failure of GPU memory allocation for d_arr" << std::endl;
        std::cerr << cudaGetErrorString(error) << std::endl;
        return 1;
    }   

    error = cudaMalloc((void **)&d_arr2, SIZE * sizeof(float));   
    if(error != cudaSuccess)
    {
        std::cerr << "Failure of GPU memory allocation for d_arr" << std::endl;
        std::cerr << cudaGetErrorString(error) << std::endl;
        return 1;
    }     
 
    error=cudaMemcpy(d_arr, arr, SIZE * sizeof(float), cudaMemcpyHostToDevice); 
    if(error != cudaSuccess)
    {
        std::cerr << "Failure of transforing memory from host to GPU for d_arr" << std::endl;
        std::cerr << cudaGetErrorString(error) << std::endl;
        return 1;
    }
    
    mergeSortGPU<<<blocks, threads1>>>(d_arr, d_arr2, 2);
    error = cudaGetLastError();
    if ( error != cudaSuccess )
    {
        std::cerr << cudaGetErrorString(error) << std::endl;
       return 1;      
    }  

    mergeSortGPU<<<blocks, threads2>>>(d_arr2, d_arr, 4);
    error = cudaGetLastError();
    if ( error != cudaSuccess )
    {
        std::cerr << cudaGetErrorString(error) << std::endl;
       return 1;      
    }  

    mergeSortGPU<<<blocks, threads3>>>(d_arr, d_arr2, 8);
    error = cudaGetLastError();
    if ( error != cudaSuccess )
    {
        std::cerr << cudaGetErrorString(error) << std::endl;
       return 1;      
    } 

    mergeSortGPU<<<blocks, threads4>>>(d_arr2, d_arr, 16);
    error = cudaGetLastError();
    if ( error != cudaSuccess )
    {
        std::cerr << cudaGetErrorString(error) << std::endl;
       return 1;      
    } 

    error = cudaMemcpy(arr, d_arr, SIZE*sizeof(float), cudaMemcpyDeviceToHost);
    error = cudaGetLastError();
    if ( error != cudaSuccess )
    {
        std::cerr << cudaGetErrorString(error) << std::endl;
       return 1;      
    }  
    
    std::cout << "Sorted array:";

    for(int i = 0; i < SIZE; ++i)
    {
        std::cout << arr[i] << "    " ;
    }

    std::cout << std::endl;
    
    
    cudaFree(d_arr);
    cudaFree(d_arr2);
    
    return 0;
}