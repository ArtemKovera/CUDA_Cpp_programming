#include<iostream>
#include<cuda_runtime.h>


#define ROWS 16
#define COLUMNS 32
#define VALUE_1 1.0
#define VALUE_2 2.0

//matrix size (rows*columns) must be less than or equal to 1024
float * createMatrix(const int rows, const int columns, const float value1, const float value2)
{
   float * matrix = new float [rows * columns];
   
   const int SIZE = rows * columns;
   int k = 0;
   int t = 3;
   int i = columns;
   int j = i;

   while(true)
   { 
       if(t%2)
       {
           while(k<columns)
           {
              matrix[--j] = value1;
              k++;
           }
       }
       else
       {
           while(k<columns)
           {
              matrix[--j] = value2;
              k++;
           }            
       }
       
       t++;
       k=0;
       i += columns;
       j = i;

       if(i > SIZE)
           break;
   }   
   
   return matrix;
}

void printMatrix(const float * const ptr, const int size, const int columns)
{   
    for(int i = 0; i < size; i++)
    {
        if( !(i%columns) && i )
            std::cout<<"\n";

        std::cout << ptr[i] << " ";    
    }
    std::cout<<std::endl;     
}

__global__ void transpose(const float * src, float * dst)
{
    //for debug
    //printf("blockDim.x = %d blockDim.y = %d blockIdx.x = %d blockIdx.y = %d threadIdx.x = %d threadIdx.y = %d\n", blockDim.x, blockDim.y, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
    
    int indexSrc = threadIdx.x * blockDim.y + threadIdx.y;
    int indexDst = threadIdx.x + blockDim.x * threadIdx.y;

    dst[indexDst] = src[indexSrc];    
} 

int main()
{
    const int SIZE = ROWS * COLUMNS;
    
    float * h_matrix = createMatrix(ROWS, COLUMNS, VALUE_1, VALUE_2);
   
    std::cout << "Original matrix: " << std::endl;
    printMatrix(h_matrix, SIZE, COLUMNS);

    const int ARRAY_BYTES = SIZE * sizeof(float);
    
    float * d_matrix = nullptr;
    float * d_transpose = nullptr;

    cudaError_t error = cudaMalloc( (void**)&d_matrix, ARRAY_BYTES);
    if(error != cudaSuccess)
    {
        fprintf(stderr, "Failure of allocating memory on GPU for d_matrix");
        cudaGetErrorString(error);
        return 1;
    }

    error = cudaMalloc( (void**)&d_transpose, ARRAY_BYTES);
    if(error != cudaSuccess)
    {
        fprintf(stderr, "Failure of allocating memory on GPU for d_transpose");
        cudaGetErrorString(error);
        return 1;
    } 

    error = cudaMemcpy(d_matrix, h_matrix, ARRAY_BYTES, cudaMemcpyHostToDevice);
    if(error != cudaSuccess)
    {
        fprintf(stderr, "Failure of transforing memory from host to GPU for d_matrix");
        cudaGetErrorString(error);
        return 1;
    }      
    
    dim3 threads = dim3(ROWS, COLUMNS, 1);
    dim3 blocks = dim3(1, 1, 1);
    
    transpose<<<blocks, threads>>>(d_matrix, d_transpose);
    cudaDeviceSynchronize(); 

    float * h_transpose =  new float [SIZE];
    
    error = cudaMemcpy(h_transpose, d_transpose, ARRAY_BYTES, cudaMemcpyDeviceToHost);
    if(error != cudaSuccess)
    {
        fprintf(stderr, "Failure of transforing memory from GPU to host for h_transpose");
        cudaGetErrorString(error);
        return 1;
    }  
    
    std::cout << "\nTransposed matrix: " << std::endl;
    printMatrix(h_transpose, SIZE, ROWS);    
    
    cudaFree(d_matrix);
    cudaFree(d_transpose);

    delete [] h_matrix;
    delete [] h_transpose;
    cudaDeviceReset();

    return 0;
}

//nvcc matrix_transpose_naive.cu