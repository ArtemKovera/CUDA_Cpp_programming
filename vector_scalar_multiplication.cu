#include<iostream>
#include<vector>

#define ARRAY_SIZE 8
#define SCALAR 10.0
#define BLOCKS 1

__global__ void vectorScalarMultiplication (float * d_array, float scalar)
{
    d_array[threadIdx.x] *= scalar;
}

float h_vector[ARRAY_SIZE];

int main(void)
{
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

   
    for(int i = 0; i < ARRAY_SIZE; i++)
    {
        h_vector[i] = float(i);
    }

    float * d_vector  = NULL;
    
    //allocate memory on GPU
    cudaMalloc( (void**) &d_vector, ARRAY_BYTES);

    //transfer data to GPU
    cudaMemcpy(d_vector, h_vector, ARRAY_BYTES, cudaMemcpyHostToDevice);

    vectorScalarMultiplication <<<BLOCKS, ARRAY_SIZE>>>(d_vector, SCALAR);

    //transfer result from GPU to the application
    cudaMemcpy(h_vector, d_vector, ARRAY_BYTES, cudaMemcpyDeviceToHost);
    
    //print the result
    for(int i = 0; i < ARRAY_SIZE; i++)
    {
        std::cout << h_vector[i] << std::endl;
    }

    cudaFree(d_vector);
    
    cudaDeviceReset();

    return 0;
}