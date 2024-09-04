#include<filesystem>
#include<iostream>
#include<fstream> 
#include<cuda_runtime.h>
#include<unistd.h>

#define FILE_TO_PROCESS "snail.bmp"
#define FILE_RESULT "snail_2.bmp"

#define IMAGE_HEIGHT 256
#define IMAGE_WIDTH  256

#define OFFSET 54

using namespace std;

__global__ void makeBlurryImage(unsigned char * arr, const size_t imageHeight, const size_t imageWidth)
{



    //get index
    int index = threadIdx.x * blockIdx.x;

    //exclude top and bottom pixels from blurring 
    if(index <= imageWidth * 3 && index >= imageWidth * 3 * (imageHeight - 1)) 
    {
        return;
    }  
    
    /*
    //exlude rightmost and leftmost pixels from blurring
    if(index%(imageWidth*3) == 0 || index%(imageWidth*3 + 1) == 0 || index%(imageWidth*3 + 2)== 0 || 
       index%((imageWidth-1)*3) == 0 || index%((imageWidth-1)*3+1) == 0 || index%((imageWidth-1)*3+2) == 0) 
    {
        return;
    }
    */
    
    //printf("index = %d\n", index);
    //read and compute value
    unsigned char temp = (arr[index] + arr[index + 3] + arr[index - 3] + 
                         arr[index - imageWidth * 3] + arr[index - (imageWidth-1) * 3] + arr[index - (imageWidth+1) * 3] +
                         arr[index + imageWidth * 3] + arr[index + (imageWidth-1) * 3] + arr[index + (imageWidth+1) * 3])/9 ;
    
 
    //sync threads before writing
    __syncthreads();
    //printf("index = %d; value = %d\n", index, temp);

    //write computed value
    arr[index] = temp;
    
     
    //for debug
    //printf("index = %d; thredIdx.x = %d; blockIdx.x = %d\n", index, threadIdx.x, blockIdx.x);
}

int main(int argc, char ** argv)
{
    filesystem::path filePath = FILE_TO_PROCESS;

    const unsigned int fileSize = static_cast<unsigned int>(filesystem::file_size(filePath));

    cout << "File size is: " << fileSize << " bytes" << endl;

    ifstream inputFile(filePath, std::ios::binary); 
  
    if (!inputFile.is_open()) 
    { 
        cerr << "Error opening the file!" << endl; 
        return 1; 
    } 
    else
    {
        std::cout << "File has been opened" << std::endl;
    }    

    char * buffer = new char [fileSize];

    inputFile.read(buffer, fileSize);

    if(!inputFile) 
    {
        std::cerr << "Error reading file, could only read " << inputFile.gcount() << " bytes" << std::endl;
    }
    else
    {
        std::cout << "File has been put in buffer" << std::endl;
    } 

    char * h_img = buffer + OFFSET; 



    //For debugging
    //std::cout << "First symbol in ptr is \"" <<  *h_img << "\" " <<std::endl;
    
    char * imgResult = new char [fileSize];

    std::copy(buffer, buffer+OFFSET, imgResult);     
    
    
    //------ CUDA PART ---------------
    char * d_img = nullptr;
    
    cudaError_t error = cudaMalloc((void**)&d_img, fileSize - OFFSET);

    if(error != cudaSuccess)
    {
        cerr << "Failure of GPU memory allocation for d_img" << std::endl;
        cudaGetErrorString(error);
        return 1;
    }

    error = cudaMemcpy(d_img, h_img, fileSize - OFFSET, cudaMemcpyHostToDevice);
    if(error != cudaSuccess)
    {
        cerr << "Failure of transforing memory from host to GPU for d_img" << std::endl;
        cudaGetErrorString(error);
        return 1;
    }
    
    makeBlurryImage<<<IMAGE_HEIGHT * 3, IMAGE_WIDTH>>>((unsigned char*)d_img, IMAGE_HEIGHT, IMAGE_WIDTH);

    cudaError_t err = cudaGetLastError();
    if ( err != cudaSuccess )
    {
       printf("CUDA Error: %s\n", cudaGetErrorString(err));       
    }    

    error = cudaMemcpy(imgResult + OFFSET, d_img, fileSize - OFFSET, cudaMemcpyDeviceToHost);
    if(error != cudaSuccess)
    {
        cerr << "Failure of transforing memory from GPU to host for d_img" << std::endl;
        cudaGetErrorString(error);
        return 1;
    }  
    
    
    ofstream result(FILE_RESULT, ofstream::binary);
    
    if (!result.is_open()) 
    { 
        cerr << "Error opening the output file!" << endl; 
        return 1; 
    } 
    else
    {
        std::cout << "Output file has been opened" << std::endl;
    }   


    result.write(imgResult, fileSize);  
    if(!result) 
    {
        std::cerr << "Error writing to output file " << std::endl;
    }
    else
    {
        std::cout << "Output file has been written" << std::endl;
    }     
    
    
    
    cudaFree(d_img);    

    
    delete [] buffer;
    delete [] imgResult;

    return 0;
}
