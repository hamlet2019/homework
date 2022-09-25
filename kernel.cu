#include "./kernel.h"
#include <host_defines.h>




__device__ const int WIDTH = 8192; 
__device__ const int HEIGHT = 4608;
__device__ const int MAX_Iteration = 256;// for color


static __global__ void
//__launch_bounds__(256,10)
kernel(const float lower_left_real, const float lower_left_imag, float factorX, float factorY, uchar4* __restrict__ dp_ptr)
{

    unsigned int x_dim = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y_dim = blockIdx.y * blockDim.y + threadIdx.y;

    //int index = WIDTH * y_dim + x_dim;//;
    int index = ((gridDim.x * blockDim.x))* y_dim + x_dim;


    // 1 / WIDTH = 0.00012207031f
    float c_real = (static_cast<float>(x_dim)* 0.00012207031f) * factorX + lower_left_real;
    float c_imag = (static_cast<float>(y_dim)* 0.00012207031f) * factorY + lower_left_imag;

    float z_real = 0.0f;
    float z_imag = 0.0f;
    float z_real_square = 0.0f;
    float z_imag_square = 0.0f;
    int iteration = 0;
    
 /*
#pragma unroll
    do
    {
        z_real_square = z_real*z_real; //use __fmul_rd more slowly
        z_imag_square = z_imag * z_imag;
        //float real_temp = z_real_square - z_imag_square + c_real; // a^2 - b^2 + c_real
        z_imag = 2.0 * z_real * z_imag + c_imag;                 // 2ab+c_imag
        z_real = z_real_square - z_imag_square + c_real; // a^2 - b^2 + c_real;
        ++iteration;
    } while ((z_real_square+z_imag_square) <= 4.0f && (iteration < MAX_Iteration));
 */    
 
 #pragma unroll
    while ((z_real_square + z_imag_square) <= 4.0f && (iteration < MAX_Iteration))
    {
        z_real_square = z_real * z_real; //use __fmul_rd more slowly
        z_imag_square = z_imag * z_imag;
        //float real_temp = z_real_square - z_imag_square + c_real; // a^2 - b^2 + c_real
        z_imag = 2.0 * z_real * z_imag + c_imag;                 // 2ab+c_imag
        z_real = z_real_square - z_imag_square + c_real; // a^2 - b^2 + c_real;
        ++iteration;
    }



    
    /*
    while (((z_real*z_real) + (z_imag*z_imag)) <= 4.0f && (iteration < MAX_Iteration))
    {
        //Z^2+c
        float real_temp =z_real*z_real - z_imag*z_imag + c_real; // a^2 - b^2 + c_real
        z_imag = 2.0*z_real*z_imag + c_imag;                 // 2ab+c_imag
        z_real = real_temp;
        ++iteration;
    }
    */
       

    //very low, not efficient
    /*
    int iteration = 0;
    int max_iteration = 256;// Iterate up to 255 times, corresponding to 255 colors
    while ((__fmul_rd(z_real, z_real) + __fmul_rd(z_imag, z_imag)) <= 4 && (iteration < max_iteration))
    {
        //Z^2+c
        float real_temp = __fmul_rd(z_real,z_real) - __fmul_rd(z_imag,z_imag) + c_real; // a^2 - b^2 + c_real
        z_imag = __fmul_rd(2.0,__fmul_rd(z_real,z_imag)) + c_imag;                 // 2ab+c_imag
        z_real = real_temp;
        ++iteration;
    }
    */
    
    int factor = (MAX_Iteration - iteration) * 10; //cool
    if (iteration == MAX_Iteration)
    {
        uchar4 output;
        output.w = 0;
        output.x = 0;
        output.y = 0;
        output.z = 255;
        dp_ptr[index] = output;
    }
    else
    {

        uchar4 output;
        output.w = factor;
        output.x = factor;
        output.y = factor;
        output.z = 255;
        dp_ptr[index] = output;
        //dp_ptr[index] = make_uchar4(iteration*0.5, iteration, iteration,255);
    }

    //try to avoid if else, but not so much influence
   /* int factor = (MAX_Iteration-iteration)*10;
    uchar4 output;
    output.w = factor;
    output.x = factor;
    output.y = factor;
    output.z = 255;
    dp_ptr[index] = output;*/

}

void CallingKernel(const float lower_left_real, const float lower_left_imag, float factorX, float factorY, uchar4* dp_ptr)
{
    //use 32*32 thread more slowly

    dim3 blockDim(16, 16, 1);
    dim3 gridDim(WIDTH / blockDim.x, HEIGHT / blockDim.y, 1);
    //printf("grid.x =  %d grid.y =  %d grid.z =  %d\n", gridDim.x, gridDim.y, gridDim.z);
    //printf("block.x =  %d block.y =  %d block.z =  %d\n", blockDim.x, blockDim.y, blockDim.z);
    kernel << < gridDim, blockDim, 0 >> > (lower_left_real, lower_left_imag, factorX, factorY,dp_ptr);
}
