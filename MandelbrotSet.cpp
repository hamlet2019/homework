#include "./kernel.h"
#include "./bitmap.h"
#include "./CUDAHelper.h"
#include "./Timer.h"


#include <iostream>
#include <string>
#include <stdio.h>
#include <concepts>
#include <stdlib.h>
#include <ctime>
#include <chrono>
#include <thread>
#include <concepts>
#include <ratio>



#include "./jobs.h"
#include "./complex.h"
using namespace pfc;

const int WIDTH = 8192; 
const int HEIGHT = 4608; 


void DisplayCUDAInfo()
{
	printf("Starting...\n");

	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);

	if (deviceCount == 0)
	{
		printf("There are no available device(s) that support CUDA\n");
	}
	else
	{
		printf("Detected %d CUDA Capable device(s)\n", deviceCount);
	}

	int dev = 0, driverVersion = 0, runtimeVersion = 0;
	cuda::check(cudaSetDevice(dev));
	cudaDeviceProp deviceProp;
	cuda::check(cudaGetDeviceProperties(&deviceProp, dev));
	printf("Device %d: \"%s\"\n", dev, deviceProp.name);

	cudaDriverGetVersion(&driverVersion);
	cudaRuntimeGetVersion(&runtimeVersion);
	printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
		driverVersion / 1000, (driverVersion % 100) / 10,
		runtimeVersion / 1000, (runtimeVersion % 100) / 10);
	printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
		deviceProp.major, deviceProp.minor);
	printf("  Total amount of global memory:                 %.2f MBytes (%llu "
		"bytes)\n", (float)deviceProp.totalGlobalMem / pow(1024.0, 3),
		(unsigned long long)deviceProp.totalGlobalMem);
	printf("  GPU Clock rate:                                %.0f MHz (%0.2f "
		"GHz)\n", deviceProp.clockRate * 1e-3f,
		deviceProp.clockRate * 1e-6f);
	printf("  Memory Clock rate:                             %.0f Mhz\n",
		deviceProp.memoryClockRate * 1e-3f);
	printf("  Memory Bus Width:                              %d-bit\n",
		deviceProp.memoryBusWidth);

	if (deviceProp.l2CacheSize)
	{
		printf("  L2 Cache Size:                                 %d bytes\n",
			deviceProp.l2CacheSize);
	}

	printf("  Max Texture Dimension Size (x,y,z)             1D=(%d), "
		"2D=(%d,%d), 3D=(%d,%d,%d)\n", deviceProp.maxTexture1D,
		deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
		deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1],
		deviceProp.maxTexture3D[2]);
	printf("  Max Layered Texture Size (dim) x layers        1D=(%d) x %d, "
		"2D=(%d,%d) x %d\n", deviceProp.maxTexture1DLayered[0],
		deviceProp.maxTexture1DLayered[1], deviceProp.maxTexture2DLayered[0],
		deviceProp.maxTexture2DLayered[1],
		deviceProp.maxTexture2DLayered[2]);
	printf("  Total amount of constant memory:               %lu bytes\n",
		deviceProp.totalConstMem);
	printf("  Total amount of shared memory per block:       %lu bytes\n",
		deviceProp.sharedMemPerBlock);
	printf("  Total number of registers available per block: %d\n",
		deviceProp.regsPerBlock);
	printf("  Warp size:                                     %d\n",
		deviceProp.warpSize);
	printf("  Maximum number of threads per multiprocessor:  %d\n",
		deviceProp.maxThreadsPerMultiProcessor);
	printf("  Maximum number of threads per block:           %d\n",
		deviceProp.maxThreadsPerBlock);
	printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
		deviceProp.maxThreadsDim[0],
		deviceProp.maxThreadsDim[1],
		deviceProp.maxThreadsDim[2]);
	printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
		deviceProp.maxGridSize[0],
		deviceProp.maxGridSize[1],
		deviceProp.maxGridSize[2]);
	printf("  Maximum memory pitch:                          %lu bytes\n",
		deviceProp.memPitch);
}








int main()
{
	int cudaDeviceCount{};
	cuda::check(cudaGetDeviceCount(&cudaDeviceCount));

	size_t buffer;
	float time_GPU = 0.0;
	pfc::bitmap bmp{ WIDTH, HEIGHT };

	int ipos = 0;
	unsigned char r = 0;
	unsigned char g = 0;
	unsigned char b = 0;
	unsigned char no = 0;

	double factor_x = 0.0;
	double factor_y = 0.0;
	long int trans_size= 0;
	uchar4* pinned;
	uchar4* mem_host;
	uchar4* mem_dev;

	DisplayCUDAInfo();

	for (size_t i = 0; i < cudaDeviceCount; i++)
	{
		cudaSetDevice(i);
		cudaDeviceProp deviceProp = cudaDevicePropDontCare;
		cudaGetDeviceProperties(&deviceProp, i);
		//std::cout << deviceProp.name << " " << deviceProp.major << "." << deviceProp.minor << "\n";

		buffer = bmp.size() * sizeof(uchar4); //RGB Color
		
		//cuda malloc
		auto dp_bmp{ cuda::makeUnique<uchar4>(buffer) };
		auto hp_bmp{ std::make_unique<uchar4[]>(buffer) };

		//Pinned Host Memory
		cuda::check(cudaHostAlloc((void**)&pinned, buffer, cudaHostAllocDefault));

		//zero copy
		//cuda::check(cudaHostAlloc((void**)&mem_host, buffer, cudaHostAllocWriteCombined | cudaHostAllocMapped));
		// pass the pointer to device
		//cuda::check(cudaHostGetDevicePointer(&mem_dev,mem_host, 0));
	

		//pitched
		//size_t pitch;
		//cudaMallocPitch((void**)&pinned, (size_t*)&pitch, WIDTH* sizeof(uchar4), (size_t)HEIGHT);
		//printf("pitch: %d , num is %d \n", pitch, pitch / sizeof(uchar4));

		for (std::size_t i{}; auto const& [lower_l, upper_r, center_point, wh] : pfc::jobs <>{ "./jobs/jobs-004.txt" })
		{
			//std::cout << "ll image is: " << lower_l.imag << " - " << "ll real: " << lower_l.real << std::endl;
			//std::cout << "ur image is: " << upper_r.imag << "-" << "ur real: " << upper_r.real << std::endl;
			//std::cout << "cp image is: " << center_point.imag << "-" << "cp real: " << center_point.real << std::endl;
			//std::cout << "wh is: " << wh.first << "-" << "wh second : " << wh.second << std::endl;

			factor_x = (static_cast<double>(center_point.real-lower_l.real) * (static_cast<double>(WIDTH))) / (static_cast<float>(WIDTH) / 2);
			factor_y = (static_cast<double>(center_point.imag-lower_l.imag) * static_cast<double>(WIDTH)) / (HEIGHT/2);

			double test = (center_point.imag - lower_l.imag)* (static_cast<double>(WIDTH));
			trans_size += (buffer / 1024 / 1024);
			std::cout << "Image size(Megabytes): " << trans_size <<" will be transferred" << std::endl;
			//std::cout << "factor1: " << factor_x << std::endl;
			//std::cout << "factor2: " << factor_y << std::endl;

			double x_dim = WIDTH/2;
			double y_dim = HEIGHT/2;

			double d_real = (static_cast<double>(x_dim) / static_cast<double>(WIDTH)) * factor_x +lower_l.real;
			double d_imag = (static_cast<double>(y_dim) / static_cast<double>(WIDTH)) * factor_y + lower_l.imag;
			std::cout << "**************************************************" << std::endl;

			auto const elapsed = Timer::timedRun<std::chrono::steady_clock>(
				[&hp_bmp, &buffer, &dp_bmp, &lower_l, &factor_x, &factor_y, &pinned,&mem_host, &mem_dev] {

					CallingKernel(static_cast<float>(lower_l.real), static_cast<float>(lower_l.imag),
						static_cast<float>(factor_x), static_cast<float>(factor_y), dp_bmp.get());

					//CallingKernel(static_cast<float>(lower_l.real), static_cast<float>(lower_l.imag),
						//static_cast<float>(factor_x), static_cast<float>(factor_y), mem_dev);
					
					//for cuda malloc
					//cuda::check(cudaMemcpy(hp_bmp.get(), dp_bmp.get(), buffer, cudaMemcpyDeviceToHost));
					
					//for cuda pinned host memory
					cuda::check(cudaMemcpy(pinned, dp_bmp.get(), buffer, cudaMemcpyDeviceToHost));
					
					// transfer data from host to device for zero copy
					//cuda::check(cudaMemcpy(mem_host, mem_dev, buffer, cudaMemcpyDeviceToHost));
					
				});

			
			time_GPU += Timer::to<std::ratio<1>>(elapsed);
			std::cout << "-"<< i+1<< ":" << "GPU time elapsed : " << time_GPU << std::endl;
			
			//write pixel into File

			for (auto& pixel : bmp.span())
			{
				r = pinned[ipos].w;// hp_bmp for cudamalloc
				b = pinned[ipos].x;
				g = pinned[ipos].y;
				pixel = { b,g,r };//b,g,r
				ipos += 1;
			}
			bmp.to_file(std::to_string(i) +".bmp");
			++i;
			ipos = 0;

		}
			


		cuda::check(cudaFreeHost(pinned));
		//cuda::check(cudaFreeHost(mem_host));

	}
	//std::cout << "CPU time elapsed :" << 0.0952095 << std::endl;
	//std::cout << "Speed Up is " << (0.0952095 / time_GPU) * 100 << "%" << std::endl;
	//std::cout << "It works" << std::endl;
	cudaDeviceReset();
}
