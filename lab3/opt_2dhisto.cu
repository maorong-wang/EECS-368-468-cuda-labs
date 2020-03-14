#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"

__global__ void histo(uint32_t* d_input, uint32_t* t_bin);
__global__ void convert(uint8_t* d_bin,uint32_t* t_bin);
void opt_2dhisto(uint32_t* d_input,uint8_t* d_bin,uint32_t *t_bin)
{
    /* This function should only contain a call to the GPU 
       histogramming kernel. Any memory allocations and
       transfers must be done outside this function */

	dim3 dimgrid(INPUT_WIDTH,1,1);
	dim3 dimblock(128,1,1);
	//dim3 dimblock(512,1,1);

	histo<<<dimgrid,dimblock>>>(d_input,t_bin);
	
	cudaThreadSynchronize();
	dim3 dimgrid2(1+(HISTO_WIDTH*HISTO_HEIGHT-1)/256,1,1);
	dim3 dimblock2(256,1,1);
	convert<<<dimgrid2,dimblock2>>>(d_bin,t_bin);
	cudaThreadSynchronize();
}

/* Include below the implementation of any other functions you need */
__global__ void histo(uint32_t* d_input, uint32_t* t_bin)
{
	const int globalTid = blockIdx.x*blockDim.x + threadIdx.x;
	const int numThreads = blockDim.x*gridDim.x;
	//there is a small risk here. When the size of the input is smaller than the size of the histogram, 
	//this may cause wrong solutions(several bins will have non-zero initial values). But in reality, size of histograms is always larger than size of input.
	if (globalTid < HISTO_WIDTH*HISTO_HEIGHT) 
	{
		t_bin[globalTid] = 0;
	}
	__syncthreads();
	int input_size = INPUT_WIDTH*INPUT_HEIGHT;	
	//the idea of this loop comes from the slides, it's good to let threads executing multiple times for several inputs, rather than a single input.
	//I guess this kind of memory access also helps in avoiding bank conflict (but I'm not sure)
	for (int pos = globalTid; pos < input_size; pos+=numThreads)
	{
		int index=d_input[pos];
		//abandoned code:
		//atomicAdd(&(t_bin[index]),1);
		//reason:
		//instead of calling atomicAdd here directly, we can check whether the value has exceeded 255, this check can reduce the call of atomicAdd, and help a lot with the efficiency
		// we can gain 3x performance here with the if statement
		if (t_bin[index]<255)
		{
			atomicAdd(&(t_bin[index]),1);
		}
	}
}


__global__ void convert(uint8_t* d_bin,uint32_t* t_bin)
{
	int globalid=blockIdx.x*blockDim.x+threadIdx.x;
	// This check is not necessary now, with the if statement above, but I'd like to keep them. :)
	d_bin[globalid]=(uint8_t)(t_bin[globalid]>255? 255:t_bin[globalid]);
}


void opt_2dhisto_init(uint32_t** input, uint32_t* (&d_input), uint8_t* (&d_bin), uint32_t* (&t_bin))
{
	cudaMalloc((void**) &d_input, INPUT_WIDTH*INPUT_HEIGHT*sizeof(uint32_t));
	cudaMalloc((void**) &d_bin, HISTO_WIDTH*HISTO_HEIGHT*sizeof(uint8_t));
	cudaMalloc((void**) &t_bin, HISTO_WIDTH*HISTO_HEIGHT*sizeof(uint32_t));
	
	uint32_t* temp=d_input;	
	for(int i=0;i<INPUT_HEIGHT;i++)
	{
		cudaMemcpy(temp,input[i],INPUT_WIDTH*sizeof(uint32_t),cudaMemcpyHostToDevice);
		temp+=INPUT_WIDTH;
	}
} 

void opt_2dhisto_finalize(uint32_t* &d_input,uint8_t* &d_bin,uint32_t* &t_bin, uint8_t* kernel_bin)
{
	cudaMemcpy(kernel_bin,d_bin, HISTO_WIDTH*HISTO_HEIGHT*sizeof(uint8_t), cudaMemcpyDeviceToHost);
	cudaFree(d_input);
	cudaFree(d_bin);
	cudaFree(t_bin);
}
