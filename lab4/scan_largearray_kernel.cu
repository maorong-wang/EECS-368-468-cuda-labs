#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

// includes, kernels
#include <assert.h>


#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
// Lab4: You can use any other block size you wish.
#define BLOCK_SIZE 1024
//use padding to avoid bank conflicts
#define pad(x) ((x)+(x)/NUM_BANKS)
// Lab4: Host Helper Functions (allocate your own data structure...)


// Lab4: Device Functions


// Lab4: Kernel Functions

__global__ void reduction(float* outArray, float *inArray, float* sum,int numElements)
{

	int globalId = threadIdx.x+blockIdx.x*blockDim.x;
	//use shared memory here to be more effective
	//also use tiled implementation here for effectiveness.
	//again, we use padding to avoid bank conflicts.(pad macro defined above)
	__shared__ float scan_array[pad(BLOCK_SIZE)];
	/*
	if(globalId<numElements)
	{
		scan_array[threadIdx.x]=inArray[globalId];
	}
	else
	{
		scan_array[threadIdx.x]=0;
	}*/
	//actrually we do not need the if statement above, since the redundant elements will never influence the result. Since reading those values do not create a segment fault, just ignore them.
	scan_array[pad(threadIdx.x)]=inArray[globalId];
	__syncthreads();


	int stride = 1;
	//declear index out of loop here to increase the performance
	int index;
	#pragma unroll
	while(stride < BLOCK_SIZE)
	{
		index = ((threadIdx.x+1)*stride)*2 -1;
		if(index < BLOCK_SIZE)
		{
			scan_array[pad(index)] += scan_array[pad(index-stride)];
		}
		stride = stride << 1;
		__syncthreads();
	}
	
	outArray[globalId]=scan_array[pad(threadIdx.x)];

	if(threadIdx.x==BLOCK_SIZE-1)
	{
		sum[blockIdx.x]=scan_array[pad(threadIdx.x)];
	}
}

__global__ void fixBetweenBlocks(float* outArray,float* sum,float* cumsum, int numBlocks)
{
	cumsum[0]=0;
	int i;
	//#pragma unroll
	//do not unroll here, the performance get worse with it.
	for(i=1;i<numBlocks;++i)
	{
		cumsum[i] = cumsum[i-1] + sum[i-1];
	}
}

__global__ void postScan(float* outArray,float* cumsum,int numElements)
{
	
	int globalId = threadIdx.x+blockIdx.x*blockDim.x;

	__shared__ float scan_array[pad(BLOCK_SIZE)];

	scan_array[pad(threadIdx.x)]=outArray[globalId];

	__syncthreads();
	
	int stride = BLOCK_SIZE >> 1;
	int index;
	#pragma unroll
	while(stride > 0)
	{
		index = (threadIdx.x+1)*stride*2 -1;
		if(index < BLOCK_SIZE) 
		{
			scan_array[pad(index+stride)] += scan_array[pad(index)];
		}
		stride = stride >> 1;
		__syncthreads();
	}
	
	scan_array[pad(threadIdx.x)] += cumsum[blockIdx.x];
	__syncthreads();
	if(globalId < numElements-1)
	{
		outArray[globalId+1]=scan_array[pad(threadIdx.x)];
	}
	if(globalId==0)
		outArray[0]=0;
}
// **===-------- Lab4: Modify the body of this function -----------===**
// You may need to make multiple kernel calls, make your own kernel
// function in this file, and then call them from here.
void prescanArray(float *outArray, float *inArray, float *sum, float* cumsum, int numElements)
{
	int numBlocks;
	numBlocks = (numElements-1)/BLOCK_SIZE + 1;
	dim3 grids(numBlocks,1,1);
	dim3 blocks(BLOCK_SIZE,1,1);
	reduction<<<grids,blocks,BLOCK_SIZE * sizeof(float)>>>(outArray,inArray,sum,numElements);
	fixBetweenBlocks<<<1,1>>>(outArray,sum,cumsum,numBlocks);
	postScan<<<grids,blocks,BLOCK_SIZE* sizeof(float)>>>(outArray,cumsum,numElements);
}
// **===-----------------------------------------------------------===**


#endif // _PRESCAN_CU_
