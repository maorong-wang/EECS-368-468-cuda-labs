/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  
 *
 * This software and the information contained herein is PROPRIETARY and 
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and 
 * conditions of a Non-Disclosure Agreement.  Any reproduction or 
 * disclosure to any third party without the express written consent of 
 * NVIDIA is prohibited.     
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.  This source code is a "commercial item" as 
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer software" and "commercial computer software 
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */

/* Matrix multiplication: C = A * B.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "matrixmul.h"

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)
{
	//use padding here to avoid bank conflict
	__shared__ float Ms[17*16];
	__shared__ float Ns[17*16];
	__shared__ double Ps[17*16];
	unsigned int xIndex = __mul24(blockDim.x,blockIdx.x)+threadIdx.x;
	unsigned int yIndex = __mul24(blockDim.y,blockIdx.y)+threadIdx.y;
	
	int iter;
	Ps[(blockDim.x+1)*threadIdx.y+threadIdx.x]=0.0f;
	
	for(iter=0;iter<1+(M.width-1)/16;iter++)
	{
		if(xIndex<N.width || yIndex < M.height)
		{
			Ms[threadIdx.y*(blockDim.x+1)+threadIdx.x] = M.elements[yIndex*M.width+iter*blockDim.x+threadIdx.x];
			Ns[threadIdx.y*(blockDim.x+1)+threadIdx.x] = N.elements[xIndex+(iter*blockDim.y+threadIdx.y)*N.width];
		}
		else
		{
			Ms[threadIdx.y*(blockDim.x+1)+threadIdx.x] = 0.0f;
			Ns[threadIdx.y*(blockDim.x+1)+threadIdx.x] = 0.0f;	
		}
		__syncthreads();
		
		if(xIndex<N.width && yIndex < M.height)
		{
			int temp;
			for(temp=0;temp<blockDim.x;temp++)
			{
				Ps[(blockDim.x+1)*threadIdx.y+threadIdx.x] += Ms[threadIdx.y*(blockDim.x+1)+temp] * Ns[temp*(blockDim.x+1)+threadIdx.x];
			}
		}
		__syncthreads();
	}


	if(xIndex<N.width && yIndex < M.height)
	{
		P.elements[yIndex*P.width+xIndex]=Ps[threadIdx.y*(blockDim.x+1)+threadIdx.x];
	}
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
