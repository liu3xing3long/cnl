// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

// Standard headers
#include <iostream>
#include <tchar.h>
#include <vector>
#include <map>
#include <stdarg.h>
#include <limits.h>
//#include <math.h>

// On Visual Studio 2008, M_PI constant is not defined even when math.h is included
#define M_PI       3.14159265358979323846

// If true, floating point numbers on a GPU is float. If false, it is double (possible only on 1.3 CUDA devices)
#define REAL_GPU_IS_FLOAT 1

const int iMaxBlockDimSize = 65535;
const int iMaxNumberOfTrainedElements = 1024;
const int iMaxNumberOfTHreadsInBlock = 512;

#define NO_PRINT_DEBUG 1

#define PRINT_MEMORY_INFO(a,b)
#define PRINT_MEMORY_INFO2(a,b)																																\
	/*if(blockIdx.x == gridDim.x-1 && threadIdx.x == blockDim.x-1)*/																							\
	{																																						\
		void *pPointer = (void*)(a);																														\
		int iMove = ((int)((b)-(a)))*4;																														\
		bool bFound = false;																																\
		for(int iAllocatedElement=0;iAllocatedElement<m_iAllocatedMemoryElements;++iAllocatedElement)											\
		{																																					\
			if(m_allocatedMemoryAddress[iAllocatedElement] == pPointer)																			\
			{																																				\
				bFound = true;																																\
				m_WasUsed[iAllocatedElement] = true;																										\
				if(iMove < m_allocatedMemorySize[iAllocatedElement])																				\
					/*printf("Used memory\t%x\tindex\t%d\t(OK , MAX %d)\n",pPointer,iMove,m_allocatedMemorySize[iAllocatedElement])*/;				\
				else																																		\
					printf("Used memory\t%x\tindex\t%d\t\t\t\t(TOO BIG , MAX %d)\n",pPointer,iMove,m_allocatedMemorySize[iAllocatedElement]);			\
				break;																																		\
			}																																				\
		}																																					\
		if(!bFound)																																			\
			printf("Used memory\t%x\tindex\t%d\t\t\t\t(NOT FOUND)\n",pPointer,iMove);																				\
	}

#ifndef NO_PRINT_DEBUG
	#define PRINT_DEBUG_INFO(a,...) printf(a,__VA_ARGS__)
	#define logTextParamsDebug(a,...) logTextParams(Logging::LT_DEBUG,a,__VA_ARGS__)
#else
	#define PRINT_DEBUG_INFO(a,...)
	#define logTextParamsDebug(a,...)
#endif

#define REAL_GPU_IS_FLOAT 1

#ifdef REAL_GPU_IS_FLOAT
	typedef float real_gpu;
#else
	typedef double real_gpu;
#endif

using namespace std;

#define ALIGN_UP(offset, alignment)												\
	(((offset) + (alignment) - 1) / (alignment)) * (alignment)

#define ALIGN_UP_ASSIGN(offset, alignment)										\
	(offset) = (((offset) + (alignment) - 1) / (alignment)) * (alignment)
	
#define HALF_WARP 16

// XML/string headers
#include "tinystr.h"
#include "tinyxml.h"
#include "MersenneTwister.h"

// Other project headers
#include "Logging.h"
#include "Global.h"
#include "InputTest.h"
#include "InputTestSet.h"
#include "Neuron.h"
#include "Layer.h"
#include "NeuralNetwork.h"
#include "MLP.h"
#include "CUDATools.h"
