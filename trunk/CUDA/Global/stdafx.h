// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

// Standard headers
#ifdef _MSC_VER       // Identifies Microsoft compilers
	#include <windows.h>
	//#include <sdgstd.h>
	//#include <io.h>
	//#include <fcntl.h>
	//#include <sys/types.h>
	//#include <sys/stat.h>
	//#include <share.h>
#else
	#include <sys/time.h>
	#include <ctime>
#endif

#include <iostream>
#include <tchar.h>
#include <vector>
#include <map>
#include <stdarg.h>
#include <limits.h>
#include <algorithm>
#include <limits>
#include <math.h>

// On Visual Studio 2008, M_PI constant is not defined even when math.h is included
#define M_PI       3.14159265358979323846
const double dMinNeuralNetworkValue = -1.0;
const double dMaxNeuralNetworkValue = 1.0;
#define XML_CLASSIFICATION_CHAR_START	'['
#define XML_CLASSIFICATION_CHAR_END		']'
const char cDivider = ';';

// If true, floating point numbers on a GPU is float. If false, it is double (possible only on 1.3 CUDA devices)
#define REAL_GPU_IS_FLOAT 1

const int iMaxBlockDimSize = 65535;
const int iMaxNumberOfTrainedElements = 1024;


#if defined(__DEVICE_EMULATION__)
	#define PRINT_MEMORY_INFO(a,b)																																\
		/*if(blockIdx.x == gridDim.x-1 && threadIdx.x == blockDim.x-1)*/																						\
		if(g_uiAllowedTypesConsole & (unsigned int)Logging::LT_MEMORY)																							\
		{																																						\
			void *pPointer = (void*)(a);																														\
			int iMove = ((int)((b)-(a)))*4;																														\
			bool bFound = false;																																\
			for(int iAllocatedElement=0;iAllocatedElement<m_iAllocatedMemoryElements;++iAllocatedElement)														\
			{																																					\
				if(m_allocatedMemoryAddress[iAllocatedElement] == pPointer)																						\
				{																																				\
					bFound = true;																																\
					m_WasUsed[iAllocatedElement] = true;																										\
					if(iMove < m_allocatedMemorySize[iAllocatedElement])																						\
						/*printf("Used memory\t%x\tindex\t%d\t(OK , MAX %d)\n",pPointer,iMove,m_allocatedMemorySize[iAllocatedElement])*/;						\
					else																																		\
						printf("Used memory\t%x\tindex\t%d\t\t\t\t(TOO BIG , MAX %d)\tBlockIdx\t%d\tthreadIdx\t%d\n"											\
							,pPointer,iMove,m_allocatedMemorySize[iAllocatedElement],blockIdx.x,threadIdx.x);													\
					break;																																		\
				}																																				\
			}																																					\
			if(!bFound)																																			\
				printf("Used memory\t%x\tindex\t%d\t\t\t\t(NOT FOUND)\tBlockIdx\t%d\tthreadIdx\t%d\n",pPointer,iMove,blockIdx.x,threadIdx.x);					\
		}
#else
	#define PRINT_MEMORY_INFO(a,b)
#endif


#if defined(__DEVICE_EMULATION__)
	#define PRINT_DEBUG_INFO(a,...)										\
	if(g_uiAllowedTypesConsole & (unsigned int)Logging::LT_DEBUG)		\
	{																	\
		printf(a,__VA_ARGS__);											\
	}
#else
	#define PRINT_DEBUG_INFO(a,...)
#endif



#define REAL_GPU_IS_FLOAT 1

#ifdef REAL_GPU_IS_FLOAT
	typedef float real_gpu;
#else
	typedef double real_gpu;
#endif

const int iMaxNumberOfSharedMemoryElementsForWeights = 15600 / sizeof(real_gpu);

using namespace std;

#define ALIGN_UP(offset, alignment)												\
	(((offset) + (alignment) - 1) / (alignment)) * (alignment)

#define ALIGN_UP_ASSIGN(offset, alignment)										\
	(offset) = (((offset) + (alignment) - 1) / (alignment)) * (alignment)
	
const int HALF_WARP = 16;

// XML/string headers
#include "tinystr.h"
#include "tinyxml.h"
#include "MersenneTwister.h"

// Other project headers
#include "Logging.h"
#include "AttributeMapping.h"
#include "InputTest.h"
#include "InputTestSet.h"
#include "Global.h"
#include "Neuron.h"
#include "Layer.h"
#include "NeuralNetwork.h"
#include "MLP.h"
#include "CUDATools.h"
