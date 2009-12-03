// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

// Standard headers
#include <iostream>
#include <tchar.h>
#include <vector>
#include <stdarg.h>
#include <limits.h>
//#include <math.h>

// On Visual Studio 2008, M_PI constant is not defined even when math.h is included
#define M_PI       3.14159265358979323846

// If true, floating point numbers on a GPU is float. If false, it is double (possible only on 1.3 CUDA devices)
#define REAL_GPU_IS_FLOAT 1

//#ifdef REAL_GPU_IS_FLOAT
	typedef float real_gpu;
//#else
//	typedef double real_gpu;
//#endif

using namespace std;

#define ALIGN_UP(offset, alignment)												\
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
