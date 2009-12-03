#include "..\Global\stdafx.h"

__global__ void calculateErrorInLastLayerKernel(const real_gpu *dp_pCorrectOutput,const real_gpu *dp_pNetworkOutput,real_gpu *dp_pErrors)
{
	dp_pErrors[threadIdx.x] = dp_pNetworkOutput[threadIdx.x] - dp_pCorrectOutput[threadIdx.x];
}

extern "C" void calculateErrorInLastLayerCUDA(const real_gpu *dp_pCorrectOutput,const real_gpu *dp_pNetworkOutput,real_gpu *dp_pErrors,int p_iOutputNeuronCount)
{
	//int iSharedMem = 
	calculateErrorInLastLayerKernel <<<1,p_iOutputNeuronCount>>> (dp_pCorrectOutput,dp_pNetworkOutput,dp_pErrors);
}

__global__ void calculateErrorInNotLastLayerKernel(const real_gpu *dp_pNextLayerWeights,const real_gpu *dp_pNextLayerError,real_gpu *dp_pThisLayerError,int p_iNextLayerNeuronCount)
{
	real_gpu dError = 0.0f;
	
	for(int iWeightIndex = 0;iWeightIndex < p_iNextLayerNeuronCount; ++iWeightIndex)
	{
		dError += dp_pNextLayerWeights[iWeightIndex*blockDim.x + threadIdx.x] * dp_pNextLayerError[iWeightIndex];
	}
	
	dp_pThisLayerError[threadIdx.x] = dError;
}

extern "C" void calculateErrorInNotLastLayerCUDA(const real_gpu *dp_pNextLayerWeights,const real_gpu *dp_pNextLayerError,real_gpu *dp_pThisLayerError,int p_iThisLayerNeuronCount,int p_iNextLayerNeuronCount)
{
	calculateErrorInNotLastLayerKernel <<<1,p_iThisLayerNeuronCount>>> (dp_pNextLayerWeights,dp_pNextLayerError,dp_pThisLayerError,p_iNextLayerNeuronCount);
}
