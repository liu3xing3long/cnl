#include "..\Global\stdafx.h"

__global__ void calculateErrorInLastLayerKernel(const real_gpu *dp_pCorrectOutput,const real_gpu *dp_pNetworkOutput,real_gpu *dp_pErrors)
{
	dp_pErrors[threadIdx.x] = dp_pNetworkOutput[threadIdx.x] - dp_pCorrectOutput[threadIdx.x];
	PRINT_DEBUG_INFO("GPU: Test in batch nr 0 , Output %d : Network = %f , Correct  = %f , Error = %f\n",threadIdx.x,dp_pNetworkOutput[threadIdx.x],dp_pCorrectOutput[threadIdx.x],dp_pErrors[threadIdx.x]);
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

	PRINT_DEBUG_INFO("GPU: Test index 0 , Neuron index %d : Error %f\n",threadIdx.x,dError);
}

extern "C" void calculateErrorInNotLastLayerCUDA(const real_gpu *dp_pNextLayerWeights,const real_gpu *dp_pNextLayerError,real_gpu *dp_pThisLayerError,int p_iThisLayerNeuronCount,int p_iNextLayerNeuronCount)
{
	calculateErrorInNotLastLayerKernel <<<1,p_iThisLayerNeuronCount>>> (dp_pNextLayerWeights,dp_pNextLayerError,dp_pThisLayerError,p_iNextLayerNeuronCount);
}

__global__ void updateWeightsInTrainingKernel(const real_gpu *dp_pThisLayerError,const real_gpu *dp_pDerivativeOfLastOutput,const real_gpu *dp_pLayerBeforeOutputs,real_gpu p_dEta,int p_iNumOutputsLayerBefore,real_gpu *dp_pThisLayerWeights)
{
	//dp_pThisLayerWeights[0] = 3456;


	real_gpu *d_pThisNeuronWeights = dp_pThisLayerWeights + threadIdx.x * (p_iNumOutputsLayerBefore+1);
	real_gpu dErrorMultDerivativeMultEta = dp_pThisLayerError[threadIdx.x] * dp_pDerivativeOfLastOutput[threadIdx.x] * p_dEta;

	PRINT_DEBUG_INFO("GPU: Test 0 , Neuron %d : First weight: %f , dErrorMultDerivativeMultEta: %f , p_iNumOutputsLayerBefore: %d\n",threadIdx.x,d_pThisNeuronWeights[0],dErrorMultDerivativeMultEta,p_iNumOutputsLayerBefore);
	
	for(unsigned uWeightIndex = 0;uWeightIndex < p_iNumOutputsLayerBefore;++uWeightIndex)
	{ 
		real_gpu dChange = dErrorMultDerivativeMultEta * dp_pLayerBeforeOutputs[uWeightIndex];
		double dCurrentValue = d_pThisNeuronWeights[uWeightIndex];
		double dChangedValue = dCurrentValue - dChange;
		PRINT_DEBUG_INFO("GPU: Test 0 , Neuron %d , uWeightIndex %d: dCurrentValue = %f , dChange %f , dChangedValue %f\n",threadIdx.x,uWeightIndex,dCurrentValue,dChange,dChangedValue);
		d_pThisNeuronWeights[uWeightIndex] = dChangedValue;
	}
	
	PRINT_DEBUG_INFO("GPU: Test 0 , Neuron %d , Bias : dChange %f\n",threadIdx.x,dErrorMultDerivativeMultEta);
	d_pThisNeuronWeights[p_iNumOutputsLayerBefore] = d_pThisNeuronWeights[p_iNumOutputsLayerBefore] - dErrorMultDerivativeMultEta;
}

extern "C" void updateWeightsInTrainingCUDA(const real_gpu *dp_pThisLayerError,const real_gpu *dp_pDerivativeOfLastOutput,const real_gpu *dp_pLayerBeforeOutputs,real_gpu p_dEta,int p_iThisLayerNeuronCount,int p_iNumOutputsLayerBefore,real_gpu *dp_pThisLayerWeights)
{
	updateWeightsInTrainingKernel <<<1,p_iThisLayerNeuronCount>>> (dp_pThisLayerError,dp_pDerivativeOfLastOutput,dp_pLayerBeforeOutputs,p_dEta,p_iNumOutputsLayerBefore,dp_pThisLayerWeights);
}
