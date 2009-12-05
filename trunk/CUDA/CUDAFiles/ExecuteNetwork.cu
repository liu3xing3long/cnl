#include "..\Global\stdafx.h"

__global__ void executeLayerKernel(const real_gpu *dp_pLayerInput,const real_gpu *dp_pWeights,real_gpu *dp_pLayerOutput,real_gpu *dp_pDerivativeOfLastOutput,int p_iNumInputNeurons, Neuron::NeuronType p_eNeuronType)
{
	int iNumInputNeuronsAligned = p_iNumInputNeurons;
	ALIGN_UP(iNumInputNeuronsAligned, HALF_WARP);
	int iNumOutputNeuronsAligned = blockDim.x;
	ALIGN_UP(iNumOutputNeuronsAligned, HALF_WARP);
	
	const real_gpu *d_LayerInputThisTest = dp_pLayerInput + blockIdx.x*iNumInputNeuronsAligned;
	const real_gpu *d_WeightsThisTest = dp_pWeights + threadIdx.x*p_iNumInputNeurons;
	real_gpu *d_pLayerOutputThisTest = dp_pLayerOutput + blockIdx.x*iNumOutputNeuronsAligned + threadIdx.x;
	real_gpu *d_pDerivativeOfLastOutputThisTest = dp_pDerivativeOfLastOutput + blockIdx.x*iNumOutputNeuronsAligned + threadIdx.x;
	
	real_gpu dResult = 0.0f;
	
	/*if(threadIdx.x == 1 && blockIdx.x == 1)
	{
		printf("INPUT %d | WEIGHTS %d | OUTPUT %d\n",d_LayerInputThisTest - dp_pLayerInput,d_WeightsThisTest - dp_pWeights,d_pLayerOutputThisTest - dp_pLayerOutput);
	}*/	
	
	for(int iWeightIndex = 0;iWeightIndex < p_iNumInputNeurons; ++iWeightIndex)
	{
		PRINT_DEBUG_INFO("GPU: Test %d , Neuron %d , iWeightIndex %d : d_LayerInputThisTest %f , d_WeightsThisTest %f , MULT %f\n",blockIdx.x,threadIdx.x,iWeightIndex,d_LayerInputThisTest[iWeightIndex],d_WeightsThisTest[iWeightIndex],d_LayerInputThisTest[iWeightIndex] * d_WeightsThisTest[iWeightIndex]);
		dResult += d_LayerInputThisTest[iWeightIndex] * d_WeightsThisTest[iWeightIndex];
	}
	
	double dDerivativeOfLastOutput = 0.0f;

	PRINT_DEBUG_INFO("GPU: Test %d , Neuron %d : dResult before output function %f\n",blockIdx.x,threadIdx.x,dResult);

	switch(p_eNeuronType)
	{		
		case Neuron::NT_LINEAR: 
			dDerivativeOfLastOutput = 1.0f;
			break;	// Do nothing
		case Neuron::NT_SIGMOID: 
			double dExp = exp(-dResult);
			dResult = 1.0 / (1.0 + dExp);
			dDerivativeOfLastOutput = dExp / pow(1.0 + dExp,2);
			break;	
	}
	
	if(threadIdx.x == blockDim.x - 1)
		dResult = 1.0f; /* bias */
		
	*d_pLayerOutputThisTest = dResult;
	
	// We only need derivative of last output if we are in training!
	if(dp_pDerivativeOfLastOutput != NULL)
		*d_pDerivativeOfLastOutputThisTest = dDerivativeOfLastOutput;

	PRINT_DEBUG_INFO("GPU: Test %d , Neuron %d : first d_LayerInputThisTest %f , first d_WeightsThisTest %f , dResult %f , dDerivativeOfLastOutput %f\n",blockIdx.x,threadIdx.x,d_LayerInputThisTest[0],d_WeightsThisTest[0],dResult,dDerivativeOfLastOutput);
}

extern "C" void executeLayerCUDA(const real_gpu *dp_pLayerInput,const real_gpu *dp_pWeights,real_gpu *dp_pLayerOutput,real_gpu *dp_pDerivativeOfLastOutput,int p_iTestCount,int p_iOutputNeuronCount,int p_iNumInputNeurons,Neuron::NeuronType p_eNeuronType)
{
	//int iSharedMem = 
	executeLayerKernel <<<p_iTestCount,p_iOutputNeuronCount+1>>> (dp_pLayerInput,dp_pWeights,dp_pLayerOutput,dp_pDerivativeOfLastOutput,p_iNumInputNeurons,p_eNeuronType);
}
