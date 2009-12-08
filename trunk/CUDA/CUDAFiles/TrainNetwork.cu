#include "..\Global\stdafx.h"

__constant__ int iTestIndices[iMaxNumberOfTrainedElements];

__global__ void executeLayerKernel(const real_gpu *dp_pLayerInput,const real_gpu *dp_pWeights,real_gpu *dp_pLayerOutput,real_gpu *dp_pDerivativeOfLastOutput
								   ,int p_iNumInputNeurons, Neuron::NeuronType p_eNeuronType,int p_iOutputNeuronCount,bool p_bInTraining)
{
	extern __shared__ real_gpu s_InputNeurons[];
	real_gpu* s_InputWeights = &s_InputNeurons[p_iNumInputNeurons];

	int iTestIndex;
	if(p_bInTraining)
		iTestIndex = blockIdx.x; //////iTestIndices[blockIdx.x];
	else
		iTestIndex = blockIdx.x;

	int iNumInputNeuronsAligned = ALIGN_UP(p_iNumInputNeurons, HALF_WARP);
	int iNumOutputNeuronsAligned = ALIGN_UP(blockDim.x, HALF_WARP);
	
	const real_gpu *d_LayerInputThisTest = dp_pLayerInput + iTestIndex*iNumInputNeuronsAligned;
	int iMoveWeightsForThisTest = threadIdx.x*p_iNumInputNeurons;
	const real_gpu *d_WeightsThisTest = dp_pWeights + iMoveWeightsForThisTest;
	real_gpu *d_pLayerOutputThisTest = dp_pLayerOutput + iTestIndex*iNumOutputNeuronsAligned + threadIdx.x;
	real_gpu *d_pDerivativeOfLastOutputThisTest = dp_pDerivativeOfLastOutput + iTestIndex*iNumOutputNeuronsAligned + threadIdx.x;

	// first, we copy d_LayerInputThisTest to s_InputNeurons
	for(int iInputIndex = threadIdx.x;iInputIndex < p_iNumInputNeurons; iInputIndex+=blockDim.x)
	{
		s_InputNeurons[iInputIndex] = d_LayerInputThisTest[iInputIndex];
	}

	// we have to make sure that all data was written to shared memory
	__syncthreads();

	real_gpu dResult = 0.0f;
	
	//if(threadIdx.x == 1 && blockIdx.x == 1)
	//{
	//	printf("INPUT %d | WEIGHTS %d | OUTPUT %d\n",d_LayerInputThisTest - dp_pLayerInput,d_WeightsThisTest - dp_pWeights,d_pLayerOutputThisTest - dp_pLayerOutput);
	//}

	int iNumOfWeights = p_iNumInputNeurons * p_iOutputNeuronCount;
	int iNumOfWeightsAligned = ALIGN_UP(iNumOfWeights,blockDim.x);
	for(int iWeightIndex = threadIdx.x, iWeightIndexBase = 0 ; iWeightIndex < iNumOfWeightsAligned ; iWeightIndex += blockDim.x, iWeightIndexBase += blockDim.x)
	{
		/*if(threadIdx.x == 0)
		{
			PRINT_DEBUG_INFO("GPU: NEW BATCH!!!!!!!!! iWeightIndexBase = %d , blockDim.x = %d\n",iWeightIndexBase,blockDim.x);
		}*/

		// first, we copy d_WeightsThisTest to s_InputWeights (it is only a part of weights)
		if(iWeightIndex < iNumOfWeights) 
		{ // JRTODO - without this 'if' it was 5% faster!
			s_InputWeights[threadIdx.x] = dp_pWeights[iWeightIndex];
		}

		__syncthreads(); // We make sure that all data was written to shared memory

		int iFirstElementInThisBatch = iMoveWeightsForThisTest - iWeightIndexBase;
		int iLastElementInThisBatch = iFirstElementInThisBatch + p_iNumInputNeurons;

		// Not all threads are used in calulations
		//PRINT_DEBUG_INFO("GPU: Test %d , Neuron %d : iFirstElementInThisBatch %d , iLastElementInThisBatch %d , T1  = [%d] , T2 = [%d] , T3 = [%d]\n",blockIdx.x,threadIdx.x,iFirstElementInThisBatch,iLastElementInThisBatch,(threadIdx.x < p_iOutputNeuronCount),(iLastElementInThisBatch >= 0),(iFirstElementInThisBatch < 0 || iFirstElementInThisBatch < blockDim.x));
		if(threadIdx.x < p_iOutputNeuronCount && iLastElementInThisBatch >= 0 && (iFirstElementInThisBatch < 0 || iFirstElementInThisBatch < blockDim.x))
		{
			int iFirstWeightIndex = max(0,-iFirstElementInThisBatch);
			int iLastWeightIndex = min(p_iNumInputNeurons,p_iNumInputNeurons - (iLastElementInThisBatch - blockDim.x));
			for(int iWeightIndex = iFirstWeightIndex;iWeightIndex < iLastWeightIndex; ++iWeightIndex)
			{
				int iWeightIndexHere = iWeightIndex - iWeightIndexBase + iMoveWeightsForThisTest;
				//PRINT_DEBUG_INFO("GPU: Test %d , Neuron %d , iWeightIndex %d : d_LayerInputThisTest %f , d_WeightsThisTest %f , iWeightIndexHere %d, val[%d] %f , MULT %f\n",blockIdx.x,threadIdx.x,iWeightIndex,d_LayerInputThisTest[iWeightIndex],d_WeightsThisTest[iWeightIndex],iWeightIndexHere,iWeightIndexHere,s_InputWeights[iWeightIndexHere],d_LayerInputThisTest[iWeightIndex] * d_WeightsThisTest[iWeightIndex]);

				dResult += s_InputNeurons[iWeightIndex] * s_InputWeights[iWeightIndexHere];
			}
		}

		__syncthreads(); // We make sure that all data was read by all threads
	}

	if(threadIdx.x <= p_iOutputNeuronCount)
	{
		double dDerivativeOfLastOutput = 0.0f;

		//PRINT_DEBUG_INFO("GPU: Test %d , Neuron %d : dResult before output function %f\n",blockIdx.x,threadIdx.x,dResult);

		switch(p_eNeuronType)
		{		
			case Neuron::NT_LINEAR: 
				dDerivativeOfLastOutput = 1.0f;
				break;	// Do nothing
			case Neuron::NT_SIGMOID:
				double dExp = __expf(-dResult);
				dResult = 1.0f / (1.0f + dExp);
				dDerivativeOfLastOutput = dExp / __powf(1.0f + dExp,2);
				break;
		}
		
		if(threadIdx.x == p_iOutputNeuronCount)
			dResult = 1.0f; // bias
			
		//PRINT_DEBUG_INFO("XXXXXXXXXXXXXXXXXXXXXXXXXXXXGPU: Test %d , Neuron %d : %d\n",blockIdx.x,threadIdx.x,blockIdx.x*iNumOutputNeuronsAligned + threadIdx.x);
		*d_pLayerOutputThisTest = dResult;
		
		// We only need derivative of last output if we are in training!
		if(dp_pDerivativeOfLastOutput != NULL)
			*d_pDerivativeOfLastOutputThisTest = dDerivativeOfLastOutput;

		//PRINT_DEBUG_INFO("GPU: Test %d , Neuron %d : first d_LayerInputThisTest %f , first d_WeightsThisTest %f , dResult %f , dDerivativeOfLastOutput %f\n",blockIdx.x,threadIdx.x,d_LayerInputThisTest[0],d_WeightsThisTest[0],dResult,dDerivativeOfLastOutput);
	}
}

extern "C" void executeLayerCUDA(const real_gpu *dp_pLayerInput,const real_gpu *dp_pWeights,real_gpu *dp_pLayerOutput,real_gpu *dp_pDerivativeOfLastOutput
								 ,int p_iTestCount,int p_iOutputNeuronCount,int p_iNumInputNeurons,Neuron::NeuronType p_eNeuronType,const int *p_pVecTestIndices)
{
	// blockDim.x should be a multiple of 16 (half warp). We will be able to retrieve global data using coalescing
	int iBlockDimUpdated = ALIGN_UP(p_iOutputNeuronCount+1,HALF_WARP);
	int iSharedMemorySize = p_iNumInputNeurons * sizeof(real_gpu); // memory for input
	iSharedMemorySize += iBlockDimUpdated * sizeof(real_gpu); // memory for weights

	// If p_pVecTestIndices!=NULL , then we use constant memory to set test indices for the kernel
	if(p_pVecTestIndices!=NULL)
	{
		cudaMemcpyToSymbol("iTestIndices",p_pVecTestIndices,p_iTestCount*sizeof(int),0);
	}
	executeLayerKernel <<<p_iTestCount,iBlockDimUpdated,iSharedMemorySize>>> (dp_pLayerInput,dp_pWeights,dp_pLayerOutput,dp_pDerivativeOfLastOutput,p_iNumInputNeurons,p_eNeuronType,p_iOutputNeuronCount,(p_pVecTestIndices!=NULL));
}


__global__ void calculateErrorInLastLayerKernel(const real_gpu *dp_pCorrectOutput,const real_gpu *dp_pNetworkOutput,real_gpu *dp_pErrors)
{
	dp_pErrors[threadIdx.x] = dp_pNetworkOutput[threadIdx.x] - dp_pCorrectOutput[threadIdx.x];
	PRINT_DEBUG_INFO("GPU: Test in batch nr 0 , Output %d : Network = %f , Correct  = %f , Error = %f\n",threadIdx.x,dp_pNetworkOutput[threadIdx.x],dp_pCorrectOutput[threadIdx.x],dp_pErrors[threadIdx.x]);
}

extern "C" void calculateErrorInLastLayerCUDA(const real_gpu *dp_pCorrectOutput,const real_gpu *dp_pNetworkOutput,real_gpu *dp_pErrors,int p_iOutputNeuronCount)
{
	calculateErrorInLastLayerKernel <<<1,p_iOutputNeuronCount>>> (dp_pCorrectOutput,dp_pNetworkOutput,dp_pErrors);
}


__global__ void calculateErrorInNotLastLayerKernel(const real_gpu *dp_pNextLayerWeights,const real_gpu *dp_pNextLayerError,real_gpu *dp_pThisLayerError,int p_iNextLayerNeuronCount)
{
	real_gpu dError = 0.0f;
	
	for(int iWeightIndex = 0;iWeightIndex < p_iNextLayerNeuronCount; ++iWeightIndex)
	{
		PRINT_DEBUG_INFO("GPU: Test index 0 , Neuron index %d , Weight index %d : dp_pNextLayerWeights [%d] = %f , dp_pNextLayerError[%d] = %f , MULT = %f\n"
			,threadIdx.x,iWeightIndex,iWeightIndex*(blockDim.x + 1) + threadIdx.x,dp_pNextLayerWeights[iWeightIndex*(blockDim.x + 1) + threadIdx.x],iWeightIndex
			,dp_pNextLayerError[iWeightIndex],dp_pNextLayerWeights[iWeightIndex*(blockDim.x + 1) + threadIdx.x] * dp_pNextLayerError[iWeightIndex]);
		dError += dp_pNextLayerWeights[iWeightIndex*(blockDim.x + 1) + threadIdx.x] * dp_pNextLayerError[iWeightIndex];
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
