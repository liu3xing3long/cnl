#include "..\Global\stdafx.h"

__constant__ int iTestIndices[iMaxNumberOfTrainedElements];

// JRTODO - dylemat - czy zawsze uzywac iTestIndices przy czytaniu i zapisywaniu, czy tylko na wejsciu pierwszego layera?..
// JRTODO - zmien mnozenie integerow na specjalna funkcje mnozaca najnizsze 24 bity

//void *CUDATools::m_allocatedMemoryAddress[iMemoryElementsSize];
//int CUDATools::m_allocatedMemorySize[iMemoryElementsSize];

const int iNumTestsInOneExecuteBlock = 4;

__global__ void executeLayerKernel(const real_gpu *dp_pLayerInput,const real_gpu *dp_pWeights,real_gpu *dp_pLayerOutput,real_gpu *dp_pDerivativeOfLastOutput,int p_iNumInputNeurons
								   ,int p_iNumInputNeuronsAligned, Neuron::NeuronType p_eNeuronType,int p_iOutputNeuronCount,bool p_bInTraining,int p_iHowMuchMemoryForWeights,int p_iTestCount)
{
	extern __shared__ real_gpu s_InputNeurons[];
	real_gpu* s_InputWeights = &s_InputNeurons[p_iNumInputNeurons*iNumTestsInOneExecuteBlock];

	int iNumTestsToCalculate = min(iNumTestsInOneExecuteBlock,p_iTestCount - iNumTestsInOneExecuteBlock * blockIdx.x+1);

	__shared__ int iTestIndex[iNumTestsInOneExecuteBlock];
	// JRTODO do it faster
	if(p_bInTraining)
	{
		for(int iNumTestExecute = 0;iNumTestExecute < iNumTestsToCalculate; ++iNumTestExecute)
			iTestIndex[iNumTestExecute] = iTestIndices[iNumTestsInOneExecuteBlock*blockIdx.x+iNumTestExecute];
	}
	else
	{
		for(int iNumTestExecute = 0;iNumTestExecute < iNumTestsToCalculate; ++iNumTestExecute)
			iTestIndex[iNumTestExecute] = iNumTestsInOneExecuteBlock*blockIdx.x+iNumTestExecute;
	}
	
	const real_gpu *d_LayerInputThisTest[iNumTestsInOneExecuteBlock];
	for(int iNumTestExecute = 0;iNumTestExecute < iNumTestsToCalculate; ++iNumTestExecute)
		d_LayerInputThisTest[iNumTestExecute] = dp_pLayerInput + iTestIndex[iNumTestExecute]*p_iNumInputNeuronsAligned;
	
	int iMoveWeightsForThisTest = threadIdx.x*p_iNumInputNeurons;
	
	real_gpu *d_pLayerOutputThisTest[iNumTestsInOneExecuteBlock];
	for(int iNumTestExecute = 0;iNumTestExecute < iNumTestsToCalculate; ++iNumTestExecute)
		d_pLayerOutputThisTest[iNumTestExecute] = dp_pLayerOutput + (iNumTestsInOneExecuteBlock*blockIdx.x+iNumTestExecute)*blockDim.x + threadIdx.x;

	real_gpu *d_pDerivativeOfLastOutputThisTest[iNumTestsInOneExecuteBlock];
	for(int iNumTestExecute = 0;iNumTestExecute < iNumTestsToCalculate; ++iNumTestExecute)
		d_pDerivativeOfLastOutputThisTest[iNumTestExecute] = dp_pDerivativeOfLastOutput + (iNumTestsInOneExecuteBlock*blockIdx.x+iNumTestExecute)*blockDim.x + threadIdx.x;

#ifdef PRINT_DEBUG
	const real_gpu *d_WeightsThisTest = dp_pWeights + iMoveWeightsForThisTest;
#endif

	// first, we copy d_LayerInputThisTest to s_InputNeurons
	for(int iInputIndex = threadIdx.x;iInputIndex < p_iNumInputNeurons; iInputIndex+=blockDim.x)
	{
		for(int iNumTestExecute = 0;iNumTestExecute < iNumTestsToCalculate; ++iNumTestExecute)
			s_InputNeurons[p_iNumInputNeurons*iNumTestExecute+iInputIndex] = d_LayerInputThisTest[iNumTestExecute][iInputIndex];
		PRINT_MEMORY_INFO(dp_pLayerInput,&d_LayerInputThisTest[iInputIndex]);
	} 

	// we have to make sure that all data was written to shared memory
	__syncthreads();

	real_gpu dResult[iNumTestsInOneExecuteBlock];
	for(int iNumTestExecute = 0;iNumTestExecute < iNumTestsToCalculate; ++iNumTestExecute)
		dResult[iNumTestExecute] = 0.0f;
	
	//if(threadIdx.x == 1 && blockIdx.x == 1)
	//{
	//	PRINT_DEBUG_INFO("BX %d TX %d | INPUT %d | WEIGHTS %d | OUTPUT %d\n",blockIdx.x,threadIdx.x,d_LayerInputThisTest - dp_pLayerInput,d_WeightsThisTest - dp_pWeights,d_pLayerOutputThisTest - dp_pLayerOutput);
	//}

	int iNumOfWeights = p_iNumInputNeurons * p_iOutputNeuronCount;
	int iNumOfWeightsAligned = ALIGN_UP(iNumOfWeights,blockDim.x);
	for(int iWeightIndex = threadIdx.x, iWeightIndexBase = 0 ; iWeightIndex < iNumOfWeightsAligned ; iWeightIndex += p_iHowMuchMemoryForWeights, iWeightIndexBase += p_iHowMuchMemoryForWeights)
	{
		/*if(threadIdx.x == 0)
		{
			PRINT_DEBUG_INFO("GPU: NEW BATCH!!!!!!!!! iWeightIndexBase = %d , blockDim.x = %d\n",iWeightIndexBase,blockDim.x);
		}*/

		// first, we copy d_WeightsThisTest to s_InputWeights (it is only a part of weights).
		// We don't have to use 'if(iWeightIndex < iNumOfWeights)', because memory for weights was already padded (now it's about 5% faster)
		for(int iCopiedWeightIndex = 0;iCopiedWeightIndex<p_iHowMuchMemoryForWeights;iCopiedWeightIndex += blockDim.x)
		{
			s_InputWeights[iCopiedWeightIndex + threadIdx.x] = dp_pWeights[iCopiedWeightIndex + iWeightIndex];
			PRINT_MEMORY_INFO(dp_pWeights,&dp_pWeights[iCopiedWeightIndex + iWeightIndex]);
		}

		__syncthreads(); // We make sure that all data was written to shared memory

		int iFirstElementInThisBatch = iMoveWeightsForThisTest - iWeightIndexBase;
		int iLastElementInThisBatch = iFirstElementInThisBatch + p_iNumInputNeurons;

		// Not all threads are used in calulations
		//PRINT_DEBUG_INFO("GPU: Test %d , Neuron %d : iFirstElementInThisBatch %d , iLastElementInThisBatch %d , T1  = [%d] , T2 = [%d] , T3 = [%d]\n",blockIdx.x,threadIdx.x,iFirstElementInThisBatch,iLastElementInThisBatch,(threadIdx.x < p_iOutputNeuronCount),(iLastElementInThisBatch >= 0),(iFirstElementInThisBatch < 0 || iFirstElementInThisBatch < blockDim.x));
		if(threadIdx.x < p_iOutputNeuronCount && iLastElementInThisBatch >= 0 && (iFirstElementInThisBatch < 0 || iFirstElementInThisBatch < p_iHowMuchMemoryForWeights))
		{
			int iFirstWeightIndex = max(0,-iFirstElementInThisBatch);
			int iLastWeightIndex = min(p_iNumInputNeurons,p_iNumInputNeurons - (iLastElementInThisBatch - p_iHowMuchMemoryForWeights));
			for(int iWeightIndexToAdd = iFirstWeightIndex;iWeightIndexToAdd < iLastWeightIndex; ++iWeightIndexToAdd)
			{
				int iWeightIndexHere = iWeightIndexToAdd - iWeightIndexBase + iMoveWeightsForThisTest;
				//PRINT_DEBUG_INFO("GPU: Test %d , Neuron %d , iWeightIndexToAdd %d : d_LayerInputThisTest %f , d_WeightsThisTest %f , iWeightIndexHere %d, val[%d] %f , MULT %f\n",blockIdx.x,threadIdx.x,iWeightIndexToAdd,d_LayerInputThisTest[iWeightIndexToAdd],d_WeightsThisTest[iWeightIndexToAdd],iWeightIndexHere,iWeightIndexHere,s_InputWeights[iWeightIndexHere],d_LayerInputThisTest[iWeightIndexToAdd] * d_WeightsThisTest[iWeightIndexToAdd]);

				for(int iNumTestExecute = 0;iNumTestExecute < iNumTestsToCalculate; ++iNumTestExecute)
					dResult[iNumTestExecute] += s_InputNeurons[p_iNumInputNeurons*iNumTestExecute+iWeightIndexToAdd] * s_InputWeights[iWeightIndexHere];
			}
		}

		__syncthreads(); // We make sure that all data was read by all threads
	}

	if(threadIdx.x <= p_iOutputNeuronCount)
	{
		real_gpu dDerivativeOfLastOutput[iNumTestsInOneExecuteBlock];
		for(int iNumTestExecute = 0;iNumTestExecute < iNumTestsToCalculate; ++iNumTestExecute)
			dDerivativeOfLastOutput[iNumTestExecute] = 0.0f;

		//PRINT_DEBUG_INFO("GPU: Test %d , Neuron %d : dResult before output function %f\n",blockIdx.x,threadIdx.x,dResult);

		switch(p_eNeuronType)
		{		
			case Neuron::NT_LINEAR: 
				for(int iNumTestExecute = 0;iNumTestExecute < iNumTestsToCalculate; ++iNumTestExecute)
					dDerivativeOfLastOutput[iNumTestExecute] = 1.0f;
				break;	
			case Neuron::NT_SIGMOID:
				for(int iNumTestExecute = 0;iNumTestExecute < iNumTestsToCalculate; ++iNumTestExecute)
				{
					real_gpu dExp = __expf(-dResult[iNumTestExecute]);
					dResult[iNumTestExecute] = 1.0f / (1.0f + dExp);
					dDerivativeOfLastOutput[iNumTestExecute] = dExp / __powf(1.0f + dExp,2);
				}
				break;
		}
		
		if(threadIdx.x == p_iOutputNeuronCount)
		{
			for(int iNumTestExecute = 0;iNumTestExecute < iNumTestsToCalculate; ++iNumTestExecute)
				dResult[iNumTestExecute] = 1.0f; // bias
		}

		//PRINT_DEBUG_INFO("XXXXXXXXXXXXXXXXXXXXXXXXXXXXGPU: Test %d , Neuron %d : %d\n",blockIdx.x,threadIdx.x,blockIdx.x*iNumOutputNeuronsAligned + threadIdx.x);
		for(int iNumTestExecute = 0;iNumTestExecute < iNumTestsToCalculate; ++iNumTestExecute)
			*(d_pLayerOutputThisTest[iNumTestExecute]) = dResult[iNumTestExecute];

		PRINT_MEMORY_INFO(dp_pLayerOutput,d_pLayerOutputThisTest);

		// We only need derivative of last output if we are in training!
		if(dp_pDerivativeOfLastOutput != NULL)
		{
			for(int iNumTestExecute = 0;iNumTestExecute < iNumTestsToCalculate; ++iNumTestExecute)
				*(d_pDerivativeOfLastOutputThisTest[iNumTestExecute]) = dDerivativeOfLastOutput[iNumTestExecute];

			PRINT_MEMORY_INFO(dp_pDerivativeOfLastOutput,d_pDerivativeOfLastOutputThisTest);
		}

		//PRINT_DEBUG_INFO("GPU: Test %d , Neuron %d : first d_LayerInputThisTest %f , first d_WeightsThisTest %f , dResult %f , dDerivativeOfLastOutput %f\n",blockIdx.x,threadIdx.x,d_LayerInputThisTest[0],d_WeightsThisTest[0],dResult,dDerivativeOfLastOutput);
	}
}

extern "C" void executeLayerCUDA(const real_gpu *dp_pLayerInput,const real_gpu *dp_pWeights,real_gpu *dp_pLayerOutput,real_gpu *dp_pDerivativeOfLastOutput
								 ,int p_iTestCount,int p_iOutputNeuronCount,int p_iNumInputNeurons,Neuron::NeuronType p_eNeuronType,const int *p_pVecTestIndices)
{
	// blockDim.x should be a multiple of 16 (half warp). We will be able to retrieve global data using coalescing
	int iBlockDimUpdated = ALIGN_UP(p_iOutputNeuronCount+1,HALF_WARP);
	int iSharedMemorySize = iNumTestsInOneExecuteBlock * p_iNumInputNeurons * sizeof(real_gpu); // memory for input

	int iNumOfWeights = p_iNumInputNeurons * p_iOutputNeuronCount;
	int iNumOfWeightsAligned = ALIGN_UP(iNumOfWeights,iBlockDimUpdated);
	int iMaxNumberOfSimulatenousBlocks = 512 / iBlockDimUpdated + ((512 % iBlockDimUpdated) != 0);

	int iMaxMemPerBlock = 0;
	/*if(iMaxNumberOfSimulatenousBlocks > 7)
	{ // We give some more shared memory...
		iMaxMemPerBlock = max(0,((iMaxNumberOfSharedMemoryElementsForWeights) / iMaxNumberOfSimulatenousBlocks - p_iNumInputNeurons));
	}
	else
	{*/
		iMaxMemPerBlock = max(0,(iMaxNumberOfSharedMemoryElementsForWeights / iMaxNumberOfSimulatenousBlocks - iNumTestsInOneExecuteBlock * p_iNumInputNeurons));
	//}

	int iHowMuchMemoryForWeights = (min(iNumOfWeightsAligned,max(512,iMaxMemPerBlock)) / iBlockDimUpdated) * iBlockDimUpdated;

	iSharedMemorySize += iHowMuchMemoryForWeights * sizeof(real_gpu); // memory for weights

	// If p_pVecTestIndices!=NULL , then we use constant memory to set test indices for the kernel
	if(p_pVecTestIndices!=NULL)
	{
		cudaMemcpyToSymbol("iTestIndices",p_pVecTestIndices,p_iTestCount*sizeof(int),0);
	}

	int iNumInputNeuronsAligned = ALIGN_UP(p_iNumInputNeurons, HALF_WARP);

	executeLayerKernel <<<(p_iTestCount+iNumTestsInOneExecuteBlock-1)/iNumTestsInOneExecuteBlock,iBlockDimUpdated,iSharedMemorySize>>> (dp_pLayerInput,dp_pWeights,dp_pLayerOutput,dp_pDerivativeOfLastOutput,p_iNumInputNeurons
		,iNumInputNeuronsAligned,p_eNeuronType,p_iOutputNeuronCount,(p_pVecTestIndices!=NULL),iHowMuchMemoryForWeights,p_iTestCount);
}


__global__ void calculateErrorInLastLayerKernel(const real_gpu *dp_pCorrectOutput,const real_gpu *dp_pNetworkOutput,real_gpu *dp_pErrors,int p_iSpaceBetweenTestsInOutput)
{
	int iElementIndexCorrectOutput = p_iSpaceBetweenTestsInOutput * iTestIndices[blockIdx.x] + threadIdx.x;
	int iElementIndex = p_iSpaceBetweenTestsInOutput * blockIdx.x + threadIdx.x;
	dp_pErrors[iElementIndex] = dp_pNetworkOutput[iElementIndex] - dp_pCorrectOutput[iElementIndexCorrectOutput];
	PRINT_DEBUG_INFO("GPU: Test in batch nr %d (test %d) , Output %d (iElementIndex %d) : Network = %f , Correct  = %f , Error = %f\n",blockIdx.x,iTestIndices[blockIdx.x],threadIdx.x
		,iElementIndex,dp_pNetworkOutput[iElementIndex],dp_pCorrectOutput[iElementIndexCorrectOutput],dp_pErrors[iElementIndex]);
	PRINT_MEMORY_INFO(dp_pErrors,&dp_pErrors[iElementIndex]);
	PRINT_MEMORY_INFO(dp_pNetworkOutput,&dp_pNetworkOutput[iElementIndex]);
	PRINT_MEMORY_INFO(dp_pCorrectOutput,&dp_pCorrectOutput[iElementIndexCorrectOutput]);
}

extern "C" void calculateErrorInLastLayerCUDA(const real_gpu *dp_pCorrectOutput,const real_gpu *dp_pNetworkOutput,real_gpu *dp_pErrors,int p_iOutputNeuronCount,int p_iNumTestsInBatch,int p_iSpaceBetweenTestsInOutput)
{
	calculateErrorInLastLayerKernel <<<p_iNumTestsInBatch,p_iOutputNeuronCount>>> (dp_pCorrectOutput,dp_pNetworkOutput,dp_pErrors,p_iSpaceBetweenTestsInOutput);
}


__global__ void calculateErrorInNotLastLayerKernel(const real_gpu *dp_pNextLayerWeights,const real_gpu *dp_pNextLayerError,real_gpu *dp_pThisLayerError,int p_iNextLayerNeuronCount
												   ,int p_iNextLayerNeuronCountAligned,int p_iThisLayerNeuronCount,int p_iNumTestsInBatch)
{
	extern __shared__ real_gpu s_NextLayerErrorThisTest[];
	real_gpu* s_NextLayerErrorThisTest2 = &s_NextLayerErrorThisTest[p_iNextLayerNeuronCount];
	real_gpu* s_NextLayerWeights = &s_NextLayerErrorThisTest2[p_iNextLayerNeuronCount];
	real_gpu dError = 0.0f;
	real_gpu dError2 = 0.0f;
	int iNextLayerWeightsForOneNeuron = p_iThisLayerNeuronCount + 1;
	const real_gpu *d_pNextLayerErrorThisTest = dp_pNextLayerError + p_iNextLayerNeuronCountAligned * (2*blockIdx.x);
	const real_gpu *d_pNextLayerErrorThisTest2 = dp_pNextLayerError + p_iNextLayerNeuronCountAligned * (2*blockIdx.x+1);

	// Copying error data from global to shared memory
	for(int iErrorIndex = threadIdx.x;iErrorIndex < p_iNextLayerNeuronCount; iErrorIndex += blockDim.x)
	{
		s_NextLayerErrorThisTest[iErrorIndex] = d_pNextLayerErrorThisTest[iErrorIndex];
		s_NextLayerErrorThisTest2[iErrorIndex] = d_pNextLayerErrorThisTest2[iErrorIndex];
		PRINT_MEMORY_INFO(dp_pNextLayerError,&d_pNextLayerErrorThisTest[iErrorIndex]);
	}

	__syncthreads();

	// I can't check thread index, because later I use __syncthreads() ...
	//if(threadIdx.x < p_iThisLayerNeuronCount)
	{
		for(int iWeightIndex = 0;iWeightIndex < p_iNextLayerNeuronCount; ++iWeightIndex)
		{
			PRINT_DEBUG_INFO("GPU: Test index %d , Neuron index %d , Weight index %d : dp_pNextLayerWeights [%d] = %f , dp_pNextLayerError[%d] = %f , MULT = %f\n"
				,blockIdx.x,threadIdx.x,iWeightIndex,iWeightIndex*iNextLayerWeightsForOneNeuron + threadIdx.x,dp_pNextLayerWeights[iWeightIndex*iNextLayerWeightsForOneNeuron + threadIdx.x],iWeightIndex
				,dp_pNextLayerError[iWeightIndex],dp_pNextLayerWeights[iWeightIndex*iNextLayerWeightsForOneNeuron + threadIdx.x] * dp_pNextLayerError[iWeightIndex]);
				
			// we load weights twice - in case the first loaded weight position is not divisible by HALF_WARP
			int iWeightFirstAddress = iWeightIndex*iNextLayerWeightsForOneNeuron;
			int iFirstAddressToLoad = (iWeightFirstAddress / HALF_WARP) * HALF_WARP;
			s_NextLayerWeights[threadIdx.x] = dp_pNextLayerWeights[iFirstAddressToLoad + threadIdx.x];
			s_NextLayerWeights[blockDim.x + threadIdx.x] = dp_pNextLayerWeights[iFirstAddressToLoad + blockDim.x + threadIdx.x];
			__syncthreads();

			int iWeightIndexInSharedMemory = iWeightFirstAddress - iFirstAddressToLoad + threadIdx.x;
			dError += s_NextLayerWeights[iWeightIndexInSharedMemory] * s_NextLayerErrorThisTest[iWeightIndex];
			dError2 += s_NextLayerWeights[iWeightIndexInSharedMemory] * s_NextLayerErrorThisTest2[iWeightIndex];
			PRINT_MEMORY_INFO(dp_pNextLayerWeights,&dp_pNextLayerWeights[iWeightIndex*iNextLayerWeightsForOneNeuron + threadIdx.x]);

			__syncthreads();
		}
		
		if(threadIdx.x < p_iThisLayerNeuronCount)
		{
			dp_pThisLayerError[blockDim.x*(2*blockIdx.x) + threadIdx.x] = dError;

			if(2*blockIdx.x != p_iNumTestsInBatch-1)
				dp_pThisLayerError[blockDim.x*(2*blockIdx.x+1) + threadIdx.x] = dError2;

			PRINT_MEMORY_INFO(dp_pThisLayerError,&dp_pThisLayerError[blockDim.x*blockIdx.x + threadIdx.x]);
			PRINT_DEBUG_INFO("GPU: Test index %d , Neuron index %d : Error %f\n",blockIdx.x,threadIdx.x,dError);
		}
	}
}

extern "C" void calculateErrorInNotLastLayerCUDA(const real_gpu *dp_pNextLayerWeights,const real_gpu *dp_pNextLayerError,real_gpu *dp_pThisLayerError,int p_iThisLayerNeuronCount,int p_iNextLayerNeuronCount,int p_iNumTestsInBatch)
{
	int iElementsAllocatedForOneTestInNextLayerAligned = ALIGN_UP(p_iNextLayerNeuronCount+1,HALF_WARP);
	int iElementsAllocatedForOneTestInThisLayerAligned = ALIGN_UP(p_iThisLayerNeuronCount+1,HALF_WARP);
	int iSharedMemorySize = 2 * p_iNextLayerNeuronCount * sizeof(real_gpu); // memory for error
	iSharedMemorySize += 2 * iElementsAllocatedForOneTestInThisLayerAligned * sizeof(real_gpu); // memory for weights

	calculateErrorInNotLastLayerKernel <<<(p_iNumTestsInBatch+1)/2,iElementsAllocatedForOneTestInThisLayerAligned,iSharedMemorySize>>> 
		(dp_pNextLayerWeights,dp_pNextLayerError,dp_pThisLayerError,p_iNextLayerNeuronCount,iElementsAllocatedForOneTestInNextLayerAligned,p_iThisLayerNeuronCount,p_iNumTestsInBatch);
}


__global__ void updateWeightsInTrainingKernel(const real_gpu *dp_pThisLayerError,const real_gpu *dp_pDerivativeOfLastOutput,const real_gpu *dp_pLayerBeforeOutputs
											  ,real_gpu p_dEta,real_gpu *dp_pThisLayerWeights,int p_iNumTestsInBatch,int iElementsAllocatedForOneTestInThisLayerAligned
											  ,int p_iElementsAllocatedForOneTestInLayerBeforeAligned,bool p_bLayerBeforeOutputsHaveSpecificIndexes,int p_iThisLayerNeuronCount)
{
	// We change: neuron blockIdx.x , weight threadIdx.x
	extern __shared__ real_gpu s_ThisLayerError[];
	real_gpu* s_DerivativeOfLastOutput = &s_ThisLayerError[2*p_iNumTestsInBatch];

	// Two first threads in each block copy global memory to shared memory
	if(threadIdx.x == 0 || threadIdx.x == 1)
	{
		for(unsigned uTestIndex = 0;uTestIndex < p_iNumTestsInBatch;++uTestIndex)
		{
			s_ThisLayerError[2*uTestIndex+threadIdx.x] = dp_pThisLayerError[iElementsAllocatedForOneTestInThisLayerAligned*uTestIndex + 2*blockIdx.x + threadIdx.x];
			s_DerivativeOfLastOutput[2*uTestIndex+threadIdx.x] = dp_pDerivativeOfLastOutput[iElementsAllocatedForOneTestInThisLayerAligned*uTestIndex + 2*blockIdx.x + threadIdx.x];
		}
	}

	__syncthreads();

	real_gpu dChange = 0.0f;
	real_gpu dChange2 = 0.0f;
	for(unsigned uTestIndex = 0;uTestIndex < p_iNumTestsInBatch;++uTestIndex)
	{
		int iTestIndexForOutputBefore = ( p_bLayerBeforeOutputsHaveSpecificIndexes ? iTestIndices[uTestIndex] : uTestIndex );
		real_gpu dLayerBeforeOutput = dp_pLayerBeforeOutputs[p_iElementsAllocatedForOneTestInLayerBeforeAligned*iTestIndexForOutputBefore + threadIdx.x];

		real_gpu dChangeThisTest = s_ThisLayerError[2*uTestIndex] * s_DerivativeOfLastOutput[2*uTestIndex] * dLayerBeforeOutput * p_dEta;
		real_gpu dChangeThisTest2 = s_ThisLayerError[2*uTestIndex+1] * s_DerivativeOfLastOutput[2*uTestIndex+1] * dLayerBeforeOutput * p_dEta;
		PRINT_DEBUG_INFO("GPU: Test %d , Neuron %d , Weight %d : dError %f , dDerivativeOfLastOutput %f , dLayerBeforeOutput %f , dChangeThisTest %f\n",uTestIndex,blockIdx.x,threadIdx.x,dError,dDerivativeOfLastOutput,dLayerBeforeOutput,dChangeThisTest);
		dChange += dChangeThisTest;
		dChange2 += dChangeThisTest2;

		PRINT_MEMORY_INFO(dp_pThisLayerError,&dp_pThisLayerError[iElementsAllocatedForOneTestInThisLayerAligned*uTestIndex + blockIdx.x]);
		PRINT_MEMORY_INFO(dp_pDerivativeOfLastOutput,&dp_pDerivativeOfLastOutput[iElementsAllocatedForOneTestInThisLayerAligned*uTestIndex + blockIdx.x]);
		PRINT_MEMORY_INFO(dp_pLayerBeforeOutputs,&dp_pLayerBeforeOutputs[p_iElementsAllocatedForOneTestInLayerBeforeAligned*iTestIndexForOutputBefore + threadIdx.x]);
	}

	//int iTestIndexForWeights = ( p_bLayerBeforeOutputsHaveSpecificIndexes ? iTestIndices[blockIdx.x] : blockIdx.x );
	int iWeightIndex = blockDim.x*(2*blockIdx.x) + threadIdx.x;
	int iWeightIndex2 = blockDim.x*(2*blockIdx.x+1) + threadIdx.x;
	PRINT_DEBUG_INFO("GPU: Neuron %d , Weight %d (index in array %d) : Old weight %f , Change %f , New weight %f\n",blockIdx.x,threadIdx.x,iWeightIndex,dp_pThisLayerWeights[iWeightIndex],dChange,dp_pThisLayerWeights[iWeightIndex] - dChange);
	dp_pThisLayerWeights[iWeightIndex] = dp_pThisLayerWeights[iWeightIndex] - dChange;

	if(2*blockIdx.x != p_iThisLayerNeuronCount-1)
		dp_pThisLayerWeights[iWeightIndex2] = dp_pThisLayerWeights[iWeightIndex2] - dChange2;

	PRINT_MEMORY_INFO(dp_pThisLayerWeights,&dp_pThisLayerWeights[iWeightIndex]);

	/*real_gpu *d_pThisNeuronWeights = dp_pThisLayerWeights + threadIdx.x * (p_iNumOutputsLayerBefore+1);
	real_gpu dErrorMultDerivativeMultEta = dp_pThisLayerError[threadIdx.x] * dp_pDerivativeOfLastOutput[threadIdx.x] * p_dEta;

	PRINT_DEBUG_INFO("GPU: Test 0 , Neuron %d : First weight: %f , dErrorMultDerivativeMultEta: %f , p_iNumOutputsLayerBefore: %d\n",threadIdx.x,d_pThisNeuronWeights[0],dErrorMultDerivativeMultEta,p_iNumOutputsLayerBefore);
	
	for(unsigned uWeightIndex = 0;uWeightIndex < p_iNumOutputsLayerBefore;++uWeightIndex)
	{
		real_gpu dChange = dErrorMultDerivativeMultEta * dp_pLayerBeforeOutputs[uWeightIndex];
		real_gpu dCurrentValue = d_pThisNeuronWeights[uWeightIndex];
		real_gpu dChangedValue = dCurrentValue - dChange;
		PRINT_DEBUG_INFO("GPU: Test 0 , Neuron %d , uWeightIndex %d: dCurrentValue = %f , dChange %f , dChangedValue %f\n",threadIdx.x,uWeightIndex,dCurrentValue,dChange,dChangedValue);
		d_pThisNeuronWeights[uWeightIndex] = dChangedValue;
	}
	
	PRINT_DEBUG_INFO("GPU: Test 0 , Neuron %d , Bias : dChange %f\n",threadIdx.x,dErrorMultDerivativeMultEta);
	d_pThisNeuronWeights[p_iNumOutputsLayerBefore] = d_pThisNeuronWeights[p_iNumOutputsLayerBefore] - dErrorMultDerivativeMultEta;*/
}

extern "C" void updateWeightsInTrainingCUDA(const real_gpu *dp_pThisLayerError,const real_gpu *dp_pDerivativeOfLastOutput,const real_gpu *dp_pLayerBeforeOutputs,real_gpu p_dEta,int p_iThisLayerNeuronCount
											,int p_iNumOutputsLayerBefore,real_gpu *dp_pThisLayerWeights,int p_iNumTestsInBatch,bool p_bLayerBeforeOutputsHaveSpecificIndexes)
{
	int iElementsAllocatedForOneTestInThisLayerAligned = ALIGN_UP(p_iThisLayerNeuronCount+1,HALF_WARP);
	int iElementsAllocatedForOneTestInLayerBeforeAligned = ALIGN_UP(p_iNumOutputsLayerBefore+1,HALF_WARP);
	int iSharedMemorySize = 4 * p_iNumTestsInBatch * sizeof(real_gpu); // memory for ThisLayerError and DerivativeOfLastOutput (2 times)

	updateWeightsInTrainingKernel <<<(p_iThisLayerNeuronCount+1)/2,p_iNumOutputsLayerBefore+1,iSharedMemorySize>>> (dp_pThisLayerError,dp_pDerivativeOfLastOutput,dp_pLayerBeforeOutputs,p_dEta
		,dp_pThisLayerWeights,p_iNumTestsInBatch,iElementsAllocatedForOneTestInThisLayerAligned,iElementsAllocatedForOneTestInLayerBeforeAligned,p_bLayerBeforeOutputsHaveSpecificIndexes,p_iThisLayerNeuronCount);
}
