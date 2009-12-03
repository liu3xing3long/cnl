#include "stdafx.h"

//#define __DEVICE_EMULATION__
#pragma warning(push)
#pragma warning(disable:4201)
#pragma warning(disable:4408)
#include <cuda_runtime.h>
#pragma warning(pop)

#define cudaCheckError(oper)													\
{																				\
	cudaError_t resultCudaOperation = (oper);									\
	if( cudaSuccess != resultCudaOperation)										\
	{																			\
		logText(Logging::LT_ERROR,"Error performing operation \""				\
			+ Str(#oper)+"\". Error: "											\
			+ Str(cudaGetErrorString( resultCudaOperation)));					\
    }																			\
}

extern "C" void executeLayerCUDA(const real_gpu *dp_pLayerInput,const real_gpu *dp_pWeights,real_gpu *dp_pLayerOutput,real_gpu *dp_pDerivativeOfLastOutput
								 ,int p_iTestCount,int p_iOutputNeuronCount,int p_iNumInputNeurons, Neuron::NeuronType p_eNeuronType);

extern "C" void calculateErrorInLastLayerCUDA(const real_gpu *dp_pCorrectOutput,const real_gpu *dp_pNetworkOutput,real_gpu *dp_pErrors,int p_iOutputNeuronCount);

extern "C" void calculateErrorInNotLastLayerCUDA(const real_gpu *dp_pNextLayerWeights,const real_gpu *dp_pNextLayerError,real_gpu *dp_pThisLayerError,int p_iThisLayerNeuronCount,int p_iNextLayerNeuronCount);


// t(a,b) - neuron 'b' of test 'a' (in case of 11 neurons) :
// t(0,0),t(0,1),t(0,2),t(0,3)			,t(0,4)	,t(0,5)	,t(0,6)	,t(0,7)		<0-7>
// t(0,8),t(0,9),t(0,10),t(0,11) == 1.0	,0		,0		,0		,0			<8-15>
// t(1,0),t(1,1),t(1,2),t(1,3)			,t(1,4)	,t(1,5)	,t(1,6)	,t(1,7)		<16-23>
// t(1,8),t(1,9),t(1,10),t(1,11) == 1.0	,0		,0		,0		,0			<24-31>
real_gpu* CUDATools::setGPUMemoryForInputLayer(const InputTestSet &p_TestSet, int &p_iSpaceBetweenTestsInInput)
{
	// allocate memory
	int iNumNeurons = p_TestSet.getInputCount(); // and bias
	int iNumNeuronsAligned = iNumNeurons + 1;
	ALIGN_UP(iNumNeuronsAligned,HALF_WARP); // Padding neurons to 16 elements
	p_iSpaceBetweenTestsInInput = iNumNeuronsAligned;
	int iTestCount = p_TestSet.getTestCount();
	int iNumberCount = iNumNeuronsAligned * iTestCount;
	int iBytesAllocated = iNumberCount * sizeof(real_gpu);
	real_gpu *pHostMemory;
	real_gpu *d_pGPUMemory;
	allocateHostAndGPUMemory(iBytesAllocated,pHostMemory,d_pGPUMemory);
	memset(pHostMemory,0,iBytesAllocated);

	// Set host memory
	for(int iTestIndex=0;iTestIndex<iTestCount;++iTestIndex)
	{
		real_gpu *pBasePointer = &pHostMemory[iNumNeuronsAligned * iTestIndex];
		const InputTest &testNow = p_TestSet.getTest(iTestIndex);
		for(int iInputIndex=0;iInputIndex<iNumNeurons;++iInputIndex)
			pBasePointer[iInputIndex] = (real_gpu)testNow.m_vecInputs[iInputIndex];
		pBasePointer[iNumNeurons] = 1.0; // bias
	}

	cudaCheckError(cudaMemcpy(d_pGPUMemory,pHostMemory,iBytesAllocated,cudaMemcpyHostToDevice));
	cudaCheckError(cudaFreeHost(pHostMemory));

	return d_pGPUMemory;
}

real_gpu* CUDATools::setGPUMemoryForOutputLayer(const InputTestSet &p_TestSet, int &p_iSpaceBetweenTestsInOutput)
{
	// allocate memory
	int iNumNeurons = p_TestSet.getOutputCount();
	int iNumNeuronsAligned = iNumNeurons + 1;
	ALIGN_UP(iNumNeuronsAligned,HALF_WARP); // Padding neurons to 16 elements
	p_iSpaceBetweenTestsInOutput = iNumNeuronsAligned;
	int iTestCount = p_TestSet.getTestCount();
	int iNumberCount = iNumNeuronsAligned * iTestCount;
	int iBytesAllocated = iNumberCount * sizeof(real_gpu);
	real_gpu *pHostMemory;
	real_gpu *d_pGPUMemory;
	allocateHostAndGPUMemory(iBytesAllocated,pHostMemory,d_pGPUMemory);
	memset(pHostMemory,0,iBytesAllocated);

	// Set host memory
	for(int iTestIndex=0;iTestIndex<iTestCount;++iTestIndex)
	{
		real_gpu *pBasePointer = &pHostMemory[iNumNeuronsAligned * iTestIndex];
		const InputTest &testNow = p_TestSet.getTest(iTestIndex);
		for(int iInputIndex=0;iInputIndex<iNumNeurons;++iInputIndex)
			pBasePointer[iInputIndex] = (real_gpu)testNow.m_vecCorrectOutputs[iInputIndex];
		pBasePointer[iNumNeurons] = 1.0; // bias
	}

	cudaCheckError(cudaMemcpy(d_pGPUMemory,pHostMemory,iBytesAllocated,cudaMemcpyHostToDevice));
	cudaCheckError(cudaFreeHost(pHostMemory));

	return d_pGPUMemory;
}

// w(a,b) - weight 'b' of neuron 'a' (in case of 4 neurons, 3 weights) :
// t(0,0),t(0,1),t(0,2),t(1,0),t(1,1),t(1,2),t(2,0),t(2,1),t(2,2),t(3,0),t(3,1),t(3,2)		<0-11>
real_gpu* CUDATools::setGPUMemoryForWeights(const Layer &p_Layer)
{
	// allocate memory (not padded)
	int iNumNeurons = p_Layer.getNeuronCount(); 
	int iNumberWeightInNeuron = p_Layer.getWeightCount(); //with bias
	int iNumAllWeights = iNumNeurons * iNumberWeightInNeuron;
	int iBytesAllocated = iNumAllWeights * sizeof(real_gpu);
	real_gpu *pHostMemory;
	real_gpu *d_pGPUMemory;
	allocateHostAndGPUMemory(iBytesAllocated,pHostMemory,d_pGPUMemory);
	memset(pHostMemory,0,iBytesAllocated);

	// Set host memory
	for(int iNeuronIndex=0;iNeuronIndex<iNumNeurons;++iNeuronIndex)
	{
		real_gpu *pBasePointer = &pHostMemory[iNumberWeightInNeuron * iNeuronIndex];
		const Neuron &neuronNow = p_Layer.m_vecNeurons[iNeuronIndex];
		for(int iWeightIndex=0;iWeightIndex<iNumberWeightInNeuron;++iWeightIndex)
			pBasePointer[iWeightIndex] = (real_gpu)neuronNow.m_vecWeights[iWeightIndex];
	}

	cudaCheckError(cudaMemcpy(d_pGPUMemory,pHostMemory,iBytesAllocated,cudaMemcpyHostToDevice));
	cudaCheckError(cudaFreeHost(pHostMemory));

	return d_pGPUMemory;
}

real_gpu* CUDATools::allocateGPUMemoryForHiddenOrOutputLayer(const InputTestSet &p_TestSet, const Layer &p_Layer)
{
	// allocate memory
	int iNumNeurons = p_Layer.getNeuronCount()+1; // and bias
	int iNumNeuronsAligned = iNumNeurons;
	ALIGN_UP(iNumNeuronsAligned,HALF_WARP); // Padding neurons to 16 elements
	int iTestCount = p_TestSet.getTestCount();
	int iNumberCount = iNumNeuronsAligned * iTestCount;
	int iBytesAllocated = iNumberCount * sizeof(real_gpu);

	real_gpu *d_pGPUMemory = createZeroGPUMemory(iBytesAllocated);
	return d_pGPUMemory;
}

void CUDATools::retrieveOutputsGPU(const real_gpu *dp_pMemoryToAssign, InputTestSet &p_TestSet)
{
	// allocate host memory
	int iNumNeurons = p_TestSet.getOutputCount()+1; // and bias
	int iNumNeuronsAligned = iNumNeurons;
	ALIGN_UP(iNumNeuronsAligned,HALF_WARP); // Padding neurons to 16 elements
	int iTestCount = p_TestSet.getTestCount();
	int iNumberCount = iNumNeuronsAligned * iTestCount;
	int iBytesAllocated = iNumberCount * sizeof(real_gpu);
	real_gpu *pHostMemory;
	cudaCheckError(cudaMallocHost((void **)&pHostMemory, iBytesAllocated));

	cudaCheckError(cudaMemcpy(pHostMemory,dp_pMemoryToAssign,iBytesAllocated,cudaMemcpyDeviceToHost));
	
	// Set results
	for(int iTestIndex=0;iTestIndex<iTestCount;++iTestIndex)
	{
		real_gpu *pBasePointer = &pHostMemory[iNumNeuronsAligned * iTestIndex];
		InputTest &testNow = p_TestSet.getTest(iTestIndex);
		testNow.m_vecNetworkOutputsGPU.resize(iNumNeurons-1);
		for(int iInputIndex=0;iInputIndex<iNumNeurons - 1;++iInputIndex) // the last element is bias, so we don't need it
			testNow.m_vecNetworkOutputsGPU[iInputIndex] = (real_gpu)pBasePointer[iInputIndex];
	}

	cudaCheckError(cudaFreeHost(pHostMemory));
}

void CUDATools::executeLayerGPU(const real_gpu *dp_pLayerInput,const real_gpu *dp_pWeights,real_gpu *dp_pLayerOutput
	,const InputTestSet &p_TestSet, const Layer &p_Layer)
{
	//int iNumNeurons = p_TestSet.getOutputCount()+1; // and bias
	executeLayerCUDA(dp_pLayerInput,dp_pWeights,dp_pLayerOutput,NULL,p_TestSet.getTestCount(),p_Layer.getNeuronCount(),p_Layer.getWeightCount(),p_Layer.getNeuronType());
}

void CUDATools::executeLayerGPUForTraining(const real_gpu *dp_pLayerInput ,const InputTestSet &/*p_TestSet*/, const Layer &p_Layer)
{
	executeLayerCUDA(dp_pLayerInput,p_Layer.md_pLayerWeights,p_Layer.md_pLastOutputWithOutputFunction
		,p_Layer.md_pDerivativeOfLastOutput,1,p_Layer.getNeuronCount(),p_Layer.getWeightCount(),p_Layer.getNeuronType());
}

void CUDATools::allocateAndSetGPUMemoryForLayerTraining(Layer &p_Layer)
{
	p_Layer.md_pLayerWeights = setGPUMemoryForWeights(p_Layer);

	// JRTODO - currently we only allocate memory for one test
	int iBytesAllocated = p_Layer.getNeuronCount() * sizeof(real_gpu);

	p_Layer.md_pDerivativeOfLastOutput = createZeroGPUMemory(iBytesAllocated);
	p_Layer.md_pLastOutputWithOutputFunction = createZeroGPUMemory(iBytesAllocated);
	p_Layer.md_pLastError = createZeroGPUMemory(iBytesAllocated);
}

void CUDATools::allocateAndSetGPUMemoryForTestTraining(real_gpu *dp_pTestsInput,real_gpu *dp_pTestsOutput,const InputTestSet &p_TestSet,int &p_iSpaceBetweenTestsInInput,int &p_iSpaceBetweenTestsInOutput)
{
	dp_pTestsInput = CUDATools::setGPUMemoryForInputLayer(p_TestSet,p_iSpaceBetweenTestsInInput);
	dp_pTestsOutput = CUDATools::setGPUMemoryForOutputLayer(p_TestSet,p_iSpaceBetweenTestsInOutput);
}

void CUDATools::calculateErrorInLastLayer(Layer &p_LastLayer,real_gpu *dp_pCorrectOutput)
{
	calculateErrorInLastLayerCUDA(dp_pCorrectOutput,p_LastLayer.md_pLastOutputWithOutputFunction,p_LastLayer.md_pLastError,p_LastLayer.getNeuronCount());
}

void CUDATools::calculateErrorInNotLastLayer(Layer &p_Layer)
{
	calculateErrorInNotLastLayerCUDA(p_Layer.getLayerAfter()->md_pLayerWeights,p_Layer.getLayerAfter()->md_pLastError,p_Layer.md_pLastError
		,p_Layer.getNeuronCount(),p_Layer.getLayerAfter()->getNeuronCount());
}

void CUDATools::updateWeightsInTraining(Layer &p_Layer,const real_gpu *d_pOutputsLayerBefore,int p_iNumOutputsLayerBefore, double p_dEta)
{
	//updateWeightsInTrainingCUDA(p_Layer.md_pLastError,p_Layer.md_pDerivativeOfLastOutput,d_pOutputsLayerBefore,p_dEta,p_iNumOutputsLayerBefore,p_Layer.md_pLayerWeights);
		/*
		,p_Layer.getLayerAfter()->md_pLastError,p_Layer.md_pLastError
		,p_Layer.getNeuronCount(),p_Layer.getLayerAfter()->getNeuronCount());

	double dErrorMultDerivative = m_vecLastError * m_vecDerivativeOfLastOutput[uTestIndex];

	for(unsigned uWeightIndex = 0;uWeightIndex < p_vecOutputsLayerBefore.size();++uWeightIndex)
	{
		double dChange = dErrorMultDerivative * p_vecOutputsLayerBefore[uWeightIndex];
		m_vecWeights[uWeightIndex] -= p_dEta * dChange;
	}
	m_vecWeights[p_vecOutputsLayerBefore.size()] -= p_dEta * dErrorMultDerivative;*/
}

void CUDATools::retrieveGPUWeightsForLayerTraining(Layer &p_Layer)
{
	// allocate memory (not padded)
	int iNumNeurons = p_Layer.getNeuronCount(); 
	int iNumberWeightInNeuron = p_Layer.getWeightCount(); //with bias
	int iNumAllWeights = iNumNeurons * iNumberWeightInNeuron;
	int iBytesAllocated = iNumAllWeights * sizeof(real_gpu);

	real_gpu *pHostMemory;
	cudaCheckError(cudaMallocHost((void **)&pHostMemory, iBytesAllocated));
	cudaCheckError(cudaMemcpy(pHostMemory,p_Layer.md_pLayerWeights,iBytesAllocated,cudaMemcpyDeviceToHost));

	// Set results
	for(int iNeuronIndex=0;iNeuronIndex<iNumNeurons;++iNeuronIndex)
	{
		real_gpu *pBasePointer = &pHostMemory[iNumberWeightInNeuron * iNeuronIndex];
		Neuron &neuronNow = p_Layer.m_vecNeurons[iNeuronIndex];
		for(int iWeightIndex=0;iWeightIndex<iNumberWeightInNeuron;++iWeightIndex)
			neuronNow.m_vecWeights[iWeightIndex] = pBasePointer[iWeightIndex];
	}

	cudaCheckError(cudaFreeHost(pHostMemory));
}

void CUDATools::freeGPUMemoryForTestTraining(real_gpu *&dp_pTestsInput,real_gpu *&dp_pTestsOutput)
{
	freeGPUMemory(dp_pTestsInput);
	freeGPUMemory(dp_pTestsOutput);
}

void CUDATools::freeGPUMemoryForLayerTraining(Layer &p_Layer)
{
	freeGPUMemory(p_Layer.md_pLayerWeights);
	freeGPUMemory(p_Layer.md_pDerivativeOfLastOutput);
	freeGPUMemory(p_Layer.md_pLastOutputWithOutputFunction);
	freeGPUMemory(p_Layer.md_pLastError);
}

void CUDATools::freeGPUMemory(real_gpu *&dp_pMemoryToDeallocate)
{
	cudaCheckError(cudaFree(dp_pMemoryToDeallocate));
	dp_pMemoryToDeallocate=NULL;
}

void CUDATools::allocateHostAndGPUMemory(int p_iBytes,real_gpu *&p_pHostMemory,real_gpu *&dp_pGPUMemory)
{
	cudaCheckError(cudaMalloc((void **)&dp_pGPUMemory, p_iBytes));
	cudaCheckError(cudaMallocHost((void **)&p_pHostMemory, p_iBytes));
}

real_gpu* CUDATools::createZeroGPUMemory(int p_iBytes)
{
	real_gpu *pHostMemory;
	real_gpu *d_pGPUMemory;
	allocateHostAndGPUMemory(p_iBytes,pHostMemory,d_pGPUMemory);
	memset(pHostMemory,0,p_iBytes);

	cudaCheckError(cudaMemcpy(d_pGPUMemory,pHostMemory,p_iBytes,cudaMemcpyHostToDevice));
	cudaCheckError(cudaFreeHost(pHostMemory));

	return d_pGPUMemory;
}

