#pragma once

class CUDATools
{
public:

	// Methods for execute network
	static real_gpu* setGPUMemoryForInputLayer(const InputTestSet &p_TestSet, int &p_iSpaceBetweenTestsInInput);
	static real_gpu* setGPUMemoryForOutputLayer(const InputTestSet &p_TestSet, int &p_iSpaceBetweenTestsInOutput);
	static real_gpu* setGPUMemoryForWeights(const Layer &p_Layer);
	static real_gpu* allocateGPUMemoryForHiddenOrOutputLayer(const InputTestSet &p_TestSet, const Layer &p_Layer);
	static void retrieveOutputsGPU(const real_gpu *dp_pMemoryToAssign, InputTestSet &p_TestSet);
	static void executeLayerGPU(const real_gpu *dp_pLayerInput,const real_gpu *dp_pWeights,real_gpu *dp_pLayerOutput
		,const InputTestSet &p_TestSet, const Layer &p_Layer);

	// Methods for train network
	static void executeLayerGPUForTraining(const real_gpu *dp_pLayerInput,const Layer &p_Layer,const vector<int> &p_vecTrainedElements);
	static void allocateAndSetGPUMemoryForLayerTraining(Layer &p_Layer);
	static void allocateAndSetGPUMemoryForTestTraining(real_gpu *&dp_pTestsInput,real_gpu *&dp_pTestsOutput,const InputTestSet &p_TestSet,int &p_iSpaceBetweenTestsInInput,int &p_iSpaceBetweenTestsInOutput);
	static void calculateErrorInLastLayer(Layer &p_LastLayer,real_gpu *dp_pCorrectOutput);
	static void calculateErrorInNotLastLayer(Layer &p_Layer);
	static void updateWeightsInTraining(Layer &p_Layer,const real_gpu *d_pOutputsLayerBefore,int iNumOutputsLayerBefore, double p_dEta);
	static void retrieveGPUWeightsForLayerTraining(Layer &p_Layer);
	static void freeGPUMemoryForTestTraining(real_gpu *&dp_pTestsInput,real_gpu *&dp_pTestsOutput);
	static void freeGPUMemoryForLayerTraining(Layer &p_Layer);

	static void freeGPUMemory(real_gpu *&dp_pMemoryToDeallocate);

private:
	static void allocateHostAndGPUMemory(int p_iBytes,real_gpu *&p_pHostMemory,real_gpu *&dp_pGPUMemory);
	static real_gpu* createZeroGPUMemory(int p_iBytes);
};
