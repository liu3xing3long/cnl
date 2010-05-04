#include "stdafx.h"

const Str m_XMLLayer("Layer");

void MLP::executeNetwork(InputTestSet &p_TestSet)
{
	logAssert(m_vecLayers.size() != 0 && p_TestSet.getTestCount() != 0); // Assert if there are any layers and tests
	logAssert(m_vecLayers[0].getWeightCount() == (int) p_TestSet.getInputCount() + 1); // Assert if number of inputs is correct
	logAssert(m_vecLayers[m_vecLayers.size()-1].getNeuronCount() == (int) p_TestSet.getOutputCount()); // Assert if number of outputs is correct

	// execute network on all tests
	for(unsigned iTestIndex = 0;iTestIndex < p_TestSet.getTestCount();++iTestIndex)
	{
		executeNetwork(p_TestSet.getTest(iTestIndex));
	}

	cleanTemporaryData();
}

void MLP::executeNetwork(InputTest &p_Test)
{
	// execute network on all layers for this test
	vector<double> vecLayerInput(p_Test.getInputs());
	vector<double> vecLayerOutput;
	for(unsigned iLayerIndex = 0;iLayerIndex < m_vecLayers.size();++iLayerIndex)
	{
		m_vecLayers[iLayerIndex].executeLayer(vecLayerInput,vecLayerOutput);

		// Now output layer becomes input layer (except when in the last layer)
		if(iLayerIndex != m_vecLayers.size() - 1)
			vecLayerInput.swap(vecLayerOutput);
	}

	// We put the result into the test
	p_Test.setOutputs(vecLayerOutput);
}

void MLP::trainNetwork(InputTestSet &p_TestSet,int p_iTrainedElements, double p_dEta,int p_iNumTestsInBatch,MTRand *p_pRandomGenerator)
{
	logAssert(m_vecLayers.size() != 0 && p_TestSet.getTestCount() != 0); // Assert if there are any layers and tests
	logAssert(m_vecLayers[0].getWeightCount() == (int) p_TestSet.getInputCount() + 1); // Assert if number of inputs is correct
	logAssert(m_vecLayers[m_vecLayers.size()-1].getNeuronCount() == (int) p_TestSet.getOutputCount()); // Assert if number of outputs is correct
	logAssert(p_dEta > 0.0 && p_dEta < 1.0);

	for(int iTrainedElement=0;iTrainedElement<p_iTrainedElements;++iTrainedElement)
	{
		printf("Iter %d\n",iTrainedElement);
		// We get test index to be used in training

		vector<InputTest *> vecTests;

		// Clean everything
		cleanTemporaryData();

		logTextParams(Logging::LT_DEBUG,"iTrainedElement = %d",iTrainedElement);

		for(int iTestInBatchIndex=0;iTestInBatchIndex<p_iNumTestsInBatch;++iTestInBatchIndex)
		{
			int iTestIndex = (int) (getRandom01(p_pRandomGenerator) * p_TestSet.getTestCount()); // iTrainedElement % p_TestSet.getTestCount()

			logTextParams(Logging::LT_DEBUG,"Test in batch nr %d = %d",iTestInBatchIndex,iTestIndex);

			InputTest &test = p_TestSet.getTest(iTestIndex);
			vecTests.push_back(&test);

			// we Execute network and get differences between required results and network output
			executeNetwork(test);

			// We assign error values in the last layer
			for(unsigned uOutputElement=0;uOutputElement<p_TestSet.getOutputCount();++uOutputElement)
			{
				double dError = test.getNetworkOutput(uOutputElement) - test.getCorrectOutput(uOutputElement);
				logTextParams(Logging::LT_DEBUG,"Test in batch nr %d , Output %d : Network = %f , Correct  = %f , Error = %f"
					,iTestInBatchIndex,uOutputElement,test.getNetworkOutput(uOutputElement),test.getCorrectOutput(uOutputElement),dError);
				//vecDifferences.push_back(dError);
				m_vecLayers[m_vecLayers.size()-1].m_vecNeurons[uOutputElement].m_vecLastError.push_back(dError);
			}
		}

		// We move backwards through layers - we set neuron errors
		//vector<double> vecDifferencesBeforeLayer;
		for(int iLayerIndex=(int)(m_vecLayers.size()-2);iLayerIndex>=0;--iLayerIndex) // it has to be signed int!
		{
			logTextParams(Logging::LT_DEBUG,"Errors in layer %d",iLayerIndex);
			m_vecLayers[iLayerIndex].updateErrorValues();
		}

		// We can finally update weights in all neurons
		for(unsigned uLayerIndex=0;uLayerIndex<m_vecLayers.size();++uLayerIndex) 
		{
			vector< vector<double> > vecOutputsLayerBefore; // outputs from a layer before the current layer
			if(uLayerIndex == 0)
			{
				for(unsigned uTestIndex=0;uTestIndex<vecTests.size();++uTestIndex) 
					vecOutputsLayerBefore.push_back(vecTests[uTestIndex]->getInputs());
			}
			else
			{				
				Layer &layerBefore = m_vecLayers[uLayerIndex-1];

				for(unsigned uTestIndex=0;uTestIndex<vecTests.size();++uTestIndex) 
				{
					vecOutputsLayerBefore.push_back( vector<double> () );
					for(int iNeuronIndex=0;iNeuronIndex<layerBefore.getNeuronCount();++iNeuronIndex) 
					{
						vecOutputsLayerBefore[uTestIndex].push_back(layerBefore.m_vecNeurons[iNeuronIndex].m_vecLastOutputWithOutputFunction[uTestIndex]);
					}
				}
			}

			logTextParams(Logging::LT_DEBUG,"Errors in layer %d , elements in vecOutputsLayerBefore: %d",uLayerIndex,vecOutputsLayerBefore.size());
			for(unsigned a=0;a<vecOutputsLayerBefore.size();++a)
			{
				logTextParams(Logging::LT_DEBUG,"vecOutputsLayerBefore [%d], size %d",a,vecOutputsLayerBefore[a].size());
				for(unsigned b=0;b<vecOutputsLayerBefore[a].size();++b)
				{
					logTextParams(Logging::LT_DEBUG,"vecOutputsLayerBefore [%d] [%d] = %f",a,b,vecOutputsLayerBefore[a][b]);
				}
			}

			m_vecLayers[uLayerIndex].updateWeights(vecOutputsLayerBefore,p_dEta);
		}
	}
}

void MLP::executeNetworkGPU(InputTestSet &p_TestSet,unsigned int *p_uiFullMilliseconds,unsigned int *p_uiKernelMilliseconds)
{
	logAssert(m_vecLayers.size() != 0 && p_TestSet.getTestCount() != 0); // Assert if there are any layers and tests
	logAssert(m_vecLayers[0].getWeightCount() == (int) p_TestSet.getInputCount() + 1); // Assert if number of inputs is correct
	logAssert(m_vecLayers[m_vecLayers.size()-1].getNeuronCount() == (int) p_TestSet.getOutputCount()); // Assert if number of outputs is correct

	Logging::Timer timerFull, timerKernel;
	timerFull.start();
	unsigned int uiKernelMilliseconds = 0;
	// execute network on all layers for this test
	real_gpu *d_pLayerInput = NULL;
	real_gpu *d_pWeights = NULL;
	real_gpu *d_pLayerOutput = NULL;
	
	for(unsigned iLayerIndex = 0;iLayerIndex < m_vecLayers.size();++iLayerIndex)
	{
		const Layer &thisLayer = m_vecLayers[iLayerIndex];
		// unallocate and allocate memory
		//int iMemorySizeForOutput = getMemorySizeForOutput();
		//int iMemorySizeForWeights = getMemorySizeForWeights();
		if(iLayerIndex == 0)
		{
			int iDummy;
			d_pLayerInput = CUDATools::setGPUMemoryForInputLayer(p_TestSet,iDummy);
		}
		else
		{
			CUDATools::freeGPUMemory(d_pLayerInput);
			d_pLayerInput = d_pLayerOutput;
			CUDATools::freeGPUMemory(d_pWeights);
		}

		d_pWeights =  CUDATools::setGPUMemoryForWeights(thisLayer);
		d_pLayerOutput = CUDATools::allocateGPUMemoryForHiddenOrOutputLayer(p_TestSet,thisLayer);

		timerKernel.start();
		CUDATools::executeLayerGPU(d_pLayerInput,d_pWeights,d_pLayerOutput,p_TestSet,thisLayer);
		uiKernelMilliseconds += timerKernel.stop();
	}

	CUDATools::retrieveOutputsGPU(d_pLayerOutput,p_TestSet);
	CUDATools::freeGPUMemory(d_pLayerInput);
	CUDATools::freeGPUMemory(d_pWeights);
	CUDATools::freeGPUMemory(d_pLayerOutput);

	if(p_uiFullMilliseconds != NULL)
		*p_uiFullMilliseconds = timerFull.stop();
	if(p_uiKernelMilliseconds != NULL)
		*p_uiKernelMilliseconds = uiKernelMilliseconds;
}

void MLP::trainNetworkGPU(InputTestSet &p_TestSet,int p_iTrainedElements,double p_dEta,int p_iNumTestsInBatch,MTRand *p_pRandomGenerator)
{
	logAssert(m_vecLayers.size() != 0 && p_TestSet.getTestCount() != 0); // Assert if there are any layers and tests
	logAssert(m_vecLayers[0].getWeightCount() == (int) p_TestSet.getInputCount() + 1); // Assert if number of inputs is correct
	logAssert(m_vecLayers[m_vecLayers.size()-1].getNeuronCount() == (int) p_TestSet.getOutputCount()); // Assert if number of outputs is correct
	logAssert(p_dEta > 0.0 && p_dEta < 1.0);

	for(unsigned iLayerIndex = 0;iLayerIndex < m_vecLayers.size();++iLayerIndex)
	{
		CUDATools::allocateAndSetGPUMemoryForLayerTraining(m_vecLayers[iLayerIndex],p_iNumTestsInBatch);
	}

	vector<int>vecTrainedElements;
	real_gpu *d_pTestsInput = NULL; // tests input
	real_gpu *d_pTestsOutput = NULL; // correct outputs
	int iSpaceBetweenTestsInInput; // position difference between each test in d_pTestInput
	int iSpaceBetweenTestsInOutput; // position difference between each test in d_pTestOutput
	CUDATools::allocateAndSetGPUMemoryForTestTraining(d_pTestsInput,d_pTestsOutput,p_TestSet,iSpaceBetweenTestsInInput,iSpaceBetweenTestsInOutput);

	for(int iTrainedElement=0;iTrainedElement<p_iTrainedElements;++iTrainedElement)
	{
		vecTrainedElements.clear();
		for(int iTestInBatchIndex=0;iTestInBatchIndex<p_iNumTestsInBatch;++iTestInBatchIndex)
		{
			// We get test index to be used in training
			int iTestIndex = (int) (getRandom01(p_pRandomGenerator) * p_TestSet.getTestCount());
			vecTrainedElements.push_back(iTestIndex);
		}

		logTextParams(Logging::LT_DEBUG,"GPU: iTrainedElement = %d",iTrainedElement);
		logTextParams(Logging::LT_DEBUG,"GPU: Test in batch nr 0 = %d",vecTrainedElements[0]);

		// 1. We execute the test on the network
		for(unsigned iLayerIndex = 0;iLayerIndex < m_vecLayers.size();++iLayerIndex)
		{
			const Layer &thisLayer = m_vecLayers[iLayerIndex];
			
			real_gpu *d_pLayerInput = NULL;
		
			if(iLayerIndex == 0)
			{
				d_pLayerInput = d_pTestsInput;
			}
			else
			{
				d_pLayerInput = m_vecLayers[iLayerIndex-1].md_pLastOutputWithOutputFunction;
			}

			CUDATools::executeLayerGPUForTraining(d_pLayerInput,thisLayer,vecTrainedElements,(iLayerIndex == 0));
		}

		// calculate error in the last layer
		CUDATools::calculateErrorInLastLayer(m_vecLayers[m_vecLayers.size()-1],(int)vecTrainedElements.size(),iSpaceBetweenTestsInOutput,d_pTestsOutput);

		// calculate error in other layers
		for(int iLayerIndex=(int)(m_vecLayers.size()-2);iLayerIndex>=0;--iLayerIndex) // it has to be signed int!
		{
			logTextParams(Logging::LT_DEBUG,"GPU: Errors in layer %d",iLayerIndex);
			CUDATools::calculateErrorInNotLastLayer(m_vecLayers[iLayerIndex],(int)vecTrainedElements.size());
		}

		// We can finally update weights in all neurons
		for(unsigned uLayerIndex=0;uLayerIndex<m_vecLayers.size();++uLayerIndex) 
		{
			const real_gpu *d_pOutputsLayerBefore;
			int p_iNumOutputsLayerBefore = 0;
			bool bLayerBeforeOutputsHaveSpecificIndexes = false;
			if(uLayerIndex == 0)
			{
				d_pOutputsLayerBefore = d_pTestsInput;
				p_iNumOutputsLayerBefore = p_TestSet.getInputCount();
				bLayerBeforeOutputsHaveSpecificIndexes = true;
			}
			else
			{
				Layer &layerBefore = m_vecLayers[uLayerIndex-1];
				d_pOutputsLayerBefore = layerBefore.md_pLastOutputWithOutputFunction;
				p_iNumOutputsLayerBefore = layerBefore.getNeuronCount();
			}

			logTextParams(Logging::LT_DEBUG,"GPU: Updating weights in layer %d",uLayerIndex);
			
			CUDATools::updateWeightsInTraining( m_vecLayers[uLayerIndex],d_pOutputsLayerBefore,p_iNumOutputsLayerBefore,p_dEta,(int)vecTrainedElements.size(),bLayerBeforeOutputsHaveSpecificIndexes);
		}
	}

	//We copy updated weights back to the Layer structures
	for(unsigned iLayerIndex = 0;iLayerIndex < m_vecLayers.size();++iLayerIndex)
	{
		CUDATools::retrieveGPUWeightsForLayerTraining(m_vecLayers[iLayerIndex]);
	}

	// JRTODO - check if all memory was freed
	// We free all memory needed for training
	CUDATools::freeGPUMemoryForTestTraining(d_pTestsInput,d_pTestsOutput);
	
	for(unsigned iLayerIndex = 0;iLayerIndex < m_vecLayers.size();++iLayerIndex)
	{
		CUDATools::freeGPUMemoryForLayerTraining(m_vecLayers[iLayerIndex]);
	}
}

void MLP::randomizeWeights(double p_dAbsMax,MTRand *p_pRandomGenerator)
{
	for(unsigned iLayerIndex = 0;iLayerIndex < m_vecLayers.size();++iLayerIndex)
	{
		m_vecLayers[iLayerIndex].randomizeWeights(p_dAbsMax,p_pRandomGenerator);
	}
}

void MLP::clearNetwork()
{
	m_vecLayers.clear();
}

void MLP::cleanTemporaryData()
{
	for(unsigned uLayerIndex=0;uLayerIndex<m_vecLayers.size();++uLayerIndex)
	{
		for(unsigned uNeuronIndex=0;uNeuronIndex<m_vecLayers[uLayerIndex].m_vecNeurons.size();++uNeuronIndex)
		{
			m_vecLayers[uLayerIndex].m_vecNeurons[uNeuronIndex].m_vecLastError.clear();
			m_vecLayers[uLayerIndex].m_vecNeurons[uNeuronIndex].m_vecDerivativeOfLastOutput.clear();
			m_vecLayers[uLayerIndex].m_vecNeurons[uNeuronIndex].m_vecLastOutputWithOutputFunction.clear();
		}
	}
}

Layer *MLP::getLayerBefore(Layer *p_pLayer)
{
	if(p_pLayer->m_iLayerIndex == 0)
		return NULL;
	return &m_vecLayers[p_pLayer->m_iLayerIndex-1];
}

Layer *MLP::getLayerAfter(Layer *p_pLayer)
{
	if(p_pLayer->m_iLayerIndex == (int)(m_vecLayers.size()-1))
		return NULL;
	return &m_vecLayers[p_pLayer->m_iLayerIndex+1];
}

void MLP::saveToXML(TiXmlElement &p_XML) const
{
	// we save all layers
	for(unsigned iLayerIndex = 0;iLayerIndex < m_vecLayers.size();++iLayerIndex)
	{
		TiXmlElement newLayerElement(m_XMLLayer.c_str());
		m_vecLayers[iLayerIndex].saveToXML(newLayerElement);
		p_XML.InsertEndChild(newLayerElement);
	}
}

void MLP::loadFromXML(const TiXmlElement &p_XML)
{
	const TiXmlElement *pXMLLayer = p_XML.FirstChildElement();
	while(pXMLLayer)
	{
		Layer newLayer;
		newLayer.loadFromXML(*pXMLLayer);
		m_vecLayers.push_back(newLayer);
		pXMLLayer = pXMLLayer->NextSiblingElement();
	}
}

MLP::MLP()
: NeuralNetwork(NNT_MLP)
{
	
}

MLP::MLP(const MLP &p_Other)
: NeuralNetwork(NNT_MLP)
{
	m_vecLayers.assign(p_Other.m_vecLayers.begin(),p_Other.m_vecLayers.end());
	for(unsigned uLayerIndex = 0;uLayerIndex < m_vecLayers.size();++uLayerIndex)
	{
		m_vecLayers[uLayerIndex].m_pNetwork = this;
	}
}

void MLP::setInputNeuronCount(int p_iInputNeuronCount)
{
	m_iInputNeuronCount = p_iInputNeuronCount;
}

void MLP::addNewLayer(unsigned p_uNumberNeurons,Neuron::NeuronType p_eNeuronType)
{
	int iWeightCount = 0;
	if(m_vecLayers.size() != 0)
		iWeightCount = m_vecLayers[m_vecLayers.size()-1].getNeuronCount();
	else
		iWeightCount = m_iInputNeuronCount;

	m_vecLayers.push_back(Layer(p_uNumberNeurons,iWeightCount,p_eNeuronType));

	m_vecLayers[m_vecLayers.size()-1].m_pNetwork = this;
	m_vecLayers[m_vecLayers.size()-1].m_iLayerIndex = (int)m_vecLayers.size()-1;
}
