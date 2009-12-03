#include "stdafx.h"

const Str m_XMLNeuron("Neuron");

#define ZERO_GPU_MEMORY()	md_pLayerWeights = NULL; \
							md_pDerivativeOfLastOutput = NULL; \
							md_pLastOutputWithOutputFunction = NULL; \
							md_pLastError = NULL; 

Layer::Layer()
{
	m_pNetwork = NULL;
	m_iLayerIndex = -1;
	ZERO_GPU_MEMORY();
}

Layer::Layer(const Layer &p_Layer)
{
	// we don't copy md_pLayerMemory, because we unallocate this memory in destructor
	m_vecNeurons.assign(p_Layer.m_vecNeurons.begin(),p_Layer.m_vecNeurons.end());
	m_pNetwork = p_Layer.m_pNetwork;
	m_iLayerIndex = p_Layer.m_iLayerIndex;
	ZERO_GPU_MEMORY();
}

Layer::Layer(unsigned p_uNumberNeurons,unsigned p_uNumberWeights,Neuron::NeuronType p_eNeuronType)
{
	for(unsigned uNeuronIndex = 0;uNeuronIndex < p_uNumberNeurons;++uNeuronIndex)
		m_vecNeurons.push_back(Neuron(this,p_uNumberWeights,p_eNeuronType));
	m_pNetwork = NULL;
	m_iLayerIndex = -1;
	ZERO_GPU_MEMORY();
}

Layer::~Layer()
{
	// JRTODO deallocate md_pLayerMemory
}

Layer *Layer::getLayerBefore()
{
	return m_pNetwork->getLayerBefore(this);
}

Layer *Layer::getLayerAfter()
{
	return m_pNetwork->getLayerAfter(this);
}

void Layer::executeLayer(const vector<double> &p_vecLayerInput, vector<double> &p_vecLayerOutput)
{
	// We execute all neurons
	p_vecLayerOutput.clear();
	for(unsigned uNeuronIndex = 0;uNeuronIndex < m_vecNeurons.size();++uNeuronIndex)
	{
		double dResult;
		m_vecNeurons[uNeuronIndex].executeNeuron(p_vecLayerInput,dResult);
		p_vecLayerOutput.push_back(dResult);
	}
}

void Layer::executeLayerGPU(real_gpu *pd_InputMemory, real_gpu *&pd_OutputMemory)
{
	pd_InputMemory;
	pd_OutputMemory;
}

void Layer::randomizeWeights(double p_dAbsMax,MTRand *p_pRandomGenerator)
{
	for(unsigned uNeuronIndex = 0;uNeuronIndex < m_vecNeurons.size();++uNeuronIndex)
	{
		m_vecNeurons[uNeuronIndex].randomizeWeights(p_dAbsMax,p_pRandomGenerator);
	}
}

int Layer::getWeightCount() const
{
	return m_vecNeurons[0].getWeightsSize();
}

int Layer::getNeuronCount() const
{
	return m_vecNeurons.size();
}

Neuron::NeuronType Layer::getNeuronType() const
{
	return m_vecNeurons[0].getNeuronType();
}

/*void Layer::updateLayerWeights(const vector<double> &p_vecDifferencesOutput,vector<double> &p_vecDifferencesLayerBefore,double p_dEta)
{
	p_vecDifferencesLayerBefore.insert(p_vecDifferencesLayerBefore.begin(),getWeightCount(),0.0);
	for(unsigned uNeuronIndex=0;uNeuronIndex<getNeuronCount.size();++uNeuronIndex)
	{
		m_vecNeurons[uNeuronIndex].updateNeuronWeights(p_vecDifferencesOutput[uNeuronIndex],p_dEta,p_vecDifferencesLayerBefore);
	}
}*/

void Layer::updateErrorValues()
{
	vector<Neuron> &vecNeuronsLayerAfter = getLayerAfter()->m_vecNeurons;
	int iNeuronsInLayerAfter = vecNeuronsLayerAfter.size();
	for(unsigned a=0;a<vecNeuronsLayerAfter[0].m_vecLastError.size();++a)
	{
		for(int iNeuronIndex=0;iNeuronIndex<getNeuronCount();++iNeuronIndex)
		{
			double dError = 0.0;
			for(int iNeuronIndexAfter=0;iNeuronIndexAfter<iNeuronsInLayerAfter;++iNeuronIndexAfter)
			{
				dError += vecNeuronsLayerAfter[iNeuronIndexAfter].m_vecWeights[iNeuronIndex] * vecNeuronsLayerAfter[iNeuronIndexAfter].m_vecLastError[a];
			}

			m_vecNeurons[iNeuronIndex].m_vecLastError.push_back(dError);
		}
	}
}

void Layer::updateWeights(const vector< vector<double> > &p_vecOutputsLayerBefore, double p_dEta)
{
	for(int iNeuronIndex=0;iNeuronIndex<getNeuronCount();++iNeuronIndex)
	{
		m_vecNeurons[iNeuronIndex].updateWeights(p_vecOutputsLayerBefore, p_dEta);
	}
}

void Layer::saveToXML(TiXmlElement &p_XML) const
{
	// we save all neurons
	for(unsigned iNeuronIndex = 0;iNeuronIndex < m_vecNeurons.size();++iNeuronIndex)
	{
		TiXmlElement newNeuronElement(m_XMLNeuron.c_str());
		m_vecNeurons[iNeuronIndex].saveToXML(newNeuronElement);
		p_XML.InsertEndChild(newNeuronElement);
	}
}

void Layer::loadFromXML(const TiXmlElement &p_XML)
{
	// we load all neurons
	const TiXmlElement *pXMLNeuron = p_XML.FirstChildElement();
	while(pXMLNeuron)
	{
		logAssert(pXMLNeuron->Value() == m_XMLNeuron);
		Neuron newNeuron(this);
		newNeuron.loadFromXML(*pXMLNeuron);
		m_vecNeurons.push_back(newNeuron);
		pXMLNeuron = pXMLNeuron->NextSiblingElement();
	}
}
