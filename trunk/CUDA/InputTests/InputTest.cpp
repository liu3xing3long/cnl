#include "stdafx.h"

const Str m_XMLInputs("Inputs");
const Str m_XMLCorrectOutputs("CorrectOutputs");
const Str m_XMLNetworkOutputs("NetworkOutputs");
const Str m_XMLNetworkOutputsGPU("NetworkOutputsGPU");

InputTest::InputTest(InputTestSet *p_pParentTestSet, unsigned p_uNumberInputs,unsigned p_uNumberOutputs)
: m_pParentTestSet(p_pParentTestSet)
{
	m_vecInputs.assign(p_uNumberInputs,0);
	m_vecCorrectOutputs.assign(p_uNumberOutputs,0);
}

void InputTest::setOutputs(const vector<double> &p_vecLayerOutput)
{
	m_vecNetworkOutputs.assign(p_vecLayerOutput.begin(),p_vecLayerOutput.end());
}

void InputTest::randomizeTest(MTRand *p_pRandomGenerator)
{
	for(unsigned uInputIndex=0;uInputIndex<m_vecInputs.size();++uInputIndex)
		m_vecInputs[uInputIndex] = getRandom01(p_pRandomGenerator);

	for(unsigned uOutputIndex=0;uOutputIndex<m_vecCorrectOutputs.size();++uOutputIndex)
		m_vecCorrectOutputs[uOutputIndex] = getRandom01(p_pRandomGenerator);
}

void InputTest::setOutputFunction(const vector< pair<double,double> > &p_vecMinMax, void (*p_fTestingFunction)(const vector<double> &p_vecInputParameters,vector<double> &p_vecOutputParameters),MTRand *p_pRandomGenerator)
{
	for(unsigned uInputIndex=0;uInputIndex<m_vecInputs.size();++uInputIndex)
		m_vecInputs[uInputIndex] = getRandom01(p_pRandomGenerator) * (p_vecMinMax[uInputIndex].second - p_vecMinMax[uInputIndex].first) + p_vecMinMax[uInputIndex].first;

	p_fTestingFunction(m_vecInputs,m_vecCorrectOutputs);
}

void InputTest::saveToXML(TiXmlElement &p_XML) const
{
	// we save Inputs
	saveDoubleVectorToXML(m_vecInputs,p_XML,m_XMLInputs);
	// we save Correct Outputs
	saveDoubleVectorToXML(m_vecCorrectOutputs,p_XML,m_XMLCorrectOutputs);
	// we save Network Outputs
	saveDoubleVectorToXML(m_vecNetworkOutputs,p_XML,m_XMLNetworkOutputs);
	// we save Network Outputs GPU
	saveDoubleVectorToXML(m_vecNetworkOutputsGPU,p_XML,m_XMLNetworkOutputsGPU);
}

void InputTest::loadFromXML(const TiXmlElement &p_XML)
{
	// we load Inputs
	loadDoubleVectorFromXML(m_vecInputs,p_XML,m_XMLInputs);
	// we load Correct Outputs
	loadDoubleVectorFromXML(m_vecCorrectOutputs,p_XML,m_XMLCorrectOutputs);
	// we load Network Outputs
	loadDoubleVectorFromXML(m_vecNetworkOutputs,p_XML,m_XMLNetworkOutputs);
	// we load Network Outputs GPU
	loadDoubleVectorFromXML(m_vecNetworkOutputsGPU,p_XML,m_XMLNetworkOutputsGPU);
}
