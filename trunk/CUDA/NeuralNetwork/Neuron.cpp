#include "stdafx.h"

const Str m_XMLNeuronType("NeuronType");
const Str m_XMLNeuronWeights("Weights");

Neuron::Neuron(Layer *p_pLayer)
: m_pLayer(p_pLayer)
, m_eNeuronType(Neuron::NT_OFF)
{
}

Neuron::Neuron(Layer *p_pLayer, unsigned p_uNumberWeights, NeuronType p_eNeuronType)
: m_pLayer(p_pLayer)
, m_eNeuronType(p_eNeuronType)
{
	// All weights + bias
	m_vecWeights.resize(p_uNumberWeights + 1);
}

Str Neuron::getNeuronTypeString() const
{
	switch(m_eNeuronType)
	{
		case NT_OFF: return Str("NT_OFF");
		case NT_LINEAR: return Str("NT_LINEAR");
		case NT_SIGMOID: return Str("NT_SIGMOID");
/*		case NT_TANH: return Str("NT_TANH");
		case NT_GAUSS: return Str("NT_GAUSS");
		case NT_SOFTMAX: return Str("NT_SOFTMAX");
		case NT_EXTERNAL: return Str("NT_EXTERNAL");*/
		default:
			logTextParams(Logging::LT_ERROR,"Incorrect neuron type: %d",m_eNeuronType);
			return Str("");
	}
}

void Neuron::setNeuronTypeString(Str p_sXMLNeuronTypeString)
{
	if(p_sXMLNeuronTypeString == "NT_OFF")		{ m_eNeuronType = NT_OFF;		return; }
	if(p_sXMLNeuronTypeString == "NT_LINEAR")	{ m_eNeuronType = NT_LINEAR;	return; }
	if(p_sXMLNeuronTypeString == "NT_SIGMOID")	{ m_eNeuronType = NT_SIGMOID;	return; }
	/*if(p_sXMLNeuronTypeString == "NT_TANH")		{ m_eNeuronType = NT_TANH;		return; }
	if(p_sXMLNeuronTypeString == "NT_GAUSS")	{ m_eNeuronType = NT_GAUSS;		return; }
	if(p_sXMLNeuronTypeString == "NT_SOFTMAX")	{ m_eNeuronType = NT_SOFTMAX;	return; }
	if(p_sXMLNeuronTypeString == "NT_EXTERNAL")	{ m_eNeuronType = NT_EXTERNAL;	return; }*/

	m_eNeuronType = NT_OFF;
	logTextParams(Logging::LT_ERROR,"Incorrect neuron type string: %s",p_sXMLNeuronTypeString.c_str());
}

void Neuron::executeNeuron(const vector<double> &p_vecLayerInput, double &p_dResult)
{
	size_t uInputCount = p_vecLayerInput.size();
	p_dResult = m_vecWeights[uInputCount]; // Bias
	for(unsigned uWeightIndex = 0;uWeightIndex < uInputCount;++uWeightIndex)
	{
		p_dResult += m_vecWeights[uWeightIndex] * p_vecLayerInput[uWeightIndex];
	}

	switch(m_eNeuronType)
	{
		case NT_LINEAR: 
			m_vecDerivativeOfLastOutput.push_back(1.0);
			break;	// Do nothing
		case NT_SIGMOID: 
			double dExp = exp(-p_dResult); 
			p_dResult = 1.0 / (1.0 + dExp);
			double dDerivative = dExp / pow(1.0 + dExp,2);
			m_vecDerivativeOfLastOutput.push_back(dDerivative);
			break;
	}

	m_vecLastOutputWithOutputFunction.push_back(p_dResult);
}

/*void Neuron::updateLayerWeights(double p_dDifferenceOutput,double p_dEta,vector<double> &p_vecDifferencesLayerBefore)
{
	
}*/

void Neuron::updateWeights(const vector< vector<double> > &p_vecOutputsLayerBefore, double p_dEta)
{
	logAssert(p_vecOutputsLayerBefore[0].size() + 1 == m_vecWeights.size());	

	for(unsigned uTestIndex = 0;uTestIndex < p_vecOutputsLayerBefore.size();++uTestIndex)
	{
		double dErrorMultDerivativeMultEta = m_vecLastError[uTestIndex] * m_vecDerivativeOfLastOutput[uTestIndex] * p_dEta;

		for(unsigned uWeightIndex = 0;uWeightIndex < p_vecOutputsLayerBefore[uTestIndex].size();++uWeightIndex)
		{
			double dChange = dErrorMultDerivativeMultEta * p_vecOutputsLayerBefore[uTestIndex][uWeightIndex];
			logTextParamsDebug("Test index %d , Weight index %d : Current %f , Change %f , Changed value %f",uTestIndex,uWeightIndex,m_vecWeights[uWeightIndex],dChange,m_vecWeights[uWeightIndex]-dChange);
			m_vecWeights[uWeightIndex] -= dChange;
			if(abs(m_vecWeights[uWeightIndex]) > 1000)
			{
				int r=2;
				r++;
			}
		}
		logTextParamsDebug("Test index %d , Bias : Change %f",uTestIndex,dErrorMultDerivativeMultEta);
		m_vecWeights[p_vecOutputsLayerBefore[uTestIndex].size()] -= dErrorMultDerivativeMultEta; // bias weight
	}
}

void Neuron::randomizeWeights(double p_dAbsMax,MTRand *p_pRandomGenerator)
{
	for(unsigned uWeightIndex = 0;uWeightIndex < m_vecWeights.size();++uWeightIndex)
	{
		m_vecWeights[uWeightIndex] = (getRandom01(p_pRandomGenerator)*2-1) * p_dAbsMax;
	}
}

int Neuron::getWeightsSize() const
{
	return (int) m_vecWeights.size();
}

Neuron::NeuronType Neuron::getNeuronType() const
{
	return m_eNeuronType;
}

void Neuron::saveToXML(TiXmlElement &p_XML) const
{
	// We save neuron type
	p_XML.SetAttribute(m_XMLNeuronType.c_str(),getNeuronTypeString().c_str());

	// we save weights
	saveDoubleVectorToXML(m_vecWeights,p_XML,m_XMLNeuronWeights);
}

void Neuron::loadFromXML(const TiXmlElement &p_XML)
{
	
	Str sNeuronType = p_XML.Attribute(m_XMLNeuronType.c_str());
	setNeuronTypeString(sNeuronType);

	// we load weights
	loadDoubleVectorFromXML(m_vecWeights,p_XML,m_XMLNeuronWeights);
}
