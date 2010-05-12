#pragma once

class Layer;

class Neuron
{
	friend class CUDATools;
	friend class MLP;
	friend class Layer;

public:
	enum NeuronType 
	{ 	
		NT_OFF,
		NT_LINEAR,
		NT_SIGMOID,
		NT_TANH
	};

private:

	//Layer *m_pLayer;

	// The last element is bias
	vector<double> m_vecWeights;
	NeuronType m_eNeuronType;
	vector<double> m_vecDerivativeOfLastOutput;			// Derivative of last neuron output (result) calculated by executeNeuron, without executing special output function
	vector<double> m_vecLastOutputWithOutputFunction;	// Last neuron output (result) calculated by executeNeuron, with executing special output function
	vector<double> m_vecLastError; 

	Str getNeuronTypeString() const;

	void setNeuronTypeString(Str p_sXMLNeuronTypeString);

public:

	Neuron(Layer *p_pLayer);
	Neuron(Layer *p_pLayer, unsigned p_uNumberWeights, NeuronType p_eNeuronType);

	void executeNeuron(const vector<double> &p_vecLayerInput, double &p_dResult);
	//void updateLayerWeights(double p_dDifferenceOutput,double p_dEta,vector<double> &p_vecDifferencesLayerBefore);
	void updateWeights(const vector< vector<double> > &p_vecOutputsLayerBefore, double p_dEta);

	void randomizeWeights(double p_dAbsMax,MTRand *p_pRandomGenerator);
	int getWeightsSize() const;
	NeuronType getNeuronType() const;

	void saveToXML(TiXmlElement &p_XML) const;
	void loadFromXML(const TiXmlElement &p_XML);
};
