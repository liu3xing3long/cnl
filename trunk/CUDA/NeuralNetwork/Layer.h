#pragma once

class NeuralNetwork;
class Layer
{
	friend class CUDATools;
	friend class MLP;

	vector<Neuron> m_vecNeurons;

	// Network containing this layer
	NeuralNetwork *m_pNetwork;
	int m_iLayerIndex;

	real_gpu *md_pLayerWeights;
	real_gpu *md_pDerivativeOfLastOutput;			// Derivative of last neuron output (result) calculated by executeNeuron, without executing special output function
	real_gpu *md_pLastOutputWithOutputFunction;		// Last neuron output (result) calculated by executeNeuron, with executing special output function
	real_gpu *md_pLastError;

	Layer *getLayerBefore();
	Layer *getLayerAfter();

public:

	Layer();
	Layer(const Layer &p_Layer);
	Layer(unsigned p_uNumberNeurons,unsigned p_uNumberWeights,Neuron::NeuronType p_eNeuronType);
	~Layer();

	void executeLayer(const vector<double> &m_vecLayerInput, vector<double> &m_vecLayerOutput);
	void executeLayerGPU(real_gpu *pd_InputMemory, real_gpu *&pd_OutputMemory);
	void randomizeWeights(double p_dAbsMax,MTRand *p_pRandomGenerator);
	int getWeightCount() const;
	int getNeuronCount() const;
	Neuron::NeuronType getNeuronType() const;
	//void updateLayerWeights(const vector<double> &p_vecDifferencesOutput,vector<double> &p_vecDifferencesLayerBefore,double p_dEta);
	void updateErrorValues();
	void updateWeights(const vector< vector<double> > &p_vecOutputsLayerBefore, double p_dEta);

	void saveToXML(TiXmlElement &p_XML) const;
	void loadFromXML(const TiXmlElement &p_XML);
};
