#pragma once

class MLP : public NeuralNetwork
{
private:
	int m_iInputNeuronCount;
	vector<Layer> m_vecLayers;

	virtual void executeNetwork(InputTest &p_Test);

protected:
	virtual void saveToXML(TiXmlElement &p_XML) const;
	virtual void loadFromXML(const TiXmlElement &p_XML);

public: 
	virtual void executeNetwork(InputTestSet &p_TestSet);
	
	virtual void trainNetwork(InputTestSet &p_TestSet,int p_iTrainedElements, double p_dEta,int p_iNumTestsInBatch,MTRand *p_pRandomGenerator);
	virtual void executeNetworkGPU(InputTestSet &p_TestSet,unsigned int *p_uiFullMilliseconds = NULL,unsigned int *p_uiKernelMilliseconds = NULL);
	
	virtual void trainNetworkGPU(InputTestSet &p_TestSet, int p_iTrainedElements, double p_dEta,int p_iNumTestsInBatch,MTRand *p_pRandomGenerator);

	virtual void randomizeWeights(double p_dAbsMax,MTRand *p_pRandomGenerator);
	virtual void clearNetwork();
	virtual void cleanTemporaryData();

	virtual Layer *getLayerBefore(Layer * p_pLayer);
	virtual Layer *getLayerAfter(Layer * p_pLayer);

	MLP();
	MLP(const MLP &p_Other);

	void setInputNeuronCount(int p_iNeuronCount);
	void addNewLayer(unsigned p_uNumberNeurons,Neuron::NeuronType p_eNeuronType);
};
