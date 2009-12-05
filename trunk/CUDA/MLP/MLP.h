#pragma once

class MLP : public NeuralNetwork
{
private:
	virtual void executeNetwork(InputTest &p_Test);
	virtual void executeNetworkGPU(InputTest &p_Test);

public: 
	virtual void executeNetwork(InputTestSet &p_TestSet);
	
	virtual void trainNetwork(InputTestSet &p_TestSet,int p_iTrainedElements, double p_dEta,int p_iNumTestsInBatch,MTRand *p_pRandomGenerator);
	virtual void executeNetworkGPU(InputTestSet &p_TestSet);
	
	virtual void trainNetworkGPU(InputTestSet &p_TestSet, int p_iTrainedElements, double p_dEta,int p_iNumTestsInBatch,MTRand *p_pRandomGenerator);

	virtual void randomizeWeights(double p_dAbsMax,MTRand *p_pRandomGenerator);
	virtual void clearNetwork();

	virtual Layer *getLayerBefore(Layer * p_pLayer);
	virtual Layer *getLayerAfter(Layer * p_pLayer);

	virtual void saveToXML(TiXmlElement &p_XML) const;
	virtual void loadFromXML(const TiXmlElement &p_XML);

	MLP();
	MLP(const MLP &p_Other);

	void addNewLayer(Layer p_LayerToAdd);

	vector<Layer> m_vecLayers;
	//int m_iNumInputNeurons;
	//int m_iNumOutputNeurons;
};
