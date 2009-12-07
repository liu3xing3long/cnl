#pragma once

class NeuralNetwork
{
public:
	enum NeuralNetworkType
	{
		NNT_UNKNOWN,
		NNT_MLP
	};

	virtual void executeNetwork(InputTestSet &p_TestSet) = 0;
	virtual void trainNetwork(InputTestSet &p_TestSet,int p_iTrainedElements, double p_dEta,int p_iNumTestsInBatch,MTRand *p_pRandomGenerator) = 0;
	virtual void executeNetworkGPU(InputTestSet &p_TestSet) = 0;
	virtual void trainNetworkGPU(InputTestSet &p_TestSet, int p_iTrainedElements, double p_dEta,int p_iNumTestsInBatch,MTRand *p_pRandomGenerator) = 0;

	virtual void randomizeWeights(double p_dAbsMax,MTRand *p_pRandomGenerator) = 0;
	virtual void clearNetwork() = 0;
	virtual void cleanTemporaryData() = 0;

	virtual Layer *getLayerBefore(Layer *p_pLayer) = 0;
	virtual Layer *getLayerAfter(Layer *p_pLayer) = 0;

	virtual void saveToXML(TiXmlElement &p_XML) const = 0;
	virtual void loadFromXML(const TiXmlElement &p_XML) = 0;

	bool saveToFile(const Str &p_sFileName) const;
	static bool loadFromFile(const Str &p_sFileName, NeuralNetwork *&p_pReturnedNetwork);

protected:
	

	NeuralNetwork(NeuralNetworkType p_eNetworkType);

private:
	NeuralNetwork(); // invisible

	NeuralNetworkType m_eNetworkType;

	Str getNeuralNetworkTypeString() const;

	static NeuralNetwork *getNetworkFromNetworkType(Str p_sNetworkType);



	//static const Str m_XMLNeuralNetworkElementName;
};
