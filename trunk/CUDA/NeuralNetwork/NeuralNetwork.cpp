#include "stdafx.h"

const Str m_XMLNeuralNetworkElement("NeuralNetwork");
const Str m_XMLNeuralNetworkType("Type");

NeuralNetwork::NeuralNetwork(NeuralNetworkType p_eNetworkType)
{
	m_eNetworkType = p_eNetworkType;
}

NeuralNetwork::NeuralNetwork()
{
}


bool NeuralNetwork::saveToFile(const Str &p_sFileName) const
{
	FILE *pSaveFile = TiXmlFOpen(p_sFileName.c_str(),"wb");
	if(!pSaveFile)
		return false;

	// Create a XML Document
	TiXmlDocument doc;
	doc.InsertEndChild(TiXmlDeclaration( "1.0", "", "" ));
	TiXmlElement neuralNetworkElement( m_XMLNeuralNetworkElement.c_str() );
	neuralNetworkElement.SetAttribute(m_XMLNeuralNetworkType.c_str(),getNeuralNetworkTypeString());

	// Use a specific method to retrieve Neural Network data
	saveToXML(neuralNetworkElement);

	// Put the retrieved data into a document
	doc.InsertEndChild(neuralNetworkElement);

	// Save the document
	TiXmlPrinter printer;
	doc.Accept( &printer );
	//fprintf( stdout, "%s", printer.CStr() );
	fprintf( pSaveFile, "%s", printer.CStr() );

	fclose(pSaveFile);

	return true;
}

bool NeuralNetwork::loadFromFile(const Str &p_sFileName, NeuralNetwork *&p_pReturnedNetwork)
{
	FILE *pLoadFile = TiXmlFOpen(p_sFileName.c_str(),"r");
	if(!pLoadFile)
		return false;

	// We find a network type to create
	TiXmlDocument doc;
	doc.LoadFile(pLoadFile);
	TiXmlElement *pRootElem = doc.RootElement();
	logAssert(pRootElem && pRootElem->Value() == m_XMLNeuralNetworkElement);
	Str sNetworkType = pRootElem->Attribute(m_XMLNeuralNetworkType.c_str());
	p_pReturnedNetwork = getNetworkFromNetworkType(sNetworkType);
	logAssert(p_pReturnedNetwork);

	// We load data using a specific load method
	p_pReturnedNetwork->loadFromXML(*pRootElem);

	fclose(pLoadFile);

	return true;
}

Str NeuralNetwork::getNeuralNetworkTypeString() const
{
	if(m_eNetworkType == NNT_MLP)
		return Str("MLP");
	else
		throw exception("Invalid NN type");
}

NeuralNetwork *NeuralNetwork::getNetworkFromNetworkType(Str p_sNetworkType)
{
	if(p_sNetworkType == "MLP")
		return new MLP;
	else
		throw exception("Invalid NN type string");
}
