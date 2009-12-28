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

void InputTest::saveDoubleTestVectorToXML(const vector<double> &p_vecDoubleValues,TiXmlElement &p_XML,Str p_sNameToSave,bool p_bOutputAttribute) const
{
	Str sTextToWrite;
	const vector<AttributeMapping> &vecAttributeMappings = m_pParentTestSet->m_vecAttributeMappings;
	unsigned uAttributesMappingSize = vecAttributeMappings.size();
	for(unsigned uAttributeIndex = 0;uAttributeIndex < uAttributesMappingSize;++uAttributeIndex)
	{
		const AttributeMapping &attributeMappingData = vecAttributeMappings[uAttributeIndex];
		if(attributeMappingData.isOutputAttribute() != p_bOutputAttribute)
			continue; // we don't want this attribute data

		Str sToAdd;
		if(!sTextToWrite.empty())
			sToAdd.format("%c",cDivider);

		int iFirstAttributeInStructure = attributeMappingData.getFirstAttributeInStructure();

		if(attributeMappingData.isLiteralAttribute())
		{
			unsigned uAttributeValuesCount = attributeMappingData.getAttributeValuesCount();
			if(uAttributeValuesCount == 2)
			{
				double dValue = p_vecDoubleValues[iFirstAttributeInStructure];
				int iIndexChosenValue = ( (dValue >= (dMinNeuralNetworkValue + dMaxNeuralNetworkValue)/2.0) ? 1 : 0);
				sToAdd += Str("%c%lf%c%c%s%c",XML_CLASSIFICATION_CHAR_START,dValue
					,XML_CLASSIFICATION_CHAR_END,XML_CLASSIFICATION_CHAR_START
					,attributeMappingData.getAttributeValue(iIndexChosenValue),XML_CLASSIFICATION_CHAR_END);
			}
			else
			{
				double dMaxFoundValue = p_vecDoubleValues[iFirstAttributeInStructure];
				int iMaxFoundValueIndex = 0;
				sToAdd.format("%s%c",sToAdd.c_str(),XML_CLASSIFICATION_CHAR_START);
				for(unsigned uPossibleAttributeIndex = 0;uPossibleAttributeIndex < uAttributeValuesCount;++uPossibleAttributeIndex)
				{
					double dThisIndexValue = p_vecDoubleValues[iFirstAttributeInStructure + uPossibleAttributeIndex];
					if(dThisIndexValue > dMaxFoundValue)
					{
						dMaxFoundValue = dThisIndexValue;
						iMaxFoundValueIndex = uPossibleAttributeIndex;
					}

					char cToAddAfterAttributeValue = ((uPossibleAttributeIndex == uAttributeValuesCount-1) ? XML_CLASSIFICATION_CHAR_END : cDivider);
					sToAdd += Str("%lf%c",dThisIndexValue,cToAddAfterAttributeValue);
				}
				sToAdd += Str("%c%s%c",XML_CLASSIFICATION_CHAR_START,attributeMappingData.getAttributeValue(iMaxFoundValueIndex),XML_CLASSIFICATION_CHAR_END);
			}
		}
		else
		{

		}

		sTextToWrite += sToAdd;
	}

	TiXmlElement elementToXML(p_sNameToSave.c_str());
	TiXmlText valueToSave(sTextToWrite.c_str());
	elementToXML.InsertEndChild(valueToSave);
	p_XML.InsertEndChild(elementToXML);
}

void InputTest::saveToXML(TiXmlElement &p_XML) const
{
	// we save Inputs
	saveDoubleTestVectorToXML(m_vecInputs,p_XML,m_XMLInputs,false);
	// we save Correct Outputs
	saveDoubleTestVectorToXML(m_vecCorrectOutputs,p_XML,m_XMLCorrectOutputs,true);
	// we save Network Outputs
	saveDoubleTestVectorToXML(m_vecNetworkOutputs,p_XML,m_XMLNetworkOutputs,true);
	// we save Network Outputs GPU
	saveDoubleTestVectorToXML(m_vecNetworkOutputsGPU,p_XML,m_XMLNetworkOutputsGPU,true);
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
