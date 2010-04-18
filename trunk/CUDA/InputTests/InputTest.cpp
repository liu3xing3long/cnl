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

	if(p_vecDoubleValues.size() > 0)
	{
		const vector<AttributeMapping> &vecAttributeMappings = m_pParentTestSet->getAttributeMappings();
		size_t uAttributesMappingSize = vecAttributeMappings.size();
		for(unsigned uAttributeIndex = 0;uAttributeIndex < uAttributesMappingSize;++uAttributeIndex)
		{
			const AttributeMapping &attributeMappingData = vecAttributeMappings[uAttributeIndex];
			if(attributeMappingData.isOutputAttribute() != p_bOutputAttribute)
				continue; // we don't want this attribute data

			Str sToAdd;
			if(!sTextToWrite.empty())
				sToAdd.format("%c",cDivider);

			int iFirstAttributeInStructure = attributeMappingData.getFirstAttributeInStructure();
			double dFirstValue = p_vecDoubleValues[iFirstAttributeInStructure];

			if(attributeMappingData.isLiteralAttribute())
			{
				unsigned uAttributeValuesCount = attributeMappingData.getAttributeValuesCount();
				if(uAttributeValuesCount == 2)
				{
					int iIndexChosenValue = ( (dFirstValue >= (dMinNeuralNetworkValue + dMaxNeuralNetworkValue)/2.0) ? 1 : 0);
					sToAdd += Str("%c%g%c%c%s%c",XML_CLASSIFICATION_CHAR_START,dFirstValue
						,XML_CLASSIFICATION_CHAR_END,XML_CLASSIFICATION_CHAR_START
						,attributeMappingData.getAttributeValue(iIndexChosenValue).c_str(),XML_CLASSIFICATION_CHAR_END);
				}
				else
				{
					double dMaxFoundValue = dFirstValue;
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
						sToAdd += Str("%g%c",dThisIndexValue,cToAddAfterAttributeValue);
					}
					sToAdd += Str("%c%s%c",XML_CLASSIFICATION_CHAR_START,attributeMappingData.getAttributeValue(iMaxFoundValueIndex).c_str(),XML_CLASSIFICATION_CHAR_END);
				}
			}
			else
			{
				double dMin = attributeMappingData.getMinValue();
				double dMax = attributeMappingData.getMaxValue();
				double dUnnormalizedValue = (dFirstValue-dMinNeuralNetworkValue) / (dMaxNeuralNetworkValue-dMinNeuralNetworkValue) * (dMax-dMin) + dMin;
				sToAdd += Str("%g",dUnnormalizedValue);
			}

			sTextToWrite += sToAdd;
		}
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

void InputTest::loadDoubleTestVectorFromXML(vector<double> &p_vecDoubleValues,const TiXmlElement &p_XML,Str p_sNameToLoad)
{
	const TiXmlElement *pElementToLoad = p_XML.FirstChildElement(p_sNameToLoad.c_str());
	logAssert(pElementToLoad);
	Str sConnectedValues = pElementToLoad->GetText();

	// We remove unneeded characters
	size_t iPos;
	while((iPos = sConnectedValues.find(XML_CLASSIFICATION_CHAR_START)) != Str::npos)
	{
		size_t iPosEnd = sConnectedValues.find(XML_CLASSIFICATION_CHAR_END);
		if(iPosEnd == Str::npos || iPosEnd < iPos)
		{
			logTextParams(Logging::LT_ERROR,"Character \'%c\' is before character \'%c\' in string %s",XML_CLASSIFICATION_CHAR_END,XML_CLASSIFICATION_CHAR_START,sConnectedValues.c_str());
			return;
		}

		if(iPosEnd == sConnectedValues.size()-1 || sConnectedValues[iPosEnd+1] != XML_CLASSIFICATION_CHAR_START)
		{
			logTextParams(Logging::LT_ERROR,"A first character after character \'%c\' on pos. %d is not \'%c\' (in string %s)",XML_CLASSIFICATION_CHAR_END,iPosEnd,XML_CLASSIFICATION_CHAR_START,sConnectedValues.c_str());
			return;
		}

		size_t iPosEndAll = sConnectedValues.find(XML_CLASSIFICATION_CHAR_END,iPosEnd+2);

		if(iPosEndAll == Str::npos)
		{
			logTextParams(Logging::LT_ERROR,"Did not find \'%c\' after position %d in string %s",XML_CLASSIFICATION_CHAR_END,iPosEnd+1,sConnectedValues.c_str());
			return;
		}

		sConnectedValues = sConnectedValues.substring(0,iPos) + sConnectedValues.substring(iPos+1,iPosEnd-iPos-1) + sConnectedValues.substring(iPosEndAll+1);
	}

	setDoubleVectorXMLString(p_vecDoubleValues,sConnectedValues);
}

void InputTest::loadFromXML(const TiXmlElement &p_XML)
{
	// we load Inputs
	loadDoubleTestVectorFromXML(m_vecInputs,p_XML,m_XMLInputs);
	// we load Correct Outputs
	loadDoubleTestVectorFromXML(m_vecCorrectOutputs,p_XML,m_XMLCorrectOutputs);
	// we load Network Outputs
	loadDoubleTestVectorFromXML(m_vecNetworkOutputs,p_XML,m_XMLNetworkOutputs);
	// we load Network Outputs GPU
	loadDoubleTestVectorFromXML(m_vecNetworkOutputsGPU,p_XML,m_XMLNetworkOutputsGPU);
}

void InputTest::setOutputs(const vector<double> &p_vecLayerOutput)
{
	m_vecNetworkOutputs.assign(p_vecLayerOutput.begin(),p_vecLayerOutput.end());
}

const vector<double>& InputTest::getInputs() const
{
	return m_vecInputs;
}

double InputTest::getNetworkOutput(unsigned int p_uiIndex) const
{
	return m_vecNetworkOutputs[p_uiIndex];
}

double InputTest::getCorrectOutput(unsigned int p_uiIndex) const
{
	return m_vecCorrectOutputs[p_uiIndex];
}
