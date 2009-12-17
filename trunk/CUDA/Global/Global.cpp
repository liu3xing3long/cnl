#include "stdafx.h"

const char cDivider = ';';
const int randomSeed = 45632;

double getRandom01(MTRand *p_pRandomGenerator)
{
	static MTRand randomGenerator(randomSeed);

	if(p_pRandomGenerator != NULL)
		return p_pRandomGenerator->randExc();
	else
		return randomGenerator.randExc();
}

void saveDoubleVectorToXML(const vector<double>&p_vecToConvert, TiXmlElement &p_XML, Str p_sNameToSave)
{
	Str vectorString = getDoubleVectorXMLString(p_vecToConvert);
	TiXmlElement elementToXML(p_sNameToSave.c_str());
	TiXmlText valueToSave(vectorString.c_str());
	elementToXML.InsertEndChild(valueToSave);
	p_XML.InsertEndChild(elementToXML);
}

Str getDoubleVectorXMLString(const vector<double>&p_vecToConvert)
{
	size_t uSize = (unsigned) p_vecToConvert.size();
	char *sBuffer = new char[17 * uSize];
	char *sPointer=sBuffer;
	for(size_t uIndex=0;uIndex<uSize;++uIndex)
	{
		sprintf(sPointer,"%lf",p_vecToConvert[uIndex]);
		sPointer += strlen(sPointer);
		*sPointer = cDivider;
		++sPointer;
	}
	*sPointer = 0;
	
	Str sToReturn(sBuffer);
	delete []sBuffer;
	return sToReturn;
}

void loadDoubleVectorFromXML(vector<double>&p_vecToConvert, const TiXmlElement &p_XML, Str p_sNameToLoad)
{
	const TiXmlElement *pElementToLoad = p_XML.FirstChildElement(p_sNameToLoad);
	logAssert(pElementToLoad);
	Str sConnectedValues = pElementToLoad->GetText();
	setDoubleVectorXMLString(p_vecToConvert,sConnectedValues);
}

void setDoubleVectorXMLString(vector<double>&p_vecToConvert, const Str &p_sConnectedValues)
{
	p_vecToConvert.clear();
	char *sDuplicate = strdup(p_sConnectedValues.c_str());
	char *sPointer = sDuplicate;

	while(*sPointer != 0)
	{
		double dNewValue;
		sscanf(sPointer,"%lf",&dNewValue);
		p_vecToConvert.push_back(dNewValue);
		sPointer = strchr(sPointer,cDivider)+1;
	}
}
