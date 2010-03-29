#include "stdafx.h"

const int randomSeed = 25576;

double getRandom01(MTRand *p_pRandomGenerator)
{
	static MTRand randomGenerator(randomSeed);

	if(p_pRandomGenerator != NULL)
		return p_pRandomGenerator->randExc();
	else
		return randomGenerator.randExc();
}

void saveDoubleVectorToXML(const vector<double>&p_vecToConvert, TiXmlElement &p_XML, Str p_sNameToSave,vector< pair<double,double> > *p_vecMinMaxInData)
{
	Str vectorString = getDoubleVectorXMLString(p_vecToConvert,p_vecMinMaxInData);
	TiXmlElement elementToXML(p_sNameToSave.c_str());
	TiXmlText valueToSave(vectorString.c_str());
	elementToXML.InsertEndChild(valueToSave);
	p_XML.InsertEndChild(elementToXML);
}

Str getDoubleVectorXMLString(const vector<double>&p_vecToConvert,vector< pair<double,double> > *p_vecMinMaxInData)
{
	size_t uSize = (unsigned) p_vecToConvert.size();
	char *sBuffer = new char[17 * uSize];
	char *sPointer=sBuffer;
	for(size_t uIndex=0;uIndex<uSize;++uIndex)
	{
		if(p_vecMinMaxInData != NULL && p_vecMinMaxInData->size() != 0)
			sprintf(sPointer,"%g",((p_vecToConvert[uIndex]+1.0)/2)*(p_vecMinMaxInData->at(uIndex).second-p_vecMinMaxInData->at(uIndex).first) - p_vecMinMaxInData->at(uIndex).first);
		else
			sprintf(sPointer,"%g",p_vecToConvert[uIndex]);

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
	const TiXmlElement *pElementToLoad = p_XML.FirstChildElement(p_sNameToLoad.c_str());
	logAssert(pElementToLoad);
	Str sConnectedValues = pElementToLoad->GetText();
	setDoubleVectorXMLString(p_vecToConvert,sConnectedValues);
}

void setDoubleVectorXMLString(vector<double>&p_vecToConvert, const Str &p_sConnectedValues)
{
	p_vecToConvert.clear();
	//char *sDuplicate = strdup(p_sConnectedValues.c_str());
	const char *sPointer = p_sConnectedValues.c_str(); //sDuplicate;

	while(((int)sPointer != 1) && (*sPointer != 0))
	{
		double dNewValue;
		sscanf(sPointer,"%lf",&dNewValue);
		p_vecToConvert.push_back(dNewValue);
		sPointer = strchr(sPointer,cDivider)+1;
	}
}

Str makeDoubleVectorString(vector< vector<double> > *p_vecResultsErrors,unsigned p_uOutputIndex)
{
	if(p_vecResultsErrors == NULL)
		return "";

	Str sResult("\t( %f",p_vecResultsErrors->at(0).at(p_uOutputIndex));
	for(unsigned a=1;a<p_vecResultsErrors->size();++a)
	{
		sResult.format("%s , %f",sResult.c_str(),p_vecResultsErrors->at(a).at(p_uOutputIndex));
	}

	sResult += " )";
	return sResult;
}

void printVectorDifferenceInfoFromVectors(const vector<double> &p_vecMaxAbsoluteErrors,const vector<double> &p_vecMeanAbsoluteErrors,InputTestSet::DifferenceStatisticsType p_eDifferenceType
										  ,vector< vector<double> > *p_vecResultsMaxAbsoluteErrors,vector< vector<double> > *p_vecResultsMeanAbsoluteErrors)
{
	logTextParams(Logging::LT_INFORMATION,"Differences between %s and %s"
		, (p_eDifferenceType == InputTestSet::DST_GPU_AND_CPU ? "GPU" : "Correct")
		, (p_eDifferenceType == InputTestSet::DST_CORRECT_AND_GPU ? "GPU" : "CPU"));
	for(unsigned uOutputIndex=0;uOutputIndex<p_vecMaxAbsoluteErrors.size();++uOutputIndex)
	{
		Str sMax = makeDoubleVectorString(p_vecResultsMaxAbsoluteErrors,uOutputIndex);
		Str sMean = makeDoubleVectorString(p_vecResultsMeanAbsoluteErrors,uOutputIndex);

		logTextParams(Logging::LT_INFORMATION,"Output %d/%d:\tMAX:\t%f%s\tMEAN:\t%f%s",uOutputIndex+1
			,p_vecMaxAbsoluteErrors.size(),p_vecMaxAbsoluteErrors[uOutputIndex],sMax.c_str(),p_vecMeanAbsoluteErrors[uOutputIndex],sMean.c_str());
	}
}