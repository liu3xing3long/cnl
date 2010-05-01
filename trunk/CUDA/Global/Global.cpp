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

Str makeDoubleVectorString(const vector<double>& p_vecDifferences)
{
	logAssert(p_vecDifferences.size());

	Str sResult;

	if(p_vecDifferences.size() > 1)
	{
		double dSum = p_vecDifferences[0];
		sResult = Str("\t( %f",p_vecDifferences[0]);
		for(unsigned a=1;a<p_vecDifferences.size();++a)
		{
			dSum += p_vecDifferences[a];
			sResult.format("%s , %f",sResult.c_str(),p_vecDifferences[a]);
		}

		sResult = Str("%f ",dSum / p_vecDifferences.size()) + sResult + " )";
	}
	else
	{
		sResult.format("%f",p_vecDifferences[0]);
	}

	return sResult;
}

void printVectorDifferenceInfoFromVectors(const vector<InputTestSet::AttributeLoggingData> &p_vecDifferencesData, InputTestSet::DifferenceStatisticsType p_eDifferenceType, unsigned int p_uiMiliseconds )
{
	Str sToLogFirst("Differences between %s and %s , %d tests"
		, (p_eDifferenceType == InputTestSet::DST_GPU_AND_CPU ? "GPU" : "Correct")
		, (p_eDifferenceType == InputTestSet::DST_CORRECT_AND_GPU ? "GPU" : "CPU"), p_vecDifferencesData[0].m_uiNumTests);
	if(p_uiMiliseconds != -1)
		sToLogFirst += Str(", %d milliseconds",p_uiMiliseconds);
	logText(Logging::LT_INFORMATION,sToLogFirst.c_str());

	for(unsigned uOutputIndex=0;uOutputIndex<p_vecDifferencesData.size();++uOutputIndex)
	{
		const InputTestSet::AttributeLoggingData &loggingData = p_vecDifferencesData[uOutputIndex];

		Str sToLog("Output %d/%d (%s), %s:\t",uOutputIndex+1,p_vecDifferencesData.size()
			,((loggingData.m_sColumnName != "") ? loggingData.m_sColumnName.c_str() : "unnamed")
			,((loggingData.m_bLiteralAttribute) ? "Literal" : "Continuous"));

		if(loggingData.m_bLiteralAttribute)
		{
			logTextParams(Logging::LT_INFORMATION,"%sPercent different results:\t%s",sToLog.c_str()
				,makeDoubleVectorString(loggingData.m_vecLiteralErrors).c_str());
		}
		else
		{
			logTextParams(Logging::LT_INFORMATION,"%sMAX:\t%s\tMEAN:\t%s",sToLog.c_str()
				,makeDoubleVectorString(loggingData.m_vecMaxErrors).c_str(),makeDoubleVectorString(loggingData.m_vecMeanErrors).c_str());
		}
	}
}
