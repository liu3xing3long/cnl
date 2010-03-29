#pragma once

double getRandom01(MTRand *p_pRandomGenerator);

void saveDoubleVectorToXML(const vector<double>&p_vecToConvert, TiXmlElement &p_XML, Str p_sNameToSave,vector< pair<double,double> > *p_vecMinMaxInData = NULL);
Str getDoubleVectorXMLString(const vector<double>&p_vecToConvert,vector< pair<double,double> > *p_vecMinMaxInData);

void loadDoubleVectorFromXML(vector<double>&p_vecToConvert, const TiXmlElement &p_XML, Str p_sNameToLoad);
void setDoubleVectorXMLString(vector<double>&p_vecToConvert, const Str &p_sConnectedValues);

Str makeDoubleVectorString(vector< vector<double> > *p_vecResultsErrors,unsigned p_uOutputIndex);
void printVectorDifferenceInfoFromVectors(const vector<double> &p_vecMaxAbsoluteErrors,const vector<double> &p_vecMeanAbsoluteErrors,InputTestSet::DifferenceStatisticsType p_eDifferenceType
										  ,vector< vector<double> > *p_vecResultsMaxAbsoluteErrors = NULL,vector< vector<double> > *p_vecResultsMeanAbsoluteErrors = NULL);
