#pragma once

double getRandom01(MTRand *p_pRandomGenerator);

void saveDoubleVectorToXML(const vector<double>&p_vecToConvert, TiXmlElement &p_XML, Str p_sNameToSave);
Str getDoubleVectorXMLString(const vector<double>&p_vecToConvert);

void loadDoubleVectorFromXML(vector<double>&p_vecToConvert, const TiXmlElement &p_XML, Str p_sNameToLoad);
void setDoubleVectorXMLString(vector<double>&p_vecToConvert, const Str &p_sConnectedValues);