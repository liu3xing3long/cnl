#include "stdafx.h"

const Str m_XMLTestSetElement("TestSet");
const Str m_XMLInColumns("InColumns");
const Str m_XMLInColumnElement("InColumnElement");
const Str m_XMLOutColumns("OutColumns");
const Str m_XMLOutColumnElement("OutColumnElement");
const Str m_XMLTests("Tests");
const Str m_XMLTest("Test");

unsigned InputTestSet::getTestCount() const
{
	return (unsigned) m_vecTests.size();
}

const InputTest& InputTestSet::getTest(int p_iIndex) const
{
	return m_vecTests[p_iIndex];
}

InputTest& InputTestSet::getTest(int p_iIndex)
{
	return m_vecTests[p_iIndex];
}

unsigned InputTestSet::getInputCount() const
{
	return(unsigned)  m_vecTests[0].m_vecInputs.size();
}

unsigned InputTestSet::getOutputCount() const
{
	return (unsigned) m_vecTests[0].m_vecCorrectOutputs.size();
}

bool InputTestSet::getDifferencesStatistics(vector<double> &p_vecMaxAbsoluteErrors,vector<double> &p_vecMeanAbsoluteErrors, DifferenceStatisticsType p_eDifferenceType) const
{
	// JRTODO - jesli nie ma albo wynikow GPU, albo CPU, to zwroc false
	if(m_vecTests.size() == 0)
	{
		logText(Logging::LT_INFORMATION, "There are no tests to check CPU/GPU statistics");
		return false;
	}

	size_t uOutputsSize = getOutputCount();
	p_vecMaxAbsoluteErrors.assign(uOutputsSize,0);
	p_vecMeanAbsoluteErrors.assign(uOutputsSize,0);

	size_t uNumTests = m_vecTests.size();

	for(unsigned uTestIndex=0;uTestIndex<uNumTests;++uTestIndex)
	{
		const InputTest &testNow = m_vecTests[uTestIndex];
		const vector<double> &vecToCompare1 = (p_eDifferenceType == DST_GPU_AND_CPU ? testNow.m_vecNetworkOutputsGPU : testNow.m_vecCorrectOutputs);
		const vector<double> &vecToCompare2 = (p_eDifferenceType == DST_CORRECT_AND_GPU ? testNow.m_vecNetworkOutputsGPU : testNow.m_vecNetworkOutputs);

		for(unsigned uOutputIndex=0;uOutputIndex<uOutputsSize;++uOutputIndex)
		{
			double dAbsoluteError = fabs(vecToCompare1[uOutputIndex] - vecToCompare2[uOutputIndex]);
			p_vecMaxAbsoluteErrors[uOutputIndex] = max(dAbsoluteError,p_vecMaxAbsoluteErrors[uOutputIndex]);
			p_vecMeanAbsoluteErrors[uOutputIndex] += dAbsoluteError / uNumTests;
			// JRTODO - proportional error
			//double dProportionalError = fabs(testNow.m_vecNetworkOutputsGPU[uOutputIndex] - vecToCompare[uOutputIndex]);
		}
	}

	return true;
}

void InputTestSet::randomizeTests(MTRand *p_pRandomGenerator)
{
	for(unsigned uTestIndex=0;uTestIndex<m_vecTests.size();++uTestIndex)
	{
		m_vecTests[uTestIndex].randomizeTest(p_pRandomGenerator);
	}
}

void InputTestSet::setOutputFunction(const vector< pair<double,double> > &p_vecMinMax, void (*p_fTestingFunction)(const vector<double> &p_vecInputParameters,vector<double> &p_vecOutputParameters),MTRand *p_pRandomGenerator)
{
	for(unsigned uTestIndex=0;uTestIndex<m_vecTests.size();++uTestIndex)
	{
		m_vecTests[uTestIndex].setOutputFunction(p_vecMinMax,p_fTestingFunction,p_pRandomGenerator);
	}
	normalizeTests();
}

bool InputTestSet::saveToFile(Str p_sFileName) const
{
	FILE *pSaveFile = TiXmlFOpen(p_sFileName.c_str(),"wb");
	if(!pSaveFile)
		return false;

	// Create a XML Document
	TiXmlDocument doc;
	doc.InsertEndChild(TiXmlDeclaration( "1.0", "", "" ));
	TiXmlElement testSetElement( m_XMLTestSetElement.c_str() );

	// Save all data
	saveToXML(testSetElement);

	// Put the retrieved data into a document
	doc.InsertEndChild(testSetElement);

	// Save the document
	TiXmlPrinter printer;
	doc.Accept( &printer );
	fprintf( stdout, "%s", printer.CStr() );
	fprintf( pSaveFile, "%s", printer.CStr() );

	fclose(pSaveFile);

	return true;
}

void InputTestSet::saveToXML(TiXmlElement &p_XML) const
{
	// we save in column names
	TiXmlElement inColumnElements(m_XMLInColumns.c_str());
	for(unsigned uColumnIndex = 0;uColumnIndex < m_vecInColumnNames.size();++uColumnIndex)
	{
		TiXmlElement newInColumnElement(m_XMLInColumnElement.c_str());
		TiXmlText newInColumnElementValue(m_vecInColumnNames[uColumnIndex].c_str());
		newInColumnElement.InsertEndChild(newInColumnElementValue);
		inColumnElements.InsertEndChild(newInColumnElement);
	}
	p_XML.InsertEndChild(inColumnElements);

	// we save out column names
	TiXmlElement outColumnElements(m_XMLOutColumns.c_str());
	for(unsigned uColumnIndex = 0;uColumnIndex < m_vecOutColumnNames.size();++uColumnIndex)
	{
		TiXmlElement newOutColumnElement(m_XMLOutColumnElement.c_str());
		TiXmlText newOutColumnElementValue(m_vecOutColumnNames[uColumnIndex].c_str());
		newOutColumnElement.InsertEndChild(newOutColumnElementValue);
		outColumnElements.InsertEndChild(newOutColumnElement);
	}
	p_XML.InsertEndChild(outColumnElements);

	// we save all tests
	TiXmlElement testsElement(m_XMLTests.c_str());
	for(unsigned uTestIndex = 0;uTestIndex < m_vecTests.size();++uTestIndex)
	{
		TiXmlElement newTestElement(m_XMLTest.c_str());
		m_vecTests[uTestIndex].saveToXML(newTestElement);
		testsElement.InsertEndChild(newTestElement);
	}
	p_XML.InsertEndChild(testsElement);
}

bool InputTestSet::loadFromFile(Str p_sFileName)
{
	cleanObject();
	FILE *pLoadFile = TiXmlFOpen(p_sFileName.c_str(),"r");
	if(!pLoadFile)
		return false;

	// We find a network type to create
	TiXmlDocument doc;
	doc.LoadFile(pLoadFile);
	TiXmlElement *pRootElem = doc.RootElement();
	logAssert(pRootElem && pRootElem->Value() == m_XMLTestSetElement);

	loadFromXML(*pRootElem);

	fclose(pLoadFile);

	normalizeTests();

	return true;
}

void InputTestSet::loadFromXML(const TiXmlElement &p_XML)
{
	// we load in column names
	const TiXmlElement *pInColumnElements = p_XML.FirstChildElement(m_XMLInColumns.c_str());
	logAssert(pInColumnElements);
	const TiXmlElement *pInColumnElement = pInColumnElements->FirstChildElement(m_XMLInColumnElement.c_str());
	while(pInColumnElement)
	{
		m_vecInColumnNames.push_back(pInColumnElement->GetText());
		pInColumnElement = pInColumnElement->NextSiblingElement(m_XMLInColumnElement.c_str());
	}

	// we load in column names
	const TiXmlElement *pOutColumnElements = p_XML.FirstChildElement(m_XMLOutColumns.c_str());
	logAssert(pOutColumnElements);
	const TiXmlElement *pOutColumnElement = pOutColumnElements->FirstChildElement(m_XMLOutColumnElement.c_str());
	while(pOutColumnElement)
	{
		m_vecOutColumnNames.push_back(pOutColumnElement->GetText());
		pOutColumnElement = pOutColumnElement->NextSiblingElement(m_XMLOutColumnElement.c_str());
	}

	// we load all tests
	const TiXmlElement *pXMLTests = p_XML.FirstChildElement(m_XMLTests.c_str());
	logAssert(pXMLTests);
	const TiXmlElement *pXMLTest = pXMLTests->FirstChildElement(m_XMLTest.c_str());
	while(pXMLTest)
	{
		InputTest newTest(this,0,0);
		newTest.loadFromXML(*pXMLTest);
		m_vecTests.push_back(newTest);
		pXMLTest = pXMLTest->NextSiblingElement(m_XMLTest.c_str());
	}
}

bool InputTestSet::loadElementsFromCSVFile(char p_cSeparator, FILE *p_pLoadFile, vector<vector<Str>> &p_vecElements)
{
	const int iStringLen = 100000;
	char sLoadedLine[iStringLen];
	// lines containing divided values
	int iLineNumber = 0;
	while(fgets(sLoadedLine,iStringLen,p_pLoadFile))
	{
		iLineNumber++;
		int iLineLen = strlen(sLoadedLine);
		if(iLineLen < 3)
			continue;
		if(iLineLen >= iStringLen-2)
		{
			logTextParams(Logging::LT_ERROR,"Line %d of file %s is longer than %d characters",iLineNumber,p_sFileName.c_str(),iStringLen);
			return false;
		}

		p_vecElements.push_back(Str(sLoadedLine).split(p_cSeparator));
	}

	if(p_vecElements.size() == 0)
	{
		logTextParams(Logging::LT_ERROR,"No elements loaded from file %s",p_sFileName.c_str());
		return false;
	}

	return true;
}

void InputTestSet::removeIncorrectCSVElements(bool p_bContainsColumnNames, vector<vector<Str>> &p_vecElements)
{
	int iErasedElements;
	unsigned uVecSize = p_vecElements.size();
	unsigned uColumnsNumber = p_vecElements[0].size();
	for(unsigned uLineIndex = 0;uLineIndex < p_uVecSize;++uLineIndex)
	{
		bool bToErase = false;
		for(unsigned uColumnIndex = 0;uColumnIndex < uColumnsNumber;++uColumnIndex)
		{
			if(p_vecElements[uLineIndex][uColumnIndex] == "?")
			{
				bToErase = true;
				break;
			}
		}

		if(bToErase)
		{
			p_vecElements.erase(p_vecElements.begin() + uLineIndex);
			uLineIndex--;
			uVecSize--;
			++iErasedElements;
		}
	}

	logTextParams(Logging::LT_INFORMATION,"Column names: %s , tests before: %d , tests removed: %d , tests after: %d"
		,(p_bContainsColumnNames ? "true":"false"),uVecSize+iErasedElements,iErasedElements,uVecSize);	
}

bool InputTestSet::checkKindsOfColumnsInCSVFile(const vector< vector<Str> > &p_vecElements, vector<bool> &p_vecIsLiteral)
{
	unsigned uVecSize = p_vecElements.size();
	for(unsigned uLineIndex = 0;uLineIndex < uVecSize;++uLineIndex)
	{
		for(unsigned uColumnIndex = 0;uColumnIndex < uColumnsNumber;++uColumnIndex)
		{
			Str sElement = p_vecElements[uLineIndex][uColumnIndex];
			if(sElement.size() == 0)
			{
				logTextParams(Logging::LT_ERROR,"Line %d, element %d - element is empty",uLineIndex,uColumnIndex);
				return false;
			}

			bool bIsLiteral = false;

			if(sElement[0] == '\"' && sElement[sElement.size()-1] == '\"')
			{
				p_vecElements[uLineIndex][uColumnIndex] = sElement.substring(1,sElement.size()-2);
				bIsLiteral = true;
			}

			unsigned uElemSize = sElement.size();
			for(unsigned uCharIndex = 0;uCharIndex < uElemSize;++uCharIndex)
			{
				if(!(sElement[uCharIndex] >= '0' && sElement[uCharIndex] <= '9') && sElement[uCharIndex] != '.')
					bIsLiteral = true;
			}

			if(bIsLiteral)
			{
				p_vecIsLiteral[uColumnIndex] = true;
			}
		}
	}
	return true;
}

bool InputTestSet::checkColumnIndexCorrectnessInCSVFile(vector<int> &p_vecOutputColumns,vector<int> &p_vecUnusedColumns)
{
	for(unsigned uColumnIndex = 0;uColumnIndex < p_vecOutputColumns.size();++uColumnIndex)
	{
		if(p_vecOutputColumns[uColumnIndex] < 0 || p_vecOutputColumns[uColumnIndex] >= uColumnsNumber)
		{
			logTextParams(Logging::LT_ERROR,"Incorrect column index in p_vecOutputColumns: %d = %d (should be <0,%d>)",uColumnIndex,p_vecOutputColumns[uColumnIndex],uColumnsNumber-1);
			return false;
		}

		if(find( p_vecUnusedColumns.begin(),p_vecUnusedColumns.end(),p_vecOutputColumns[uColumnIndex]) != p_vecUnusedColumns.end())
		{
			logTextParams(Logging::LT_ERROR,"Column %d exists in both p_vecOutputColumns and p_vecUnusedColumns",p_vecOutputColumns[uColumnIndex]);
			return false;
		}
	}
	for(unsigned uColumnIndex = 0;uColumnIndex < p_vecUnusedColumns.size();++uColumnIndex)
	{
		if(p_vecUnusedColumns[uColumnIndex] < 0 || p_vecUnusedColumns[uColumnIndex] >= uColumnsNumber)
		{
			logTextParams(Logging::LT_ERROR,"Incorrect column index in p_vecUnusedColumns: %d = %d (should be <0,%d>)",uColumnIndex,p_vecUnusedColumns[uColumnIndex],uColumnsNumber-1);
			return false;
		}
	}
}

bool InputTestSet::getColumnRangesFromCSVFile(const vector<vector<Str>> &p_vecElements, const vector<bool> &p_vecIsLiteral, vector< pair<double,double> > &p_vecMinMaxData, vector<vector<Str>> &p_vecPossibleValuesData)
{
	for(unsigned uColumnIndex = 0;uColumnIndex < p_uColumnsNumber;++uColumnIndex)
	{
		if(p_vecIsLiteral[uColumnIndex])
		{
			vector<Str> vecPossibleValuesThisColumn;
			for(unsigned uLineIndex = 0;uLineIndex < p_uVecSize;++uLineIndex)
			{
				Str sElement = p_vecElements[uLineIndex][uColumnIndex];
				if(find(vecPossibleValuesThisColumn.begin(),vecPossibleValuesThisColumn.end(),sElement) == vecPossibleValuesThisColumn.end())
					vecPossibleValuesThisColumn.push_back(sElement);
			}
			p_vecPossibleValuesData[uColumnIndex] = vecPossibleValuesThisColumn;

			if(vecPossibleValuesThisColumn.size() < 2)
			{
				logTextParams(Logging::LT_ERROR,"Literal column %d has only one value %s",uColumnIndex,vecPossibleValuesThisColumn[0]);
				return false;
			}
		}
		else
		{
			double dMin = p_vecElements[0][uColumnIndex];
			double dMax = p_vecElements[0][uColumnIndex];
			for(unsigned uTestIndex=1;uTestIndex<uTestCount;++uTestIndex)
			{
				dMin = min(dMin,m_vecTests[uTestIndex].m_vecCorrectOutputs[uOutputIndex]);
				dMax = max(dMax,m_vecTests[uTestIndex].m_vecCorrectOutputs[uOutputIndex]);
			}

			m_vecMinMaxOutData.push_back(pair<double,double> (dMin,dMax));

			if(dMin == dMax)
			{
				logTextParams(Logging::LT_ERROR,"Number column %d has only one value %lf",uColumnIndex,dMin);
				return false;
			}
		}
	}
	return true;
}
bool InputTestSet::generateInputColumnsVectorForCSVFile(const vector<int> &p_vecOutputColumns, const vector<int> &p_vecUnusedColumns, unsigned uColumnsNumber, vector<int> &p_vecInputColumns)
{
	for(unsigned uColumnIndex = 0;uColumnIndex < uColumnsNumber;++uColumnIndex)
	{
		if(find( p_vecOutputColumns.begin(),p_vecOutputColumns.end(),uColumnIndex) == p_vecOutputColumns.end()
			&& find( p_vecUnusedColumns.begin(),p_vecUnusedColumns.end(),uColumnIndex) == p_vecUnusedColumns.end())
		{
			p_vecInputColumns.push_back(uColumnIndex);
		}
	}

	if(p_vecInputColumns.size() == 0)
	{
		logTextParams(Logging::LT_ERROR,"There are no input columns...");
		return false;
	}
	return false;
}
bool InputTestSet::checkBasicValidityInCSVFile(vector< vector<Str> > &vecElements)
{
	unsigned uVecSize = vecElements.size();
	if(uVecSize < 2)
	{
		logTextParams(Logging::LT_ERROR,"Too small number of lines: %d",uVecSize);
		return false;
	}

	unsigned uColumnsNumber = vecElements[0].size();
	// All lines need to have the same number of elements
	for(unsigned uLineIndex = 1;uLineIndex < uVecSize;++uLineIndex)
	{
		if(vecElements[uLineIndex].size() != uColumnsNumber)
		{
			logTextParams(Logging::LT_ERROR,"Number of elements in line %d(%d) is different than in line %d(%d)",uLineIndex,vecElements[uLineIndex].size(),0,uColumnsNumber);
			return false;
		}
	}
	return true;
}

// JRTODO - ta i inne metody pomocnicze maja byc static
void InputTestSet::printDataAboutColumns(const vector<int> &p_vecColumnIndexes,Str p_sColumnType,const vector<bool> &p_vecIsLiteral,const vector< pair<double,double> > &p_vecMinMaxData
										 ,const vector< vector<Str> > &p_vecPossibleValuesData,const vector<Str> &p_vecColumnNames)
{
	if(p_vecColumnIndexes.size() != 0)
	{
		for(unsigned uColumnIndex = 0;uColumnIndex < p_vecColumnIndexes.size();++uColumnIndex)
		{
			unsigned uColumnIndexInInput = p_vecColumnIndexes[uColumnIndex];
			Str sLogText;
			if(p_vecColumnNames.size())
					sLogText.format("Columns type %s , column index %d , column in input %d, column name \"%s\":",p_sColumnType.c_str(),uColumnIndex,uColumnIndexInInput,p_vecColumnNames[uColumnIndexInInput]);
				else
					sLogText.format("Columns type %s , column index %d , column in input %d:",p_sColumnType.c_str(),uColumnIndex,uColumnIndexInInput);

			if(p_vecIsLiteral[uColumnIndexInInput])
			{
				sLogText += " elements ";
				for(unsigned uValueIndex = 0;uValueIndex < p_vecPossibleValuesData[uColumnIndexInInput].size();++uValueIndex)
				{
					sLogText += "\"" + p_vecPossibleValuesData[uColumnIndexInInput][uValueIndex]+"\"";
					if(uValueIndex != p_vecPossibleValuesData[uColumnIndexInInput].size() - 1)
						sLogText += " , ";
				}
			}
			else
			{
				sLogText.format("%sMinimum %lf , Maximum %lf",sLogText.c_str(),p_vecMinMaxData[uColumnIndexInInput].first,p_vecMinMaxData[uColumnIndexInInput].second);
			}

			logTextParams(Logging::LT_INFORMATION,sLogText.c_str());
		}	
	}
	else
	{
		logTextParams(Logging::LT_INFORMATION,"Columns type %s: no such columns",p_sColumnType.c_str());
		return false;
	}
}

bool InputTestSet::generateAttributeMappingsAndTestsForCSVFile(const vector<int> &p_vecInputColumns,const vector<int> &p_vecOutputColumns
			,const vector< pair<double,double> > &p_vecMinMaxData,const vector< vector<Str> > &p_vecPossibleValuesData
			,const vector<Str> &p_vecColumnNames,const vector<bool> &p_vecIsLiteral,const vector< vector<Str> > &p_vecElements)
{
	m_vecAttributeMappings.clear();
	m_vecTests.clear();
	unsigned uVecSize = p_vecElements.size();
	for(unsigned uTestIndex=0;uTestIndex<uVecSize;++uTestIndex)
	{
		m_vecTests.push_back(InputTest(this,0,0));
	}

	vector<int> *pVectorsColumnIndices[2] = { p_vecInputColumns , p_vecOutputColumns };
	for(int iVectorIndex = 0;iVectorIndex < 2;++iVectorIndex)
	{
		vector<int> &vecNow = *pVectorsColumnIndices[iVectorIndex];
		bool bIsOutputVector = (iVectorIndex == 1);
		int iElementInStructure = 0;
		for(unsigned uColumnIndex = 0;uColumnIndex < vecNow.size();++uColumnIndex)
		{
			int iColumnIndexInVecElements = vecNow[uColumnIndex];
			Str sColumnName = (p_vecColumnNames.size() != 0 ? p_vecColumnNames[iColumnIndexInVecElements] : "");
			
			// We add an element to m_vecAttributeMappings
			m_vecAttributeMappings.push_back(AttributeMapping(sColumnName,bIsOutputVector,iColumnIndexInVecElements,iElementInStructure);
			AttributeMapping &lastAttributeMapping = m_vecAttributeMappings[m_vecAttributeMappings.size()-1];
			if(p_vecIsLiteral[iColumnIndexInVecElements])
			{
				const vector<Str> &vecPosibleValues = p_vecPossibleValuesData[iColumnIndexInVecElements];
				lastAttributeMapping.setLiteralPossibleValues(vecPosibleValues);

				// if there are only 2 possible values, we make only one input/output. 
				// If more, we have the ame number of inputs/outputs as the number of possible values
				unsigned uPossibleValues = vecPosibleValues.size();
				if(iPossibleValues == 2)
				{
					Str sFirstValue = vecPosibleValues[0];
					Str sSecondValue = vecPosibleValues[1];
					for(unsigned uTestIndex=0;uTestIndex<uVecSize;++uTestIndex)
					{
						vector<double> &vecToAdd = (bIsOutputVector ? m_vecTests[uTestIndex].m_vecCorrectOutputs : m_vecTests[uTestIndex].m_vecInputs);
						Str sValueInVector = p_vecElements[uTestIndex][iColumnIndexInVecElements];
						if(sValueInVector == sFirstValue)
							vecToAdd.push_back(dMinNeuralNetworkValue);
						else if(sValueInVector == sSecondValue)
							vecToAdd.push_back(dMaxNeuralNetworkValue);
						else
						{
							logTextParams(Logging::LT_ERROR,"Unknown value in column %d = %s . Known values are %s and %s"
								,iColumnIndexInVecElements,sValueInVector,sFirstValue,sSecondValue);
							return false;
						}
					}
				}
				else
				{
					for(unsigned uTestIndex=0;uTestIndex<uVecSize;++uTestIndex)
					{
						vector<double> &vecToAdd = (bIsOutputVector ? m_vecTests[uTestIndex].m_vecCorrectOutputs : m_vecTests[uTestIndex].m_vecInputs);
						Str sValueInVector = p_vecElements[uTestIndex][iColumnIndexInVecElements];
						vector<Str>::iterator iter = find(vecPosibleValues.begin(),vecPosibleValues.end(),sValueInVector);
						if(iter != vecPosibleValues.end())
						{
							int iFoundIndex = iter - vecPosibleValues.begin();
							for(unsigned uAddedElement=0;uAddedElement<uPossibleValues;++uAddedElement)
								vecToAdd.push_back(((uAddedElement == iFoundIndex) ? dMaxNeuralNetworkValue : dMinNeuralNetworkValue);
						}
					}
				}
			}
			else
			{
				lastAttributeMapping.setPossibleRange(p_vecMinMaxData[iColumnIndexInVecElements]);
				double dMin = p_vecMinMaxData[iColumnIndexInVecElements].first;
				double dMax = p_vecMinMaxData[iColumnIndexInVecElements].second;

				for(unsigned uTestIndex=0;uTestIndex<uVecSize;++uTestIndex)
				{
					double dNewValue;
					sscanf(p_vecElements[uTestIndex][iColumnIndexInVecElements].c_str(),"%lf",&dNewValue);
					double dNewValueNormalized = (dNewValue - dMin) / (dMax - dMin) * (dMaxNeuralNetworkValue - dMinNeuralNetworkValue) - dMinNeuralNetworkValue;

					vector<double> &vecToAdd = (bIsOutputVector ? m_vecTests[uTestIndex].m_vecCorrectOutputs : m_vecTests[uTestIndex].m_vecInputs);
					vecToAdd.push_back(dNewValueNormalized);
				}
			}
		}
	}
}

bool InputTestSet::loadFromCSVFile(Str p_sFileName,bool p_bContainsColumnNames,char p_cSeparator,const vector<int> &p_vecOutputColumns,const vector<int> &p_vecUnusedColumns)
{
	if(p_vecOutputColumns.size() == 0)
	{
		logTextParams(Logging::LT_ERROR,"No output elements specified in p_vecOutputColumns");
		return false;
	}

	cleanObject();

	FILE *pLoadFile = TiXmlFOpen(p_sFileName.c_str(),"r");
	if(!pLoadFile)
		return false;

	vector< vector<Str> > vecElements;
	if(!loadElementsFromCSVFile(p_cSeparator, pLoadFile, vecElements))
		return false;

	fclose(pLoadFile);

	unsigned uColumnsNumber = vecElements[0].size();

	// If we have column names, we retrieve them
	vector<Str> vecColumnNames;
	if(p_bContainsColumnNames)
	{
		for(unsigned uColumnIndex = 0;uColumnIndex < uColumnsNumber;++uColumnIndex)
		{
			Str sElement = p_vecElements[0][uColumnIndex];
			if(sElement[0] == '\"' && sElement[sElement.size()-1] == '\"')
				sElement = sElement.substring(1,sElement.size()-2);
			vecColumnNames.push_back(sElement);
		}
		vecColumnNames.erase(vecColumnNames.begin());
	}

	if(!checkBasicValidityInCSVFile(vecElements))
		return false;

	// remove elements with '?' - we don't want tests with unknown parameters
	removeIncorrectCSVElements(p_bContainsColumnNames, vecElements, uVecSize, uColumnsNumber, iErasedElements);

	// checking which columns are literal, and which are numbers
	vector<bool> vecIsLiteral(uColumnsNumber,false);
	if(!checkKindsOfColumnsInCSVFile( vecElements, vecIsLiteral))
		return false;

	// We check correctness of p_vecOutputColumns and p_vecUnusedColumns
	if(!checkColumnIndexCorrectnessInCSVFile(p_vecOutputColumns,p_vecUnusedColumns))
		return false;

	// we generate a vector of input column indexes
	vector<int> vecInputColumns;
	if(!generateInputColumnsVectorForCSVFile(p_vecOutputColumns, p_vecUnusedColumn, vecInputColumns))
		return false;

	// We retrieve min/max and values data 
	// Also, we check if there are columns with only one possible value (if yes, it is an error)
	vector< pair<double,double> > vecMinMaxData; // min and max values for numeric data
	vecMinMaxData.resize(uColumnsNumber);
	vector< vector<Str> > vecPossibleValuesData; // all possible values for literal data
	vecPossibleValuesData.resize(uColumnsNumber);
	if(!getColumnRangesFromCSVFile(p_bContainsColumnNames, vecElements, uVecSize, uColumnsNumber, vecIsLiteral, vecMinMaxData,vecPossibleValuesData);
		return false;

	printDataAboutColumns(vecInputColumns,"Input",vecIsLiteral,vecMinMaxData,vecPossibleValuesData,vecColumnNames);
	printDataAboutColumns(p_vecOutputColumns,"Output",vecIsLiteral,vecMinMaxData,vecPossibleValuesData,vecColumnNames);
	printDataAboutColumns(p_vecUnusedColumns,"Unused",vecIsLiteral,vecMinMaxData,vecPossibleValuesData,vecColumnNames);

	// finally fill in m_vecAttributeMappings and m_vecTests attribute
	if(!generateAttributeMappingsAndTestsForCSVFile(vecInputColumns,p_vecOutputColumns,vecMinMaxData,vecPossibleValuesData,vecColumnNames,vecElements))
		return false;

	//normalizeTests();
	return true;
}

InputTestSet::InputTestSet(const InputTestSet &p_TestSet)
{
	m_vecTests.assign(p_TestSet.m_vecTests.begin(),p_TestSet.m_vecTests.end());
	m_vecInColumnNames.assign(p_TestSet.m_vecInColumnNames.begin(),p_TestSet.m_vecInColumnNames.end());
	m_vecOutColumnNames.assign(p_TestSet.m_vecOutColumnNames.begin(),p_TestSet.m_vecOutColumnNames.end());
}

InputTestSet::InputTestSet(unsigned p_uNumberTests,unsigned p_uNumberInputs,unsigned p_uNumberOutputs)
{
	for(unsigned uTestIndex=0;uTestIndex<p_uNumberTests;++uTestIndex)
	{
		m_vecTests.push_back(InputTest(this,p_uNumberInputs,p_uNumberOutputs));
	}

	// m_vecInColumnNames and m_vecOutColumnNames should have a correct length (even though they are empty)
	m_vecInColumnNames.resize(p_uNumberInputs);
	m_vecOutColumnNames.resize(p_uNumberOutputs);
}

InputTestSet::InputTestSet()
{
}

InputTestSet::~InputTestSet()
{
}

void InputTestSet::cleanObject()
{
	m_vecTests.clear();
	m_vecInColumnNames.clear();
	m_vecOutColumnNames.clear();
	m_vecMinMaxInData.clear();
	m_vecMinMaxOutData.clear();
}

/*
void InputTestSet::normalizeTests()
{
	unsigned uTestCount = getTestCount();
	
	m_vecMinMaxInData.clear();
	unsigned uInputCount = getInputCount();
	for(unsigned uInputIndex=0;uInputIndex<uInputCount;++uInputIndex)
	{
		double dMin = m_vecTests[0].m_vecInputs[uInputIndex];
		double dMax = m_vecTests[0].m_vecInputs[uInputIndex];
		for(unsigned uTestIndex=0;uTestIndex<uTestCount;++uTestIndex)
		{
			dMin = min(dMin,m_vecTests[uTestIndex].m_vecInputs[uInputIndex]);
			dMax = max(dMax,m_vecTests[uTestIndex].m_vecInputs[uInputIndex]);
		}

		m_vecMinMaxInData.push_back(pair<double,double> (dMin,dMax));

		for(unsigned uTestIndex=0;uTestIndex<uTestCount;++uTestIndex)
		{
			m_vecTests[uTestIndex].m_vecInputs[uInputIndex] = (m_vecTests[uTestIndex].m_vecInputs[uInputIndex] - dMin) / (dMax-dMin) * 2.0 - 1;
		}
	}

	m_vecMinMaxOutData.clear();
	unsigned uOutputCount = getOutputCount();
	for(unsigned uOutputIndex=0;uOutputIndex<uOutputCount;++uOutputIndex)
	{
		double dMin = m_vecTests[0].m_vecCorrectOutputs[uOutputIndex];
		double dMax = m_vecTests[0].m_vecCorrectOutputs[uOutputIndex];
		for(unsigned uTestIndex=0;uTestIndex<uTestCount;++uTestIndex)
		{
			dMin = min(dMin,m_vecTests[uTestIndex].m_vecCorrectOutputs[uOutputIndex]);
			dMax = max(dMax,m_vecTests[uTestIndex].m_vecCorrectOutputs[uOutputIndex]);
		}

		m_vecMinMaxOutData.push_back(pair<double,double> (dMin,dMax));

		for(unsigned uTestIndex=0;uTestIndex<uTestCount;++uTestIndex)
		{
			m_vecTests[uTestIndex].m_vecCorrectOutputs[uOutputIndex] = (m_vecTests[uTestIndex].m_vecCorrectOutputs[uOutputIndex] - dMin) / (dMax-dMin) * 2.0 - 1;
		}
	}
}
*/