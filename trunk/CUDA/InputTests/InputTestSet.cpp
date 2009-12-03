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
	return m_vecTests.size();
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
	return m_vecTests[0].m_vecInputs.size();
}

unsigned InputTestSet::getOutputCount() const
{
	return m_vecTests[0].m_vecCorrectOutputs.size();
}

bool InputTestSet::getDifferencesStatistics(vector<double> &p_vecMaxAbsoluteErrors,vector<double> &p_vecMeanAbsoluteErrors, DifferenceStatisticsType p_eDifferenceType) const
{
	// JRTODO - jesli nie ma albo wynikow GPU, albo CPU, to zwroc false
	if(m_vecTests.size() == 0)
	{
		logText(Logging::LT_INFORMATION, "There are no tests to check CPU/GPU statistics");
		return false;
	}

	unsigned uOutputsSize = m_vecOutColumnNames.size();
	p_vecMaxAbsoluteErrors.assign(uOutputsSize,0);
	p_vecMeanAbsoluteErrors.assign(uOutputsSize,0);

	unsigned uNumTests = m_vecTests.size();

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

InputTestSet::InputTestSet(const InputTestSet &p_TestSet)
{
	// we don't copy md_pTestSetMemory, because we unallocate this memory in destructor
	m_vecTests.assign(p_TestSet.m_vecTests.begin(),p_TestSet.m_vecTests.end());
	m_vecInColumnNames.assign(p_TestSet.m_vecInColumnNames.begin(),p_TestSet.m_vecInColumnNames.end());
	m_vecOutColumnNames.assign(p_TestSet.m_vecOutColumnNames.begin(),p_TestSet.m_vecOutColumnNames.end());
	md_pTestSetMemory = NULL;
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
	md_pTestSetMemory = NULL;
}

InputTestSet::~InputTestSet()
{
	// JRTODO deallocate md_pTestSetMemory
}

void InputTestSet::cleanObject()
{
	m_vecTests.clear();
	m_vecInColumnNames.clear();
	m_vecOutColumnNames.clear();
}