#pragma once

class InputTestSet
{
public:

	enum DifferenceStatisticsType
	{
		DST_CORRECT_AND_CPU,
		DST_CORRECT_AND_GPU,
		DST_GPU_AND_CPU
	};

	struct AttributeLoggingData
	{
		Str m_sColumnName;
		bool m_bLiteralAttribute;
		size_t m_uiNumTests;		// for literal attribute
		vector<double> m_vecMeanErrors;	// for non-literal attribute
		vector<double> m_vecMaxErrors;		// for non-literal attribute
		vector<double> m_vecLiteralErrors;	// for literal attribute
	};

private:
	vector<InputTest> m_vecTests;
	vector<AttributeMapping> m_vecAttributeMappings;
	Str m_sSourceDataFileName;

	void saveToXML(TiXmlElement &p_XML) const;
	void loadFromXML(const TiXmlElement &p_XML);

	void cleanObject();
	void normalizeTests();

	static bool loadElementsFromCSVFile(char p_cSeparator, Str p_sFileName, FILE *p_pLoadFile, vector< vector<Str> > &p_vecElements);
	static void retriveColumnNamesFromCSVFile(vector< vector<Str> > &p_vecElements, vector<Str> &p_vecColumnNames);
	static void removeIncorrectCSVElements(bool p_bContainsColumnNames, vector< vector<Str> > &p_vecElements);
	static bool checkKindsOfColumnsInCSVFile(vector< vector<Str> > &p_vecElements, vector<bool> &p_vecIsLiteral);
	static bool checkColumnIndexCorrectnessInCSVFile(const vector<int> &p_vecOutputColumns,const vector<int> &p_vecUnusedColumns,size_t p_uColumnsNumber);
	static bool getColumnRangesFromCSVFile(const vector< vector<Str> > &p_vecElements, const vector<bool> &p_vecIsLiteral, vector< pair<double,double> > &p_vecMinMaxData, vector< vector<Str> > &p_vecPossibleValuesData);
	static bool generateInputColumnsVectorForCSVFile(const vector<int> &p_vecOutputColumns, const vector<int> &p_vecUnusedColumns, size_t p_uColumnsNumber, vector<int> &p_vecInputColumns);
	static bool checkBasicValidityInCSVFile(const vector< vector<Str> > &p_vecElements);
	static void printDataAboutColumns(const vector<int> &p_vecColumnIndexes,Str p_sColumnType,const vector<bool> &p_vecIsLiteral,const vector< pair<double,double> > &p_vecMinMaxData
											 ,const vector< vector<Str> > &p_vecPossibleValuesData,const vector<Str> &p_vecColumnNames);

	bool generateAttributeMappingsAndTestsForCSVFile(const vector<int> &p_vecInputColumns,const vector<int> &p_vecOutputColumns
				,const vector< pair<double,double> > &p_vecMinMaxData,const vector< vector<Str> > &p_vecPossibleValuesData
				,const vector<Str> &p_vecColumnNames,const vector<bool> &p_vecIsLiteral,const vector< vector<Str> > &p_vecElements);
public:

	bool saveToFile(Str p_sFileName) const;
	bool loadFromFile(Str p_sFileName);
	bool loadFromCSVFile(Str p_sFileName,bool p_bContainsColumnNames,char p_cSeparator,const vector<int> &p_vecOutputColumns,const vector<int> &p_vecUnusedColumns);

	unsigned getTestCount() const;
	unsigned getInputCount() const;
	unsigned getOutputCount() const;

	const InputTest& getTest(int p_iIndex) const;
	InputTest& getTest(int p_iIndex);

	const vector<AttributeMapping>& getAttributeMappings() const;

	Str getSourceDataFileName() const;

	bool getDifferencesStatistics(DifferenceStatisticsType p_eDifferenceType, vector<AttributeLoggingData> &p_vecDifferencesData) const;
	void printVectorDifferenceInfo(InputTestSet::DifferenceStatisticsType p_eDifferenceType) const;

	InputTestSet();
	InputTestSet(const InputTestSet &p_TestSet);
	InputTestSet(unsigned p_uNumberTests,unsigned p_uNumberInputs,unsigned p_uNumberOutputs
		,const vector< pair<double,double> > &p_vecMinMax, void (*p_fTestingFunction)(const vector<double> &p_vecInputParameters
		,vector<double> &p_vecOutputParameters),MTRand *p_pRandomGenerator);


	~InputTestSet();
};
