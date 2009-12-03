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

private:
	vector<InputTest> m_vecTests;
	vector<Str> m_vecInColumnNames;
	vector<Str> m_vecOutColumnNames;
	//int inColumnCount;
	//int outColumnCount;

	real_gpu *md_pTestSetMemory;

	void saveToXML(TiXmlElement &p_XML) const;
	void loadFromXML(const TiXmlElement &p_XML);

	void cleanObject();

public:

	bool saveToFile(Str p_sFileName) const;
	bool loadFromFile(Str p_sFileName);

	unsigned getTestCount() const;
	unsigned getInputCount() const;
	unsigned getOutputCount() const;

	const InputTest& getTest(int p_iIndex) const;
	InputTest& getTest(int p_iIndex);

	// If p_bCompareWithCorrectOutput == true, compare m_vecCorrectOutputs and m_vecNetworkOutputsGPU
	// If p_bCompareWithCorrectOutput == false, compare m_vecNetworkOutputs and m_vecNetworkOutputsGPU
	bool getDifferencesStatistics(vector<double> &p_vecMaxAbsoluteErrors,vector<double> &p_vecMaxProportionalErrors, DifferenceStatisticsType p_eDifferenceType) const;

	void randomizeTests(MTRand *p_pRandomGenerator);
	void setOutputFunction(const vector< pair<double,double> > &p_vecMinMax, void (*p_fTestingFunction)(const vector<double> &p_vecInputParameters,vector<double> &p_vecOutputParameters),MTRand *p_pRandomGenerator);

	InputTestSet();
	InputTestSet(const InputTestSet &p_TestSet);
	InputTestSet(unsigned p_uNumberTests,unsigned p_uNumberInputs,unsigned p_uNumberOutputs);

	~InputTestSet();
};
