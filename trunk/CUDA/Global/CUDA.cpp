#include "stdafx.h"

//#include <omp.h>


void testingFunction(const vector<double> &p_vecInputParameters,vector<double> &p_vecOutputParameters)
{
	// 2 inputs, 2 outputs
	double dResult = cos(p_vecInputParameters[0] * p_vecInputParameters[1]) * cos (2 * p_vecInputParameters[0]);
	p_vecOutputParameters[0] = dResult;
	p_vecOutputParameters[1] = -dResult;
}

const int iInputs = 2;
const int iOutputs = 2;

vector< pair<double,double> > vecMinMax;

void checkIfGPUTrainingIsOK()
{
	// New MLP network
	MLP dummyNet;

	//const int iTrainedElements = 50000;
	const double dEta = 0.02;
	const int iTestsInTraining = 1000;
	const int iHiddenNeuronsInTesting = 45;
	const int iNumTrainedElements = 25000;
	const int iBatchSize = 8;

	dummyNet.setInputNeuronCount(iInputs);

	// New hidden layer - 20 neurons, 2 neurons in input layer, linear neurons
	dummyNet.addNewLayer(iHiddenNeuronsInTesting,Neuron::NT_SIGMOID);

	//dummyNet.addNewLayer(iHiddenNeuronsInTesting+4,Neuron::NT_SIGMOID);
	//dummyNet.addNewLayer(Layer(iHiddenNeuronsInTesting,iHiddenNeuronsInTesting,Neuron::NT_SIGMOID));

	// Output layer - 5 neurons, linear neurons
	dummyNet.addNewLayer(iOutputs,Neuron::NT_LINEAR);

	// we randomize weights in a all layers
	dummyNet.randomizeWeights(0.001,NULL);

	MLP dummyNetGPU (dummyNet);

	// 100 tests, 2 input variables, 1 output variables
	InputTestSet dummyTestSet(iTestsInTraining,iInputs,iOutputs,vecMinMax,testingFunction,NULL);
	//dummyTestSet.randomizeTests(NULL);
 
	logText(Logging::LT_INFORMATION,"Differences before training");
	dummyNet.executeNetwork(dummyTestSet);
	dummyTestSet.printVectorDifferenceInfo(InputTestSet::DST_CORRECT_AND_CPU);
	//dummyNetGPU.executeNetworkGPU(dummyTestSet);

	// Execute dummyNet on testSet (on both CPU and GPU). Output vectors in testSet are filled
	MTRand rand1(7),rand2(7);
	dummyNet.saveToFile("Before_train.xml");
	logText(Logging::LT_INFORMATION,"Started training CPU");
	dummyNet.trainNetwork(dummyTestSet,iNumTrainedElements,dEta,iBatchSize,&rand1);
	dummyNet.saveToFile("Train_CPU.xml");
	logText(Logging::LT_INFORMATION,"Started training GPU");
	dummyNetGPU.trainNetworkGPU(dummyTestSet,iNumTrainedElements,dEta,iBatchSize,&rand2);
	dummyNetGPU.saveToFile("Train_GPU.xml");
	logText(Logging::LT_INFORMATION,"Finished training GPU");

	dummyNet.executeNetwork(dummyTestSet);
	dummyNetGPU.executeNetworkGPU(dummyTestSet);
	dummyTestSet.printVectorDifferenceInfo(InputTestSet::DST_GPU_AND_CPU);
	dummyTestSet.printVectorDifferenceInfo(InputTestSet::DST_CORRECT_AND_CPU);
	dummyTestSet.printVectorDifferenceInfo(InputTestSet::DST_CORRECT_AND_GPU);
}

void makeTrainingWithManyPossibilities(const vector<InputTestSet> &p_vecTestSets, bool p_bTrainingCPU, bool p_bTrainingGPU)
{
	if((!p_bTrainingCPU && !p_bTrainingCPU) || p_vecTestSets.size() == 0)
	{
		logText(Logging::LT_ERROR,"At least one training kind and one test set has to be set");
		return;
	}

	const int numElementsInArrayTrainedElements = 2;
	const int numElementsInArrayEta = 1;
	const int numElementsInArrayTestsInTraining = 1;
	const int numElementsInArrayHiddenNeurons = 1;
	const int numElementsInArrayMaxAbsWeights = 1;
	const int iTrainedElementsArray[numElementsInArrayTrainedElements] = { 10000,50000 };
	const double dEtaArray[numElementsInArrayEta] = { 0.03 };
	const int iTestsInTrainingArray[numElementsInArrayTestsInTraining] = { 4/*1, 2, 4, 8*/ };
	const int iHiddenNeuronsArray[numElementsInArrayHiddenNeurons] = { 32/*,128,256,*/ };
	const double dMaxAbsWeightsArray[numElementsInArrayMaxAbsWeights] = { 0.02/*, 0.05*/ };

	size_t iTestsSetSize = p_vecTestSets.size();
	//const int iNumTests = 1000;

	//InputTestSet **testSetsInTraining = new InputTestSet*[iTestsSetSize];

	logText(Logging::LT_INFORMATION,"Started training");

	//for(int d=0;d<iTestsSetSize;++d)
	//{
	//	testSetsInTraining[d] = new InputTestSet(iNumTests,iInputs,iOutputs,vecMinMax,testingFunction,NULL);
	//}

	int iSeed = int(getRandom01(NULL) * 100000001);

	for(int iTrainedElementsIndex=0;iTrainedElementsIndex<numElementsInArrayTrainedElements;++iTrainedElementsIndex)
	{
		// Parallel cannot be used here - because kernels are called inside this loop ;;; #pragma omp parallel for
		for(int iEtaIndex=0;iEtaIndex<numElementsInArrayEta;++iEtaIndex)
		{
			// Each thread has its own random number generator
			
			//MTRand generatorInThreadCPUBase(iSeed);
			//MTRand generatorInThreadGPUBase(iSeed);
			for(int iTestsInTrainingIndex=0;iTestsInTrainingIndex<numElementsInArrayTestsInTraining;++iTestsInTrainingIndex)
			{
				for(int iHiddenNeuronsIndex=0;iHiddenNeuronsIndex<numElementsInArrayHiddenNeurons;++iHiddenNeuronsIndex)
				{
					for(int iMaxAbsWeightsIndex=0;iMaxAbsWeightsIndex<numElementsInArrayMaxAbsWeights;++iMaxAbsWeightsIndex)
					{ 
						vector<InputTestSet::AttributeLoggingData> vecDifferencesDataCPU;

						vector<InputTestSet::AttributeLoggingData> vecDifferencesDataGPU;

						vector<InputTestSet::AttributeLoggingData> vecDifferencesDataCPUGPU;

						MTRand generatorInThreadCPU(iSeed);
						MTRand generatorInThreadGPU(iSeed);

						Logging::Timer timer;
						unsigned int uiMilisecondsGPU = 0,uiMilisecondsCPU = 0;
						logTextParams(Logging::LT_INFORMATION,"Iterations:\t%d\tEta:\t%f\tTests in iteration:\t%d\tHidden neurons:\t%d\tMax Abs Weights:\t%f"
							,iTrainedElementsArray[iTrainedElementsIndex],dEtaArray[iEtaIndex],iTestsInTrainingArray[iTestsInTrainingIndex],iHiddenNeuronsArray[iHiddenNeuronsIndex],dMaxAbsWeightsArray[iMaxAbsWeightsIndex]);

						for(size_t iTestSetIndex=0;iTestSetIndex<iTestsSetSize;++iTestSetIndex)
						{
							InputTestSet trainTestSet(p_vecTestSets[iTestSetIndex]);

							if(p_bTrainingCPU)
							{
								MLP trainNet;
								trainNet.setInputNeuronCount(trainTestSet.getInputCount());
								trainNet.addNewLayer(iHiddenNeuronsArray[iHiddenNeuronsIndex],Neuron::NT_SIGMOID);
								trainNet.addNewLayer(iHiddenNeuronsArray[iHiddenNeuronsIndex],Neuron::NT_SIGMOID);
								trainNet.addNewLayer(trainTestSet.getOutputCount(),Neuron::NT_LINEAR);
								trainNet.randomizeWeights(dMaxAbsWeightsArray[iMaxAbsWeightsIndex],&generatorInThreadCPU);

								// We train the network using CPU
								timer.start();
								trainNet.trainNetwork(trainTestSet,iTrainedElementsArray[iTrainedElementsIndex],dEtaArray[iEtaIndex],iTestsInTrainingArray[iTestsInTrainingIndex],&generatorInThreadCPU);
								int res = timer.stop();
								uiMilisecondsCPU += res;
								logTextParams(Logging::LT_INFORMATION,"Training time CPU %d = %d",iTestSetIndex+1,res);

								// execute trained network and check difference between correct output
								trainNet.executeNetwork(trainTestSet);

								trainTestSet.getDifferencesStatistics(InputTestSet::DST_CORRECT_AND_CPU,vecDifferencesDataCPU);
							}

							if(p_bTrainingGPU)
							{
								MLP trainNet;
								trainNet.setInputNeuronCount(trainTestSet.getInputCount());
								trainNet.addNewLayer(iHiddenNeuronsArray[iHiddenNeuronsIndex],Neuron::NT_SIGMOID);
								trainNet.addNewLayer(iHiddenNeuronsArray[iHiddenNeuronsIndex],Neuron::NT_SIGMOID);
								trainNet.addNewLayer(trainTestSet.getOutputCount(),Neuron::NT_LINEAR);
								trainNet.randomizeWeights(dMaxAbsWeightsArray[iMaxAbsWeightsIndex],&generatorInThreadGPU);	

								// We train the network using GPU
								timer.start();
								trainNet.trainNetworkGPU(trainTestSet,iTrainedElementsArray[iTrainedElementsIndex],dEtaArray[iEtaIndex],iTestsInTrainingArray[iTestsInTrainingIndex],&generatorInThreadGPU);
								uiMilisecondsGPU += timer.stop();

								// execute trained network and check difference between correct output
								trainNet.executeNetworkGPU(trainTestSet);

								trainTestSet.getDifferencesStatistics(InputTestSet::DST_CORRECT_AND_GPU,vecDifferencesDataGPU);
							}

							// if both CPU and GPU were tested, then we also print differences between them
							if(p_bTrainingCPU && p_bTrainingGPU)
							{
								trainTestSet.getDifferencesStatistics(InputTestSet::DST_GPU_AND_CPU,vecDifferencesDataCPUGPU);
							}
						}

						if(p_bTrainingCPU)
							printVectorDifferenceInfoFromVectors(vecDifferencesDataCPU,InputTestSet::DST_CORRECT_AND_CPU,uiMilisecondsCPU / (int)iTestsSetSize);

						if(p_bTrainingGPU)
							printVectorDifferenceInfoFromVectors(vecDifferencesDataGPU,InputTestSet::DST_CORRECT_AND_GPU,uiMilisecondsGPU / (int) iTestsSetSize);

						if(p_bTrainingCPU && p_bTrainingGPU)
							printVectorDifferenceInfoFromVectors(vecDifferencesDataCPUGPU,InputTestSet::DST_GPU_AND_CPU);
					}
				}
			}
		}
	}
}

void makeTrainingToGenerateStatistics(int p_iTestSetType = -1)
{
	const int iTestsSetSize = 2;
	vector<InputTestSet> vecTestSets;
	vector<int> vecOutputColumns;
	vector<int> vecUnusedColumns;

	if(p_iTestSetType == -1)
		p_iTestSetType = 5;

	if(p_iTestSetType == 1)
	{
		const int iNumTests = 1000;
		for(int iTestIndex=0;iTestIndex<iTestsSetSize;++iTestIndex)
		{
			vecTestSets.push_back(InputTestSet(iNumTests,iInputs,iOutputs,vecMinMax,testingFunction,NULL));
		}
	}
	else
	{
		InputTestSet testSetCSV;
		switch(p_iTestSetType)
		{
			case 2:
				vecOutputColumns.push_back(12);
				testSetCSV.loadFromCSVFile("Resources\\Test_data\\forestfires.csv",true,',',vecOutputColumns,vecUnusedColumns);		
				break;
			case 3:
				vecOutputColumns.push_back(4);
				testSetCSV.loadFromCSVFile("Resources\\Test_data\\iris.data",false,',',vecOutputColumns,vecUnusedColumns);		
				break;
			case 4:
				vecUnusedColumns.push_back(0);
				vecOutputColumns.push_back(1);
				testSetCSV.loadFromCSVFile("Resources\\Test_data\\wdbc.data",false,',',vecOutputColumns,vecUnusedColumns);		
				break;
			case 5:
				vecOutputColumns.push_back(8);
				testSetCSV.loadFromCSVFile("Resources\\Test_data\\Concrete_Data.csv",true,';',vecOutputColumns,vecUnusedColumns);		
				break;
		}

		// We add iTestsSetSize same test sets
		for(int iTestIndex=0;iTestIndex<iTestsSetSize;++iTestIndex)
		{
			vecTestSets.push_back(testSetCSV);
		}

		logTextParams(Logging::LT_INFORMATION,"!!!!!!!!!!!!!!!!!!!!!!!!!! Testing file %s !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",testSetCSV.getSourceDataFileName().c_str());
	}

	makeTrainingWithManyPossibilities(vecTestSets,true,true);
}

void makeAllTrainingsToToGenerateStatistics()
{
	for(int iTestSetIndex = 2;iTestSetIndex<=5;++iTestSetIndex)
	{
		makeTrainingToGenerateStatistics(iTestSetIndex);
	}
}

void doExecuteNetworksCPUAndGPUAndSaveLoad()
{
	// New MLP network
	MLP dummyNet;

	const int iNumTests = 2550;
	const int iHiddenNeurons = 256;

	// New hidden layer - 20 neurons, 2 neurons in input layer, linear neurons
	dummyNet.setInputNeuronCount(iInputs);
	dummyNet.addNewLayer(iHiddenNeurons,Neuron::NT_SIGMOID);
	dummyNet.addNewLayer(iHiddenNeurons,Neuron::NT_SIGMOID);

	// Output layer - 5 neurons, linear neurons
	dummyNet.addNewLayer(iOutputs,Neuron::NT_LINEAR);

	// we randomize weights in all layers
	dummyNet.randomizeWeights(0.01,NULL);

	// 100 tests, 2 input variables, 1 output variables
	InputTestSet dummyTestSet(iNumTests,iInputs,iOutputs,vecMinMax,testingFunction,NULL);
	//dummyTestSet.randomizeTests(NULL);

	// Execute dummyNet on testSet (on both CPU and GPU). Output vectors in testSet are filled
	const int iTimesTried = 1;

	logText(Logging::LT_INFORMATION,"Started execution CPU");

	Logging::Timer timer;
	timer.start();
	for(int a=0;a<iTimesTried;++a)
	{
		dummyNet.executeNetwork(dummyTestSet);
	}
	int result = timer.stop();

	logTextParams(Logging::LT_INFORMATION,"Finished execution CPU. Time: %u ms. Started execution GPU",result);

	timer.start();
	unsigned int uiFullTime = 0,uiKernelTime = 0;
	for(int a=0;a<iTimesTried;++a)
	{
		dummyNet.executeNetworkGPU(dummyTestSet,&uiFullTime,&uiKernelTime);
	}
	result = timer.stop();

	logTextParams(Logging::LT_INFORMATION,"Finished execution GPU. Time: %u ms, %u , %u",result,uiFullTime,uiKernelTime);

	// We retrieve and print differences between CPU and GPU results for each output (these should be small).
	dummyTestSet.printVectorDifferenceInfo(InputTestSet::DST_GPU_AND_CPU);

	// check differences before training network
	dummyTestSet.printVectorDifferenceInfo(InputTestSet::DST_CORRECT_AND_CPU);
 
	dummyNet.saveToFile("NetworkStruct.xml");
	dummyTestSet.saveToFile("TestSet.xml");

	NeuralNetwork *pToLoad = NULL;
	NeuralNetwork::loadFromFile("NetworkStruct.xml",pToLoad);
	InputTestSet testSetToLoad;
	testSetToLoad.loadFromFile("TestSet.xml");

	pToLoad->saveToFile("NetworkStruct2.xml");
	testSetToLoad.saveToFile("TestSet2.xml");
}

void checkIfCSVReadingIsOK()
{
	InputTestSet testSetCSV;		// Nowy zestaw testów
	vector<int> vecOutputColumns;	// Tworzenie listy numerów kolumn wyjœciowych (wynikowych)
	vecOutputColumns.push_back(5);	// Jedyna wyjœciowa kolumna - indeks 5
	vector<int> vecUnusedColumns;	// Lista numerów kolumn nieu¿ywanych - pusta
	testSetCSV.loadFromCSVFile		// £adowanie listy testów
		("Resources\\Test_data\\forestfires2.csv"	// Plik wejœciowy z testami w formacjie CSV
		,true						// Pierwszy wiersz zawiera nazwy kolumn
		,','						// Okreœlenie znaku oddzielaj¹cego elementy - przecinek
		,vecOutputColumns			// Podanie listy kolumn wyjœciowych
		,vecUnusedColumns);			// Podanie listy kolumn nieu¿ywanych

	MLP dummyNet;								// Nowa sieæ MLP
	dummyNet.setInputNeuronCount				// Ustawienie iloœci neuronów wejœciowych
		(testSetCSV.getInputCount());
	dummyNet.addNewLayer						// Dodawanie warstwie ukrytej
		(3										// Iloœæ neuronów
		,Neuron::NT_SIGMOID);					// Funkcja aktywancji w warstwie ukrytej
	dummyNet.addNewLayer						// Ustawienie iloœci neuronów wyjœciowych ...
		(testSetCSV.getOutputCount()			// ... - tyle ile wyjœæ w zestawie testów
		,Neuron::NT_LINEAR);					// Wyjœcie sieci - linearne
	dummyNet.randomizeWeights(0.01,NULL);		// dobranie losowych wartoœci wag
	dummyNet.trainNetwork						// Uczenie sieci przez CPU
		(testSetCSV								// Uczenie wczeœniej za³adowanym zestawem testów
		,6000									// Iloœæ sekwencji uczenia sieci
		,0.01									// eta - czynnik uczenia
		,1										// Iloœæ testów uczona na raz
		,NULL);									// Generator liczb pseudolosowych - niepotrzebny

	dummyNet.executeNetwork(testSetCSV);		// Uruchomienie sieci na wszystkich testach przez CPU
	dummyNet.executeNetworkGPU(testSetCSV);		// Uruchomienie sieci na wszystkich testach przez GPU


	testSetCSV.saveToFile("TestSetFromCSV.xml");// Zapisywanie zestawu testów jako XML
	dummyNet.saveToFile("NetworkStruct.xml");	// Zapisywanie sieci MLP jako XML
	testSetCSV.loadFromFile("TestSetFromCSV.xml");
	testSetCSV.saveToFile("TestSetFromCSV2.xml");
}
 
int main()
{
	// We set, which logging types are allowed
	unsigned int uiAllowedLogging = Logging::LT_INFORMATION | Logging::LT_WARNING | Logging::LT_ERROR;
	Logging::setAllowedLoggingTypes(
		uiAllowedLogging /*| Logging::LT_MEMORY | Logging::LT_DEBUG*/		// console output
		, uiAllowedLogging);											// file output

	vecMinMax.push_back(pair<double,double> (0,M_PI)); // First input variable
	vecMinMax.push_back(pair<double,double> (0,M_PI)); // Second input variable

	logText(Logging::LT_INFORMATION,"Application Started");

	doExecuteNetworksCPUAndGPUAndSaveLoad();

	//makeTrainingToGenerateStatistics(); //makeAllTrainingsToToGenerateStatistics();

	//checkIfGPUTrainingIsOK();

	//checkIfCSVReadingIsOK();

	return 0;
}
