// CUDA.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <omp.h>

// JRTODO - napisz, ze kod bedzie dzialac pod 32bit i 64bit
// JRTODO - I ze wazne bylo, zeby dzialal tez na innych platformach
// JRTODO - Na CPU uzywam double do liczb zmiennoprzecinkowych, a na GPU - real_gpu (w zaleznosci od rodzaju GPU).
// JRTODO - jak metoda nie zmienia wnetrza obiektu, to oznacz ja jako const
// JRTODO - skalowanie danych wejsciowych
// JRTODO - maybe a possibility to change eta during training?
// JRTODO - obsluga wiecej niz 512 neuronow w warstkie, wiecej niz 65535 testow

void testingFunction(const vector<double> &p_vecInputParameters,vector<double> &p_vecOutputParameters)
{
	// 2 inputs, 1 output
	double dResult = cos(p_vecInputParameters[0] * p_vecInputParameters[1]) * cos (2 * p_vecInputParameters[0]);
	p_vecOutputParameters[0] = dResult;
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
										  ,vector< vector<double> > *p_vecResultsMaxAbsoluteErrors = NULL,vector< vector<double> > *p_vecResultsMeanAbsoluteErrors = NULL)
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

void printVectorDifferenceInfo(const InputTestSet &p_testSet,InputTestSet::DifferenceStatisticsType p_eDifferenceType)
{
	vector<double> vecMaxAbsoluteErrors;
	vector<double> vecMeanAbsoluteErrors;
	p_testSet.getDifferencesStatistics(vecMaxAbsoluteErrors,vecMeanAbsoluteErrors,p_eDifferenceType);
	printVectorDifferenceInfoFromVectors(vecMaxAbsoluteErrors,vecMeanAbsoluteErrors,p_eDifferenceType);
}

const int iInputs = 2;
const int iOutputs = 1;

vector< pair<double,double> > vecMinMax;

void checkIfGPUTrainingIsOK()
{
	// New MLP network
	MLP dummyNet;

	//const int iTrainedElements = 50000;
	const double dEta = 0.5;
	const int iTestsInTraining = 1;
	const int iHiddenNeuronsInTesting = 2;
	const int iNumTrainedElements = 1;

	// New hidden layer - 20 neurons, 2 neurons in input layer, linear neurons
	dummyNet.addNewLayer(Layer(iHiddenNeuronsInTesting,iInputs,Neuron::NT_SIGMOID));

	dummyNet.addNewLayer(Layer(iHiddenNeuronsInTesting,iHiddenNeuronsInTesting,Neuron::NT_SIGMOID));

	// Output layer - 5 neurons, linear neurons
	dummyNet.addNewLayer(Layer(iOutputs,iHiddenNeuronsInTesting,Neuron::NT_LINEAR));

	// we randomize weights in a all layers
	dummyNet.randomizeWeights(0.1,NULL);

	MLP dummyNetGPU (dummyNet);

	// 100 tests, 2 input variables, 1 output variables
	InputTestSet dummyTestSet(iTestsInTraining,iInputs,iOutputs);
	dummyTestSet.randomizeTests(NULL);

	dummyTestSet.setOutputFunction(vecMinMax,testingFunction,NULL);

	//dummyNet.executeNetwork(dummyTestSet);
	//dummyNetGPU.executeNetworkGPU(dummyTestSet);

	// Execute dummyNet on testSet (on both CPU and GPU). Output vectors in testSet are filled
	MTRand rand1(7),rand2(7);
	logText(Logging::LT_INFORMATION,"Started training CPU");
	dummyNet.trainNetwork(dummyTestSet,iNumTrainedElements,dEta,1,&rand1);
	logText(Logging::LT_INFORMATION,"Started training GPU");
	dummyNetGPU.trainNetworkGPU(dummyTestSet,iNumTrainedElements,dEta,1,&rand2);
	logText(Logging::LT_INFORMATION,"Finished training GPU");

	dummyNet.executeNetwork(dummyTestSet);
	dummyNetGPU.executeNetworkGPU(dummyTestSet);
	printVectorDifferenceInfo(dummyTestSet,InputTestSet::DST_GPU_AND_CPU);
}

void makeTraining()
{
	/*const int numElementsInArrays1 = 3;
	const int numElementsInArrays2 = 4;
	const int numElementsInArrays3 = 3;
	const int iTrainedElementsArray[numElementsInArrays1] = { 4000,8000,16000 };
	const double dEtaArray[numElementsInArrays2] = { 0.01, 0.02, 0.04, 0.08 };
	const int iTestsInTrainingArray[numElementsInArrays3] = { 1, 2, 4 };*/
	const int numElementsInArrays1 = 1;
	const int numElementsInArrays2 = 1;
	const int numElementsInArrays3 = 1;
	const int iTrainedElementsArray[numElementsInArrays1] = { 160000 };
	const double dEtaArray[numElementsInArrays2] = { 0.02 };
	const int iTestsInTrainingArray[numElementsInArrays3] = { 1 };

	const int numTriedTrainings = 3;
	const int iNumTests = 1000;
	const int iHiddenNeurons = 45;

	InputTestSet::DifferenceStatisticsType eDifferenceType = InputTestSet::DST_CORRECT_AND_CPU;

	InputTestSet **testSetsInTraining = new InputTestSet*[numTriedTrainings];

	logText(Logging::LT_INFORMATION,"Started training");

	/*testSetsInTraining[0] = new InputTestSet(iNumTests,iInputs,iOutputs);
	testSetsInTraining[0]->setOutputFunction(vecMinMax,testingFunction,NULL);
	for(int d=1;d<numTriedTrainings;++d)
		testSetsInTraining[d] = new InputTestSet(*testSetsInTraining[0]);*/

	for(int d=0;d<numTriedTrainings;++d)
	{
		testSetsInTraining[d] = new InputTestSet(iNumTests,iInputs,iOutputs);
		testSetsInTraining[d]->setOutputFunction(vecMinMax,testingFunction,NULL);
	}

	for(int a=0;a<numElementsInArrays1;++a)
	{
		//#pragma omp parallel for
		for(int b=0;b<numElementsInArrays2;++b)
		{
			// Each thread has its own random number generator
			MTRand generatorInThread(int(getRandom01(NULL) * 100000000));
			for(int c=0;c<numElementsInArrays3;++c)
			{
				vector<double> vecMaxAbsoluteErrors,vecMaxAbsoluteErrorsSum;
				vector<double> vecMeanAbsoluteErrors,vecMeanAbsoluteErrorsSum;
				vector< vector<double> > vecResultsMaxAbsoluteErrors,vecResultsMeanAbsoluteErrors;

				for(int d=0;d<numTriedTrainings;++d)
				{
					MLP trainNet;
					trainNet.addNewLayer(Layer(iHiddenNeurons,iInputs,Neuron::NT_SIGMOID));
					trainNet.addNewLayer(Layer(iOutputs,iHiddenNeurons,Neuron::NT_LINEAR));
					trainNet.randomizeWeights(0.01,&generatorInThread);

					InputTestSet trainTestSet(*testSetsInTraining[d]);

					// We train the network using CPU
					trainNet.trainNetwork(trainTestSet,iTrainedElementsArray[a],dEtaArray[b],iTestsInTrainingArray[c],&generatorInThread); // JRTODO - maybe a possibility to change eta during training?

					// execute trained network and check difference between correct output
					trainNet.executeNetwork(trainTestSet);

					trainTestSet.getDifferencesStatistics(vecMaxAbsoluteErrors,vecMeanAbsoluteErrors,eDifferenceType);

					for(unsigned e=0;e<vecMaxAbsoluteErrors.size();++e)
					{
						if(vecMaxAbsoluteErrorsSum.size())
						{
							vecMaxAbsoluteErrorsSum[e] += vecMaxAbsoluteErrors[e]/numTriedTrainings;
							vecMeanAbsoluteErrorsSum[e] += vecMeanAbsoluteErrors[e]/numTriedTrainings;
						}
						else
						{
							vecMaxAbsoluteErrorsSum.push_back(vecMaxAbsoluteErrors[e]/numTriedTrainings);
							vecMeanAbsoluteErrorsSum.push_back(vecMeanAbsoluteErrors[e]/numTriedTrainings);
						}
						vecResultsMaxAbsoluteErrors.push_back(vecMaxAbsoluteErrors);
						vecResultsMeanAbsoluteErrors.push_back(vecMeanAbsoluteErrors);
					}
				}		

				logTextParams(Logging::LT_INFORMATION,"Trained elements:\t%d\tEta:\t%f\tTests in array:\t%d",iTrainedElementsArray[a],dEtaArray[b],iTestsInTrainingArray[c]);
				printVectorDifferenceInfoFromVectors(vecMaxAbsoluteErrorsSum,vecMeanAbsoluteErrorsSum,eDifferenceType,&vecResultsMaxAbsoluteErrors,&vecResultsMeanAbsoluteErrors);
			}
		}
	}
}

void doExecuteNetworksAndSaveLoad()
{
	// New MLP network
	MLP dummyNet;

	const int iNumTests = 50000;
	const int iHiddenNeurons = 45;

	// New hidden layer - 20 neurons, 2 neurons in input layer, linear neurons
	dummyNet.addNewLayer(Layer(iHiddenNeurons,iInputs,Neuron::NT_SIGMOID));

	// Output layer - 5 neurons, linear neurons
	dummyNet.addNewLayer(Layer(iOutputs,iHiddenNeurons,Neuron::NT_LINEAR));

	// we randomize weights in a all layers
	dummyNet.randomizeWeights(0.01,NULL);

	// 100 tests, 2 input variables, 1 output variables
	InputTestSet dummyTestSet(iNumTests,iInputs,iOutputs);
	dummyTestSet.randomizeTests(NULL);

	dummyTestSet.setOutputFunction(vecMinMax,testingFunction,NULL);

	// Execute dummyNet on testSet (on both CPU and GPU). Output vectors in testSet are filled
	const int iTimesTried = 1;

	logText(Logging::LT_INFORMATION,"Started execution CPU");
	for(int a=0;a<iTimesTried;++a)
	{
		dummyNet.executeNetwork(dummyTestSet);
	}

	logText(Logging::LT_INFORMATION,"Started execution GPU");

	for(int a=0;a<iTimesTried;++a)
	{
		dummyNet.executeNetworkGPU(dummyTestSet);
	}

	logText(Logging::LT_INFORMATION,"Finished execution GPU");

	// We retrieve and print differences between CPU and GPU results for each output (these should be small).
	printVectorDifferenceInfo(dummyTestSet,InputTestSet::DST_GPU_AND_CPU);

	// check differences before training network
	printVectorDifferenceInfo(dummyTestSet,InputTestSet::DST_CORRECT_AND_CPU);

	/*dummyNet.saveToFile("NetworkStruct.xml");
	dummyTestSet.saveToFile("TestSet.xml");

	NeuralNetwork *pToLoad = NULL;
	NeuralNetwork::loadFromFile("NetworkStruct.xml",pToLoad);
	InputTestSet testSetToLoad;
	testSetToLoad.loadFromFile("TestSet.xml");

	pToLoad->saveToFile("NetworkStruct2.xml");
	testSetToLoad.saveToFile("TestSet2.xml");*/
}
 
int main()
{
	vecMinMax.push_back(pair<double,double> (0,M_PI)); // First input variable
	vecMinMax.push_back(pair<double,double> (0,M_PI)); // Second input variable

	//doExecuteNetworksAndSaveLoad();

	//makeTraining();

	checkIfGPUTrainingIsOK();

	return 0;
}
