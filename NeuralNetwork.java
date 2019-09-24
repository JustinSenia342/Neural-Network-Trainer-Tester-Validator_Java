/*
* Name: Justin Senia
* E-Number: E00851822
* Date: 11/11/2017
* Class: COSC 461
* Project: #3
*/

import java.io.*;
import java.util.*;
import java.text.*;

//neural network class
public class NeuralNetwork
{
	/***********************************************************************************/
	
	//Training record class
	private class Record
	{
		private double[] input;	//inputs of record
		private double[] output;	//outputs of record
		
		//constructor of record
		private Record(double[] input, double[] output)
		{
			this.input = input;	//assign inputs
			this.output = output;	//assign outputs
		}
	}


	/***********************************************************************************/

	//***make the following 3 public for optimization tester class use****//
	public int numberRecords;	//number of training records
	public int numberInputs;	//number of inputs
	public int numberOutputs;	//number of outputs
	
	private int numberMiddle;		//number of hidden nodes
	private int numberIterations;	//number of iterations
	private double rate;			//learning rate
	
	private ArrayList<Record> records;	//list of training records
	
	private double[] input;		//inputs
	private double[] middle;	//outputs at hidden nodes
	private double[] output;	//outputs at output nodes
	
	private double[] errorMiddle;	//errors at hidden nodes
	private double[] errorOut;		//errors at output nodes
	
	private double[] thetaMiddle;	//thetas at hidden nodes
	private double[] thetaOut;		//thetas at output nodes
	
	private double[][] matrixMiddle;	//weights between input/hidden nodes
	private double[][] matrixOut;		//weights between hidden/output nodes
	
	private double[] preProcColMin;		//keeps track of column minimum values from training
	private double[] preProcColMax;		//keeps track of column maximum values from training
	
	DecimalFormat nonSciNoteForm = new DecimalFormat("0.00000000000000000");

	/***********************************************************************************/

	//constructor of neural network
	public NeuralNetwork()
	{
		//parameters are zero
		numberRecords = 0;
		numberInputs = 0;
		numberOutputs = 0;
		numberMiddle = 0;
		numberIterations = 0;
		rate = 0;
		
		//arrays are empty
		records = null;
		input = null;
		middle = null;
		output = null;
		errorMiddle = null;
		errorOut = null;
		thetaMiddle = null;
		thetaOut = null;
		matrixMiddle = null;
		matrixOut = null;
	}

	/***********************************************************************************/

	//method loads training records from training file
	public void loadTrainingData(String trainingFile) throws IOException
	{
		String currentDir = System.getProperty("user.dir");
		File fIn = new File(currentDir + '\\' + trainingFile);
		
		Scanner inFile = new Scanner(fIn);
		
		//read number of records, inputs, outputs
		numberRecords = inFile.nextInt();
		numberInputs = inFile.nextInt();
		numberOutputs = inFile.nextInt();
		
		//empty list of new records
		records = new ArrayList<Record>();
		
		//for each training record
		for (int i = 0; i < numberRecords; i++)
		{
			//read inputs
			double[] input = new double[numberInputs];
			for (int j = 0; j < numberInputs; j++)
			{
				input[j] = inFile.nextDouble();
			}
			
			//read outputs
			double[] output = new double[numberOutputs];
			for (int j = 0; j < numberOutputs; j++)
			{
				output[j] = inFile.nextDouble();
			}
			
			//create training record record
			Record record = new Record(input, output);
			
			//add record to list
			records.add(record);
		}
		
		inFile.close();
	}

	/***********************************************************************************/

	//method sets parameters of neural network
	public void setParameters(int numberMiddle, int numberIterations, int seed, double rate)
	{
		//set hidden nodes, iterations, rate
		this.numberMiddle = numberMiddle;
		this.numberIterations = numberIterations;
		this.rate = rate;
		
		//initialize random number generation
		Random rand = new Random(seed);
		
		//create input/output arrays
		input = new double[numberInputs];
		middle = new double[numberMiddle];
		output = new double[numberOutputs];
		
		//System.out.println(input.length);
		//System.out.println(numberInputs);
		
		//create error arrays
		errorMiddle = new double[numberMiddle];
		errorOut = new double[numberOutputs];
		
		//intitialize thetas at hidden nodes
		thetaMiddle = new double[numberMiddle];
		for (int i = 0; i < numberMiddle; i++)
		{
			thetaMiddle[i] = 2*rand.nextDouble() - 1;
		}
		
		//initialize thetas at output nodes
		thetaOut = new double[numberOutputs];
		for (int i = 0; i < numberOutputs; i ++)
		{
			thetaOut[i] = 2*rand.nextDouble() - 1;
		}
		
		//intitialize weights between input/hidden nodes
		matrixMiddle = new double[numberInputs][numberMiddle];
		for (int i = 0; i < numberInputs; i++)
		{
			for (int j = 0; j < numberMiddle; j++)
			{
				matrixMiddle[i][j] = 2*rand.nextDouble() - 1;
			}
		}
		
		//initialize weights between hidden/output nodes
		matrixOut = new double[numberMiddle][numberOutputs];
		for (int i = 0; i < numberMiddle; i++)
		{
			for (int j = 0; j < numberOutputs; j++)
			{
				matrixOut[i][j] = 2*rand.nextDouble() - 1;
			}
		}
	}

	/***********************************************************************************/

	//method trains neural network
	public void train()
	{
		//repeat iteration number of times
		for (int i = 0; i < numberIterations; i++)
		{
			//for each training record
			for (int j = 0; j < numberRecords; j++)
			{
				//calculate input/output
				forwardCalculation(records.get(j).input);
				
				//compute errors, eupdate weights/thetas
				backwardCalculation(records.get(j).input);
			}
		}
	}

	/***********************************************************************************/

	//METHOD PERFORMS FORWARD PASS = COMPUTES INPUT/OUTPUT
	private void forwardCalculation(double[] trainingInput)
	{
		//System.out.println(numberInputs);
	//	System.out.println(input.length);
		//System.out.println(trainingInput.length);
		//feed inputs of record
		for (int i = 0; i < numberInputs; i++)
		{
			input[i] = trainingInput[i];
		}
		
		//for each hidden node
		for (int i = 0; i < numberMiddle; i++)
		{
			double sum = 0;
			
			//compute input at hidden node
			for (int j = 0; j < numberInputs; j++)
			{
				sum += input[j]*matrixMiddle[j][i];
			}
			
			//add theta
			sum += thetaMiddle[i];
			
			//compute output at hidden node
			middle[i] = 1/(1 + Math.exp(-sum));
		}
		
		//for each output node
		for (int i = 0; i < numberOutputs; i++)
		{
			double sum = 0;
			
			//compute input at output node
			for (int j = 0; j < numberMiddle; j++)
			{
				sum += middle[j]*matrixOut[j][i];
			}
			
			//add theta
			sum += thetaOut[i];
			
			//compute output at output node
			output[i] = 1/(1 + Math.exp(-sum));
		}
	}

	/***********************************************************************************/

	//Method performs backward pass - computes errors, updates weights/thetas
	private void backwardCalculation(double[] trainingOutput)
	{
		//compute error at each output node
		for (int i = 0; i < numberOutputs; i++)
		{
			errorOut[i] =  output[i]*(1-output[i])*(trainingOutput[i]-output[i]);
		}
		
		//compute error at each hidden node
		for (int i = 0; i < numberMiddle; i++)
		{
			double sum = 0;
			
			for (int j = 0; j < numberOutputs; j++)
			{
				sum += matrixOut[i][j]*errorOut[j];
			}
			
			errorMiddle[i] = middle[i]*(1-middle[i])*sum;
		}
		
		//update weights between hidden/output nodes
		for (int i = 0; i < numberMiddle; i++)
		{
			for (int j = 0; j < numberOutputs; j++)
			{
				matrixOut[i][j] += rate*middle[i]*errorOut[j];
			}
		}
		
		//update weights between  input/hidden nodes
		for (int i = 0; i < numberInputs; i++)
		{
			for (int j = 0; j < numberMiddle; j++)
			{
				matrixMiddle[i][j] += rate*input[i]*errorMiddle[j];
			}
		}
		
		//update thetas at output nodes
		for (int i = 0; i < numberOutputs; i++)
		{
			thetaOut[i] += rate*errorOut[i];
		}
		
		//update thetas at hidden nodes
		for (int i = 0; i < numberMiddle; i++)
		{
			thetaMiddle[i] += rate*errorMiddle[i];
		}
	}

	/***********************************************************************************/

	//method computes output of an input
	private double[] test(double[] input)
	{
		//forward pass input
		forwardCalculation(input);
		
		//return output produced
		return output;
	}

	/***********************************************************************************/

	// Method reads inputs from input file and writes outputs to output file
	public void testData(String inputFile, String outputFile) throws IOException
	{
		Scanner inFile = new Scanner(new File(inputFile));
		PrintWriter outFile = new PrintWriter(new FileWriter(outputFile));
		
		numberRecords = inFile.nextInt();
		
		//for each record
		for (int i = 0; i < numberRecords; i++)
		{
			double[] input = new double[numberInputs];
			
			//read input from input file
			for (int j = 0; j < numberInputs; j++)
			{
				input[j] = inFile.nextDouble();
			}
			
			//find output using neural network
			double[] output = test(input);
			
			//write output to output file
			for(int j = 0; j < numberOutputs; j++)
			{
				outFile.print(nonSciNoteForm.format(output[j]) + " ");
			}
			outFile.println();
		}
		
		inFile.close();
		outFile.close();
	}

	/***********************************************************************************/

	//method validates the network using the data from a file
	public void validate(String validationFile, String validationOut) throws IOException
	{
		Scanner inFile = new Scanner(new File(validationFile));
		
		File fOut = new File(validationOut);
		PrintWriter PWOut = new PrintWriter(fOut, "UTF-8");

		numberRecords = inFile.nextInt();
		
		//for each record
		for (int i = 0; i < numberRecords; i++)
		{
			//read inputs
			double[] input = new double[numberInputs];
			for (int j = 0; j < numberInputs; j++)
			{
				input[j] = inFile.nextDouble();
			}
			
			//read actual outputs
			double[] actualOutput = new double[numberOutputs];
			for (int j = 0; j < numberOutputs; j++)
			{
				actualOutput[j] = inFile.nextDouble();
			}
			
			//find predicted output
			double[] predictedOutput = test(input);
			
			//write actual and predicted outputs to file
			for (int j = 0; j < numberOutputs; j++)
			{
				PWOut.print(nonSciNoteForm.format(actualOutput[j]) + " " + nonSciNoteForm.format(predictedOutput[j]) + " ");
			}
			
			PWOut.println("");
		}
		
		PWOut.close();
		inFile.close();
	}

	/***********************************************************************************/

	//method finds root mean square error between actual and predicted output
	public double computeError(double[] actualOutput, double[] predictedOutput)
	{
		double error = 0;
		
		//sum of squares of errors
		for (int i = 0; i < actualOutput.length; i++)
		{
			error += Math.pow(actualOutput[i] - predictedOutput[i], 2);
		}
		
		//root mean square error
		return Math.sqrt(error/actualOutput.length);
	}

	/***********************************************************************************/
	
	//method preprocesses data (converts everything to 0 to 1 value)
	public void preProcessFile(Scanner sIn, PrintWriter pWrite, char typeFlag)
	{
		//gets number of records from file
		int numDataClusters = sIn.nextInt();
		
		//creating variables to store read in file data
		int numInputs = 0;	//stores number of inputs
		int numOutputs = 0;	//stores number of outputs
		int numInputAndOutput = 0;	//stores sum of number of inputs and number of outputs
		double[][] preProcArray;	//stores read in data
		
		if (typeFlag == 'T') //if file is a training file
		{
			//read in number of inputs and outputs
			numInputs = sIn.nextInt();
			numOutputs = sIn.nextInt();
			
			//add together to drive later for loops
			numInputAndOutput = numInputs + numOutputs;
			
			//declaring min and max arrays
			preProcColMin = new double[numInputAndOutput];
			preProcColMax = new double[numInputAndOutput];
			
			//delclaring array to hold data based on read input size values
			preProcArray = new double[numDataClusters][numInputAndOutput];
		}
		else //assuming file is either a validation file, or a run/testing file
		{
			//adding together input number and output number from previously read training file
			numInputAndOutput = numberInputs + numberOutputs;
			
			if (typeFlag == 'V')	//if file is validation file
			{
				//declare multidimensional array based on previously read training file sizes
				preProcArray = new double[numDataClusters][numInputAndOutput];
			}
			else					//if file is Test/run file
			{
				//declare multidimensional array with second array index based on just number of inputs
				preProcArray = new double[numDataClusters][numberInputs];
			}
		}
		
		
		//intializing arrays
		//initializes temp min max arrays if training
		for (int i = 0; i < numDataClusters; i++)
		{
			for (int j = 0; j < numInputAndOutput; j++)
			{
				//handles first line of array intitialization and min/max arrays simultaneously
				if (i == 0) //Initialized min and max arrays to default values
				{
					if ( typeFlag == 'T') //if training file
					{
						//initialize min and max arrays
						preProcColMin[j] = 9999999;
						preProcColMax[j] = 0;
					}
					
					if (typeFlag == 'R') //if run/test file
					{
						//just initialize preProcessArray based on number of inputs
						if (j < numberInputs)
						{
							preProcArray[i][j] = 0;
						}
					}
					else //if training file or validation file
					{
						//initialize preprocessarray based on sum #ofinputs & #ofoutputs
						preProcArray[i][j] = 0;
					}
				}
				else //continue to initialize the rest of the preProc array with default values
				{
					if (typeFlag == 'R')	//if it's a run file base on # of inputs
					{
						if (j < numberInputs)
						{
							preProcArray[i][j] = 0;
						}
					}
					else	//if it's a validation or training file 
					{
						//base on sum #ofinputs & #ofoutputs
						preProcArray[i][j] = 0;
					}
				}	
			}
		}
		
		//reads in data from external file, 
		//updates mins and maxes of columns if training
		for (int i = 0; i < numDataClusters; i++)
		{
			for (int j = 0; j < numInputAndOutput; j++)
			{
				
				if (typeFlag == 'R') //if file is run/test file
				{
					if (j < numberInputs)
					{
						//read in data from file
						preProcArray[i][j] = sIn.nextDouble();
					}
				}
				else if (typeFlag == 'V') //if file is validation file
				{
					//read in data from file
					preProcArray[i][j] = sIn.nextDouble();
				}
				else //if file is training file
				{
					//read in data from file and also...
					preProcArray[i][j] = sIn.nextDouble();
					
					//update min and max arrays via column value
					if (preProcArray[i][j] < preProcColMin[j])
					{
						preProcColMin[j] = preProcArray[i][j];
					}
					if (preProcArray[i][j] > preProcColMax[j])
					{
						preProcColMax[j] = preProcArray[i][j];
					}
				}
			}
		}
		
		//reformats all numbers to double values between 0 and 1
		for (int i = 0; i < numDataClusters; i++)
		{
			for (int j = 0; j < numInputAndOutput; j++)
			{
				//if file is run/test file
				if (typeFlag == 'R')
				{
					//normalize values
					if (j < numberInputs)
					{
						preProcArray[i][j] = (preProcArray[i][j]-preProcColMin[j])/(preProcColMax[j]-preProcColMin[j]);
					}
				}
				else //if file is validation or training
				{
					//normalize values
					preProcArray[i][j] = (preProcArray[i][j]-preProcColMin[j])/(preProcColMax[j]-preProcColMin[j]);
				}
				
				
			}
		}
		
		//creates new header in reformatted file
		if (typeFlag == 'T') //if file is training file
		{
			pWrite.println(numDataClusters + " " + numInputs + " " + numOutputs);
		}
		//if file is validation or run/test file
		else if (typeFlag == 'V' || typeFlag == 'R')
		{
			pWrite.println(numDataClusters);
		}
		
		//adds all data to reformatted file
		for (int i = 0; i < numDataClusters; i++)
		{
			for (int j = 0; j < numInputAndOutput; j++)
			{
				//if file is run/test file
				if (typeFlag == 'R')
				{
					if (j < numberInputs)
					{
						//write to file, preventing scientific notation
						pWrite.print(nonSciNoteForm.format(preProcArray[i][j]) + " ");
					}
				}
				else //if file is validation or training file
				{
					//write to file, preventing scientific notation
					pWrite.print(nonSciNoteForm.format(preProcArray[i][j]) + " ");
				}
			}
			pWrite.println(""); //used to format data with newline
		}
	}
	
	
	//method postprocesses data (converts everything back to normal from normalized value)
	public void postProcessFile(Scanner sIn, PrintWriter pWrite, char typeFlag)
	{
		double temp1 = 0;
		double temp2 = 0;
		
		//for each record
		for (int i = 0; i < numberRecords; i++)
		{
			System.out.println("");
			
			for (int j = 0; j < numberOutputs; j++)
			{
				if (typeFlag == 'V') //if validation file
				{
					temp1 = sIn.nextDouble(); //read in files, recalculate to normalize data for actual data
					temp1 = temp1*(preProcColMax[numberInputs + j] - preProcColMin[numberInputs + j]) + preProcColMin[numberInputs + j];
					temp2 = sIn.nextDouble(); //read in files, recalculate to normalize data for predicted data
					temp2 = temp2*(preProcColMax[numberInputs + j] - preProcColMin[numberInputs + j]) + preProcColMin[numberInputs + j];
					System.out.printf("(Actual Output = %5.2f Predicted Output = %5.2f)", temp1, temp2);
					pWrite.printf("(Actual Output = %5.2f Predicted Output = %5.2f)", temp1, temp2);
				}
				else if (typeFlag == 'R') //if run/test file
				{
					//read in files, recalculate to normalize data for actual data
					temp1 = sIn.nextDouble();
					temp1 = temp1*(preProcColMax[numberInputs + j] - preProcColMin[numberInputs + j]) + preProcColMin[numberInputs + j];
					System.out.printf("%5.2f ", temp1);
					pWrite.printf("%5.2f ", temp1);
				}
			}
			System.out.println("");
			pWrite.println("");
		}
			System.out.println("");
	}
	
	
	
}
