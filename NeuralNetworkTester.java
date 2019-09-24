/*
* Name: Justin Senia
* E-Number: E00851822
* Date: 11/11/2017
* Class: COSC 461
* Project: #3
*/

import java.io.*;
import java.util.*;


//program tests neural network
public class NeuralNetworkTester
{

//Main method for testing
	public static void main(String[] args) throws IOException
	{
		//construct neural network
		NeuralNetwork network = new NeuralNetwork();
		
		//Used for external testing class to brute force parameter config
		//NeuralNetworkOptim nNetOpt = new NeuralNetworkOptim(454647);
		//nNetOpt.NeuralOptimizationIterator();
		//nNetOpt.NeuralOptParamToString();
		
		//creating buffered reader for getting user input
		java.io.BufferedReader keyIn = new BufferedReader(new InputStreamReader(System.in));

		//message welcoming to the program/giving instructions
		System.out.println("************************************************");
		System.out.println("*            Neural network program            *");
		System.out.println("************************************************");

		//start a loop that continues querying for input as long as user
		//does not enter "quit" (without the quotes)
		while (true)
		{

			System.out.println("************************************************");
			System.out.println("*        Please type your menu selection       *");
			System.out.println("************************************************");
			System.out.println("|        - Train    (to train the AI)          |");
			System.out.println("|        - Validate (to run validation)        |");
			System.out.println("|        - Run      (to process a file)        |");
			System.out.println("|        - Quit     (to quit)                  |");
			System.out.println("________________________________________________");
			
			String userInMenu = "";			//used for file entry or quitting

			System.out.print("Please enter your choice (just the word, no punctuation): ");
			userInMenu = keyIn.readLine(); //reading user input
			
			//using String trim to remove any newline characters and/or whitespaces in filename
			userInMenu = userInMenu.trim();

			if (userInMenu.equalsIgnoreCase("quit")) //if user typed quit, stops program
				break;
			else
			{
				try
				{

					//this case handles if user enters input correctly or not
					if (userInMenu.equalsIgnoreCase("train")	||
						userInMenu.equalsIgnoreCase("validate")	||
						userInMenu.equalsIgnoreCase("run"))
					{
						//used for storing input and output file names
						String userInFile = "";
						String userOutFile = "";
						
						if (userInMenu.equalsIgnoreCase("train"))
						{
							//prompting user for filename input
							System.out.print("Please enter in a filename that contains training data: ");
							userInFile = keyIn.readLine(); //reading user filename input
						}
						else if (userInMenu.equalsIgnoreCase("validate"))
						{
							//prompting user for filename input
							System.out.print("Please enter in a filename that contains Validation data: ");
							userInFile = keyIn.readLine(); //reading user filename for input
							
							//prompting user for filename for output
							System.out.print("Please enter in a filename that will store final processed output data: ");
							userOutFile = keyIn.readLine(); //reading user filename for output
						}
						else if (userInMenu.equalsIgnoreCase("run"))
						{
							//prompting user for filename input
							System.out.print("Please enter in a filename that contains data to be processed as input: ");
							userInFile = keyIn.readLine(); //reading user filename for input
							
							//prompting user for filename for output
							System.out.print("Please enter in a filename that will store final processed output data: ");
							userOutFile = keyIn.readLine(); //reading user filename for output
						}
						
						//using String trim to remove any newline characters and/or whitespaces in filenames
						userInFile = userInFile.trim();
						if (userInMenu.equalsIgnoreCase("run"))
						{userOutFile = userOutFile.trim();}
						
						//establishing working directory for file I/O
						String currentDir = System.getProperty("user.dir");
						File fIn = new File(currentDir + '\\' + userInFile);
						
						//using scanner with new input file based on user input
						Scanner scanIn = new Scanner(fIn);
						
						//creating printwriter for file output
						File fOut = new File("preprocess_" + userInFile);
						PrintWriter PWOut = new PrintWriter(fOut, "UTF-8");
						
						
						//preprocessing data before it will be used in the neural network
						if (userInMenu.equalsIgnoreCase("train"))
						{network.preProcessFile(scanIn, PWOut, 'T');}
						else if (userInMenu.equalsIgnoreCase("validate"))
						{network.preProcessFile(scanIn, PWOut, 'V');}
						else if (userInMenu.equalsIgnoreCase("run"))
						{network.preProcessFile(scanIn, PWOut, 'R');}
						
						PWOut.close();
						scanIn.close();
						
						//creating String variable to store preprocessed file name 
						String preProcFileName = ("preprocess_" + userInFile);
						
						//creating String variable to store processed file name
						String procFileName = ("process_" + userInFile);
						
					
						if (userInMenu.equalsIgnoreCase("train"))
						{
							//loading training data
							network.loadTrainingData(preProcFileName);
							
							//set parameters of network
							System.out.print("Please enter an int value to indicate number of hidden nodes: ");
							int numMid = Integer.parseInt(keyIn.readLine()); //reading user input
								
							System.out.print("Please enter an int value to indicate number of iterations: ");
							int numIterations = Integer.parseInt(keyIn.readLine()); //reading user input
								
							System.out.print("Please enter an int value to indicate the seed value: ");
							int seedVal = Integer.parseInt(keyIn.readLine()); //reading user input

							System.out.print("Please enter a double value between 0 and 1 for learning rate: ");
							double learnRate = Double.parseDouble(keyIn.readLine()); //reading user input
							
							//applying user supplied parameters to neural network
							network.setParameters(numMid, numIterations, seedVal, learnRate);
							
							//training the network with the inputs already loaded
							network.train();
						}
						else if (userInMenu.equalsIgnoreCase("validate"))
						{
							//validate based on supplied data
							network.validate(preProcFileName, procFileName);
						}
						else if (userInMenu.equalsIgnoreCase("run"))
						{
							//running data through neural network to process it and get outputs
							network.testData(preProcFileName, procFileName);
						}
						
						if (userInMenu.equalsIgnoreCase("validate") ||
							userInMenu.equalsIgnoreCase("run"))
						{	
							//establishing working directory for file I/O
							fIn = new File(currentDir + '\\' + procFileName);
							
							//using scanner with new input file based on user input
							scanIn = new Scanner(fIn);
							
							//creating printwriter for file output
							fOut = new File(userOutFile);
							PWOut = new PrintWriter(fOut, "UTF-8");
							
							//post process files to return to original value range
							if (userInMenu.equalsIgnoreCase("validate"))
							{network.postProcessFile(scanIn, PWOut, 'V');}
							else if (userInMenu.equalsIgnoreCase("run"))
							{network.postProcessFile(scanIn, PWOut, 'R');}
						
							//closing scanner and printwriters to protect data
							scanIn.close();
							PWOut.close();
						}
						
						//cleaning up leftover files
						File preProcFilDel = new File(currentDir + '\\' + preProcFileName);
						File procFilDel = new File(currentDir + '\\' + procFileName);
						
						//deleting leftover temp files
						preProcFilDel.delete();
						procFilDel.delete();
					}
					
					//this case is for when input is unrecognized
					else
					{
						System.out.println("Error reading input, returning to main selection screen");
					}
				}
				catch (IOException e) //catches if there were any fileIO exceptions
				{
					;
				}
			}
		}
	}
}