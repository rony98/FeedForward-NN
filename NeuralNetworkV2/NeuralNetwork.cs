using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using com.shephertz.app42.paas.sdk.csharp;
using com.shephertz.app42.paas.sdk.csharp.upload;
using System.Threading;
using System.Net;
using System.Diagnostics;

namespace NeuralNetworkV2
{
    class NeuralNetwork
    {
        ////////////////////////////////////////////||\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        /////////////////////////////////////// VARIABLES \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        ////////////////////////////////////////////||\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

        
        //Node amount values for the neural network
        private int numInput;
        private int numHidden;
        private int numOutput;

        //Arrays for inputs/weights/biases/outputs
        private double[] inputs;

        private double[][] inputHiddenWeights;
        private double[] hiddenBiases;
        private double[] hiddenOutputs;

        private double[][] hiddenOutputWeights;
        private double[] outputBiases;
        private double[] outputs;

        //Random object to generate values
        private Random random;

        //Accuracy variables
        public int numCorrect;
        public int numWrong;

        //App42 variables
        public OnlineStorageHelper onlineStorageHelper;
        private int maxEpochs;
        private double learnRate;
        private double momentum;
        private string filePath;
        private double errInterval;
        private string loadFilePath;
        private int fileNum;


        ///////////////////////////////////////////||\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        ///////////////////////////////////// CONSTRUCTOR \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        ///////////////////////////////////////////||\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

        
        //Consutrctor which creates a neural network with the given attributes
        public NeuralNetwork(int numInput, int numHidden, int numOutput, int seed, int currState, string filePath, int fileNum)
        {
            onlineStorageHelper = new OnlineStorageHelper();
            onlineStorageHelper.Initialize();

            //Sets the node amount values
            this.numInput = numInput;
            this.numHidden = numHidden;
            this.numOutput = numOutput;

            //Initializes input array
            inputs = new double[numInput];

            //Initializes the input layer weights
            inputHiddenWeights = MakeMatrix(numInput, numHidden, 0.0);
            hiddenBiases = new double[numHidden];
            hiddenOutputs = new double[numHidden];

            //Initializes the hidden layer weights
            hiddenOutputWeights = MakeMatrix(numHidden, numOutput, 0.0);
            outputBiases = new double[numOutput];
            outputs = new double[numOutput];

            //Sets the random object for generating
            random = new Random(seed);

            //Intialize weights
            this.InitializeWeights(currState, filePath, fileNum);
        }


        ///////////////////////////////////////////||\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        /////////////////////////////////////// METHODS \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        ///////////////////////////////////////////||\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


        //Helper for creating the matrix 
        private double[][] MakeMatrix(int cols, int rows, double value)
        {
            //Creates the result 2D array
            double[][] result = new double[rows][];

            //Initalizes the second dimension of the array
            for (int r = 0; r < result.Length; r++)
            {
                result[r] = new double[cols];
            }

            //For each value of the 2D array, sets it's value to the value given
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i][j] = value;
                }
            }

            //Returns the resulting array
            return result;
        }


        //Helper for initializing the weights
        private void InitializeWeights(int currState, string filePath, int fileNum)
        {
            if (currState == 1)
            {
                //Calculates the amount of weights/biases needed
                int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;

                //Array for the initial weights 
                double[] initialWeights = new double[numWeights];

                //Randomly sets the initial weights
                for (int i = 0; i < initialWeights.Length; i++)
                {
                    initialWeights[i] = (0.5 + 0.5) * random.NextDouble() - 0.5;
                }

                //Sets the weights
                SetWeights(initialWeights);
            }
            else
            {
                LoadData(filePath, fileNum);
            }
        }


        //Sets the weights of the neural network
        public void SetWeights(double[] weights)
        {
            //Calculates the amount of weights/biases that should be in the array
            int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;

            //Checks if the amount of weights/biases in the array is correct. If it's not it throws an exception
            if (weights.Length != numWeights)
            {
                throw new Exception("Bad weights array in SetWeights");
            }

            //Variable for which value of the weights array is currently being used
            int k = 0;

            //Loops through the weights that should exist for the input layer 
            for (int i = 0; i < numHidden; i++)
            {
                for (int j = 0; j < numInput; j++)
                {
                    inputHiddenWeights[i][j] = weights[k++];
                }
            }

            //Loops through the weights that are the biases for the hidden layer
            for (int i = 0; i < numHidden; i++)
            {
                hiddenBiases[i] = weights[k++];
            }

            //Loops through the weights that should exist for the hidden layer 
            for (int i = 0; i < numOutput; i++)
            {
                for (int j = 0; j < numHidden; j++)
                {
                    hiddenOutputWeights[i][j] = weights[k++];
                }
            }

            //Loops through the weights that are the biases for the output layer
            for (int i = 0; i < numOutput; i++)
            {
                outputBiases[i] = weights[k++];
            }
        }


        //Method to get the weights
        public double[] GetWeights()
        {
            //Calculates the amount of weights which exist
            int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;

            //Initializes the results array
            double[] result = new double[numWeights];

            //Variable for which value of the weights array is currently being used
            int k = 0;

            //Loops through the weights for the input layer
            for (int i = 0; i < inputHiddenWeights.Length; i++)
            {
                for (int j = 0; j < inputHiddenWeights[0].Length; j++)
                {
                    result[k++] = inputHiddenWeights[i][j];
                }
            }

            //Loops through the biases for the hidden layer 
            for (int i = 0; i < hiddenBiases.Length; i++)
            {
                result[k++] = hiddenBiases[i];
            }

            //Loops through the weights for the hidden layer 
            for (int i = 0; i < hiddenOutputWeights.Length; i++)
            {
                for (int j = 0; j < hiddenOutputWeights[0].Length; j++)
                {
                    result[k++] = hiddenOutputWeights[i][j];
                }
            }

            //Loops through the biases for the output layer
            for (int i = 0; i < outputBiases.Length; i++)
            {
                result[k++] = outputBiases[i];
            }

            //Returns the results array
            return result;
        }


        //Creates the weights/bias array for the input layer to hidden layer transition
        //Used when saving the weights/biases in a file to be loaded back to continue training in the future
        public double[][] HiddenWeights()
        {
            //Creates a new 2d array which will store all the input to hidden layer weights and their biases
            double[][] temp = new double[inputHiddenWeights.Length][];

            //Loops through all the weights
            for (int i = 0; i < inputHiddenWeights.Length; i++)
            {
                //Sets the second dimension to be the size of all the weights + one bias
                temp[i] = new double[inputHiddenWeights[i].Length + 1];

                //Loop to set all the weights at each index. Each index is set except for the bias
                for (int j = 0; j < inputHiddenWeights[i].Length - 1; j++)
                {
                    temp[i][j] = inputHiddenWeights[i][j];
                }

                //Bias is set for the current index i
                temp[i][temp[i].Length - 1] = hiddenBiases[i];
            }

            //The new 2d array is returned, containing all the input to hidden layer weights and their biases
            return temp;
        }


        //Creates the weights/bias array for the hidden layer to output layer transition
        //Used when saving the weights/biases in a file to be loaded back to continue training in the future
        public double[][] OutputWeights()
        {
            //Creates a new 2d array which will store all the hidden to output layer weights and their biases
            double[][] temp = new double[hiddenOutputWeights.Length][];

            //Loops through all the weights
            for (int i = 0; i < hiddenOutputWeights.Length; i++)
            {
                //Sets the second dimension to be the size of all the weights + one bias
                temp[i] = new double[hiddenOutputWeights[i].Length + 1];

                //Loop to set all the weights at each index. Each index is set except for the bias
                for (int j = 0; j < hiddenOutputWeights[i].Length - 1; j++)
                {
                    temp[i][j] = hiddenOutputWeights[i][j];
                }

                //Bias is set for the current index i
                temp[i][temp[i].Length - 1] = outputBiases[i];
            }

            //The new 2d array is returned, containing all the hidden to output layer weights and their biases
            return temp;
        }


        //Method to shuffle an integer array (training data)
        private void Shuffle(int[] array)
        {
            //Loops through the entire array, shuffling the values of the current index, and a randomly generated index
            for (int i = 0; i < array.Length; i++)
            {
                int r = random.Next(i, array.Length);
                int temp = array[r];
                array[r] = array[i];
                array[i] = temp;
            }
        }


        //Method for checking the error in the training data
        public double Error(double[][] trainData)
        {
            //Variable for the squared error, and arrays for the training data which is being compared
            double sumSquaredError = 0.0;
            double[] trainInputValues = new double[numInput];
            double[] trainOutputValues = new double[numOutput];

            //Go through each training set
            for (int i = 0; i < trainData.Length; i++)
            {
                //Sets the training data needing to be compared
                Array.Copy(trainData[i], trainInputValues, numInput);
                Array.Copy(trainData[i], numInput, trainOutputValues, 0, numOutput);

                //Calculates the output values with the current weights/biases and the input values from the training Data
                double[] newOutputValues= ComputeOutputs(trainInputValues);

                //Loops through all the outpues
                for (int j = 0; j < numOutput; j++)
                {
                    //Calculates the error by subtracting the train data output value by the current calculated output value and adds it to the squared error
                    double err = trainOutputValues[j] - newOutputValues[j];
                    sumSquaredError += err * err;
                }
            }

            //The error is calculated and returned
            return sumSquaredError / trainData.Length;
        }


        //Method to check the accuracy of the weights/biases compared to the test data
        public double Accuracy(double[][] testData)
        {
            //Variables which store the amount of result correct/wrong and the input/output values
            int numCorrect = 0;
            int numWrong = 0;
            double[] trainInputValues = new double[numInput];
            double[] trainOutputValues = new double[numOutput];
            double[] newOutputValues;

            //Loop for all the images within the test data
            for (int i = 0; i < testData.Length; i++)
            {
                //Sets the training data which is used to calculate outputs and compared for correct/wrong result
                Array.Copy(testData[i], trainInputValues, numInput);
                Array.Copy(testData[i], numInput, trainOutputValues, 0, numOutput);

                //New outputs calculated using the inputs from the training data and the current weights/biases
                newOutputValues = ComputeOutputs(trainInputValues);

                //Gets the best result for both the training output values and the new output values which were just calculated
                int maxIndex = MaxIndex(newOutputValues);
                int trainingMaxIndex = MaxIndex(trainOutputValues);

                //Sets whether the result is correct or not
                if (maxIndex == trainingMaxIndex)
                    numCorrect++;
                else
                    numWrong++;
            }

            //Sets the global variables for the amount correct/wrong
            this.numCorrect = numCorrect;
            this.numWrong = numWrong;

            //Returns the accuracy of the current weights/biases
            return (numCorrect * 1.0) / (numCorrect + numWrong);
        }


        //Method for getting the best result out of all the output values (the one with the highest value) 
        private int MaxIndex(double[] outputs)
        {
            //Sets the current biggest index to be the first variable, check if the rest are larger then it
            int biggestIndex = 0;
            double biggestValue = outputs[0];

            //Loops through all the output values, checking if the next value is bigger then the current biggest value and updating accordingly
            for (int i = 0; i < outputs.Length; i++)
            {
                if (outputs[i] > biggestValue)
                {
                    biggestValue = outputs[i];
                    biggestIndex = i;
                }
            }

            //The index of the largest output value is returned, representing what the network best predicts an image says
            return biggestIndex;
        }

        //Method for calculating the result of the hyper tan activation function
        private double HyperTan(double x)
        {
            if (x < -20.0) return -1.0;
            else if (x > 20.0) return 1.0;
            else return Math.Tanh(x);
        }

        //Method for calculating the result of the sigmoid activation function
        private double SigmoidActivation(double x)
        {
            return 1.0 / 1.0 + Math.Exp(-1.0 * x);
        }


        //Scales the output results to add up to 1.0, ensuring a proper highest value can be found
        private double[] Softmax(double[] outputResults)
        {
            //Variable for the total sum of all the output results (sum of all weights/biases * hidden layer value)
            double sum = 0.0;

            //Loop through the output results, and update the total sum with the value of e^x where x is the current output result
            for (int i = 0; i < outputResults.Length; i++)
            {
                sum += Math.Exp(outputResults[i]);
            }

            //Creates result array which will now contain the scaled version of the outputResults
            double[] result = new double[outputResults.Length];

            //Loop through the old output results, and calculates the scaled version of each result by dividing the result of e^x, where x is the old result, by the total sum 
            for (int i = 0; i < outputResults.Length; i++)
            {
                result[i] = Math.Exp(outputResults[i]) / sum;
            }

            //Returns the result which is now scaled so that the sum adds up to 1.0
            return result;
        }


        //Method for calculating the outputs given the input values and all the weights/biases
        public double[] ComputeOutputs(double[] inputValues)
        {
            //Arrays for the sums (weights * inputs) for the hidden and output layer
            double[] hiddenSums = new double[numHidden];
            double[] outputSums = new double[numOutput];

            //Copies given input values (inputValues) into the inputs array
            for (int i = 0; i < inputValues.Length; i++)
            {
                inputs[i] = inputValues[i];
            }

            //Calculates new sums/values for the hidden layer 
            for (int j = 0; j < numHidden; j++)
            {
                for (int i = 0; i < numInput; i++)
                {
                    hiddenSums[j] += inputs[i] * inputHiddenWeights[j][i];
                }
            }

            //Adds the bias to each new sum/value of the hidden layer
            for (int i = 0; i < numHidden; i++)
            {
                hiddenSums[i] += hiddenBiases[i];
            }

            //Uses the activation function on each of the hidden layer sums
            for (int i = 0; i < numHidden; i++)
            {
                //hiddenOutputs[i] = SigmoidActivation(hiddenSums[i]);
                hiddenOutputs[i] = HyperTan(hiddenSums[i]);

                if (Double.IsNaN(hiddenOutputs[i]))
                {
                    AddError("NaN number found");
                    Console.WriteLine();
                }
            }

            //Calculates new sums/values for the output layer 
            for (int j = 0; j < numOutput; j++)
            {
                for (int i = 0; i < numHidden; i++)
                {
                    outputSums[j] += hiddenOutputs[i] * hiddenOutputWeights[j][i];
                }
            }

            //Adds the bias to each new sum/value of the output layer
            for (int i = 0; i < numOutput; i++)
            {
                outputSums[i] += outputBiases[i];
            }

            //Computes the softmax function which scales all the sums to add up to 1.0 (easier checking for best network prediction)
            double[] softOut = Softmax(outputSums);
            Array.Copy(softOut, outputs, softOut.Length);

            //The scaled output values are returned
            return outputs;
        }


        //Method to save data to an external server to be viewed from a different mobile device (through the Neural Network Extension Mobile App)
        public void SaveData(string filePath, int epoch)
        {
            //Creates the file all the hidden weights/biases will be written too
            StreamWriter writer = new StreamWriter(filePath + "/hidden-weights-" + epoch + ".txt");

            //Sets the 2D array of the hidden layer weights/biases to their values and sets a temp variable
            double[][] hiddenWeights = HiddenWeights();
            string tempString = "";

            //Loop through all the hidding layer weights/biases, adding each one to a string which is written to the file
            for (int i = 0; i < hiddenWeights.Length; i++)
            {
                tempString = "";

                for (int j = 0; j < hiddenWeights[i].Length - 1; j++)
                {
                    tempString += hiddenWeights[i][j] + " ";
                }

                tempString += hiddenWeights[i][hiddenWeights[i].Length - 1];

                writer.WriteLine(tempString);
            }

            //Hidden weights file is finished writing and is closed
            writer.Close();

            //Creates the file all the output weights/biases will be written too
            writer = new StreamWriter(filePath + "/output-weights-" + epoch + ".txt");

            //Sets the 2D array of the output layer weights/biases to their values
            double[][] outputWeights = OutputWeights();

            //Loop through all the output layer weights/biases, adding each one to a string which is written to the file
            for (int i = 0; i < outputWeights.Length; i++)
            {
                tempString = "";

                for (int j = 0; j < outputWeights[i].Length - 1; j++)
                {
                    tempString += outputWeights[i][j] + " ";
                }

                tempString += outputWeights[i][outputWeights[i].Length - 1];

                writer.WriteLine(tempString);
            }

            //Output weights file is finished writing and is closed
            writer.Close();
        }


        //Method to load data from an external server, this is used to continue from weights that have already begun training
        public void LoadData(string filePath, int fileNum)
        {
            //Calculates the total amount of weights
            int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;

            //Opens a file to read all the hidden weights/biases from
            StreamReader reader = new StreamReader(filePath + "/hidden-weights-" + fileNum + ".txt");

            //Creates the arrays and variables required for hidden and output layers weights/biases to be read
            string[] tempArray;
            double[] weights = new double[numWeights];
            double[] tempBiases = new double[numHidden];
            int currWeight = 0;
            int currBias = 0;

            //Until every line is read, each line has it's weights stored in the weights array and bias stored in the tempBiases array for the hidden layer
            while (!reader.EndOfStream)
            {
                tempArray = reader.ReadLine().Split(' ');

                for (int i = 0; i < numInput; i++)
                {
                    weights[currWeight] = Convert.ToDouble(tempArray[i]);

                    currWeight++;
                }

                tempBiases[currBias] = Convert.ToDouble(tempArray[tempArray.Length - 1]);
                currBias++;
            }

            //Adds the biases from the tempbiases array to the end of the weights array to simplify the weight setting process needing to be done in the end
            for (int i = 0; i < tempBiases.Length; i++)
            {
                weights[currWeight] = tempBiases[i];
                currWeight++;
            }

            //Hidden weights file is finished being read and is closed
            reader.Close();

            //Opens a file to read all the output weights/ biases from
            reader = new StreamReader(filePath + "/output-weights-" + fileNum + ".txt");

            //Resets the biases array/variable
            tempBiases = new double[numOutput];
            currBias = 0;

            //Until every line is read, each line has it's weights stored in the weights array and bias stored in the tempBiases array for the output layer
            while (!reader.EndOfStream)
            {
                tempArray = reader.ReadLine().Split(' ');

                for (int i = 0; i < numHidden; i++)
                {
                    weights[currWeight] = Convert.ToDouble(tempArray[i]);

                    currWeight++;
                }

                tempBiases[currBias] = Convert.ToDouble(tempArray[tempArray.Length - 1]);
                currBias++;
            }

            //Adds the biases from the tempbiases array to the end of the weights array to simplify the weight setting process needing to be done in the end
            for (int i = 0; i < tempBiases.Length; i++)
            {
                weights[currWeight] = tempBiases[i];
                currWeight++;
            }

            //Output weights file is finished being read and is closed
            reader.Close();

            //Using the single weights array containing all the weights and biases for both the hidden and output layer, the global weights are set for the training to begin
            SetWeights(weights);
        }


        //Method for training the weights and biases of the neural network to predict the correct digit for each image. 
        //This method uses the backpropagation algorithm to do the training.
        public double[] Train(double[][] trainData, int maxEpochs, double learnRate, double momentum, string filePath, double errInterval, out int currEpoch
            , string loadFilePath, int fileNum)
        {
            //Sets global variables to be the same value as the parameters past in
            this.learnRate = learnRate;
            this.momentum = momentum;
            this.filePath = filePath;
            this.errInterval = errInterval;
            this.maxEpochs = maxEpochs;
            this.loadFilePath = loadFilePath;
            this.fileNum = fileNum;

            //Arrays for the hidden to output layer gradients and biases (used to calculate new weights/biases)
            double[][] hiddenOutputGrads = MakeMatrix(numHidden, numOutput, 0.0);
            double[] outputBiasGrads = new double[numOutput];

            //Arrays for the input to hidden layer gradients and biases (used to calculate new weights/biases)
            double[][] inputHiddenGrads = MakeMatrix(numInput, numHidden, 0.0);
            double[] hiddenBiasGrads = new double[numHidden];

            //In both occassions, the gradients are used in calculating the amount the weights/biases should change (delta) in order to get 
            //more accurate predictions from the neural network

            //Arrays for the output and hidden layer signals, used in calculating the gradients
            double[] outputSignals = new double[numOutput];
            double[] hiddenSignals = new double[numHidden];

            //Delta arrays which are used to calculate the new and "improved" weights/baises (can lead to a local maximum which wouldn't be an improvement)
            double[][] inputHiddenPrevWeightsDelta = MakeMatrix(numInput, numHidden, 0.0);
            double[] hiddenPrevBiasesDelta = new double[numHidden];
            double[][] hiddenOutputPrevWeightsDelta = MakeMatrix(numHidden, numOutput, 0.0);
            double[] outputPrevBiasesDelta = new double[numOutput];

            //The training data input and output values
            double[] inputValues = new double[numInput];
            double[] targetValues = new double[numOutput];

            //Variables used in training the weights/biases
            int epoch = 0;
            double derivative = 0.0;
            double errorSignal = 0.0;

            //New sequence is created to be used in shuffling the order of training
            int[] sequence = new int[trainData.Length];
            for (int i = 0; i < sequence.Length; i++)
            {
                sequence[i] = i;
            }

            //Thread created for checking the settings of the Neural Network from an online source
            Thread settingsThread = new Thread(new ThreadStart(CheckSettingChange));
            settingsThread.IsBackground = true;
            settingsThread.Start();

            //Trains the weights/biases until the amount of times the data is trained is the set max (maxEpochs)
            while (epoch < maxEpochs)
            {
                //Check for the settings thread, if not alive then settings were changed these settings have to be applied
                if (!settingsThread.IsAlive)
                {
                    currEpoch = epoch;
                    return GetWeights();
                }

                //Adds one to the amount of cycles ran
                epoch++;

                //Saves all the data (weights and biases) in a file and upload it to online server if the statement below is met
                if (epoch % 1 == 0 && epoch < maxEpochs)
                {
                    if (epoch % errInterval == 0)
                    {
                        double trainErr = Error(trainData);
                        AddCurrInfo(filePath, "epoch = " + epoch + "  error = " + trainErr.ToString());
                        Console.WriteLine("epoch = " + epoch + "  error = " + trainErr.ToString());
                        //Console.ReadLine();
                    }

                    AddCurrInfo(filePath, "Saving Data, Curr: " + epoch);
                    Console.WriteLine("Saving Data, Curr: " + epoch);

                    SaveData(filePath, epoch);

                    Console.WriteLine("Saving Finished");
                }

                //Shuffles the sequence so the training data can be visited in random order
                Shuffle(sequence);

                //Loop for all of the training data
                for (int b = 0; b < trainData.Length; b++)
                {
                    //If the statement below is met, the title is updated with the current cycle (epoch) and trial (current set of training data)
                    if (b % 1000 == 0)
                    {
                        Console.Title = "Current: " + epoch + "  Trial: " + b;
                    }

                    //Using the random order from the sequence, a random set is grabbed and stored in inputValues and targetValues arrays
                    int currSeq = sequence[b];
                    Array.Copy(trainData[currSeq], inputValues, numInput);
                    Array.Copy(trainData[currSeq], numInput, targetValues, 0, numOutput);

                    //Using the input values from the training data, the outputs are calculated with the current weights and biases
                    ComputeOutputs(inputValues);

                    //Calculates the output node signals
                    for (int k = 0; k < numOutput; k++)
                    {
                        errorSignal = targetValues[k] - outputs[k];
                        derivative = (1.0 - outputs[k]) * outputs[k];
                        outputSignals[k] = errorSignal * derivative;
                    }

                    //Calculates the hidden to output gradients using the previously calculated output node signals
                    for (int j = 0; j < numHidden; j++)
                    {
                        for (int k = 0; k < numOutput; k++)
                        {
                            hiddenOutputGrads[k][j] = outputSignals[k] * hiddenOutputs[j];
                        }
                    }

                    //Calculates the output bias gradients using the previously calculated output node signals
                    for (int k = 0; k < numOutput; k++)
                    {
                        outputBiasGrads[k] = outputSignals[k] * outputSignals[k];
                    }

                    //Calculates the hidden node signals
                    for (int j = 0; j < numHidden; j++)
                    {
                        derivative = (1.0 + hiddenOutputs[j]) * (1.0 - hiddenOutputs[j]);
                        double sum = 0.0;
                        for (int k = 0; k < numOutput; k++)
                        {
                            sum += outputSignals[k] * hiddenOutputWeights[k][j];
                        }
                        hiddenSignals[j] = derivative * sum;
                    }

                    //Calculates the input to hidden gradients using the previously calculated hidden node signals
                    for (int i = 0; i < numInput; i++)
                    {
                        for (int j = 0; j < numHidden; j++)
                        {
                            inputHiddenGrads[j][i] = hiddenSignals[j] * inputs[i];
                        }
                    }

                    //Calculates the hidden bias gradients using the previously calculated hidden node signals
                    for (int j = 0; j < numHidden; j++)
                    {
                        hiddenBiasGrads[j] = hiddenSignals[j] * hiddenSignals[j];
                    }

                    //Updates the input to hidden layer weights using the calculated gradients
                    for (int i = 0; i < numInput; i++)
                    {
                        for (int j = 0; j < numHidden; j++)
                        {
                            double delta = inputHiddenGrads[j][i] * learnRate;
                            inputHiddenWeights[j][i] += delta;
                            inputHiddenWeights[j][i] += inputHiddenPrevWeightsDelta[j][i] * momentum;
                            inputHiddenPrevWeightsDelta[j][i] = delta;
                        }
                    }

                    //Updates the hidden layer biases using the calculated gradients
                    for (int j = 0; j < numHidden; j++)
                    {
                        double delta = hiddenBiasGrads[j] * learnRate;
                        hiddenBiases[j] += delta;
                        hiddenBiases[j] += hiddenPrevBiasesDelta[j] * momentum;
                        hiddenPrevBiasesDelta[j] = delta;
                    }

                    //Updates the hidden to output layer weights using the calculated gradients
                    for (int j = 0; j < numHidden; j++)
                    {
                        for (int k = 0; k < numOutput; k++)
                        {
                            double delta = hiddenOutputGrads[k][j] * learnRate;
                            hiddenOutputWeights[k][j] += delta;
                            hiddenOutputWeights[k][j] += hiddenOutputPrevWeightsDelta[k][j] * momentum;
                            hiddenOutputPrevWeightsDelta[k][j] = delta;
                        }
                    }

                    //Updates the output layer biases using the calculated gradients
                    for (int k = 0; k < numOutput; k++)
                    {
                        double delta = outputBiasGrads[k] * learnRate;
                        outputBiases[k] += delta;
                        outputBiases[k] += outputPrevBiasesDelta[k] * momentum;
                        outputPrevBiasesDelta[k] = delta;
                    }

                }
            }

            //Sets the amount of cycles (epoch) and returns an array of weights
            currEpoch = epoch;
            return GetWeights();
        }


        //Method for checking the difference between the settings file online and the current neural network settings and updating accordingly 
        private void CheckSettingChange()
        {
            //Creates and starts a stop watch to check settings being changed periodically 
            Stopwatch stopWatch = new Stopwatch();
            stopWatch.Start();

            //Continous loop to constantly check for changes in the settings file online
            while (true)
            {
                //If a check is required
                if (stopWatch.ElapsedMilliseconds >= 60000)
                {
                    //Stopwatch is reset and restarted
                    stopWatch.Reset();
                    stopWatch.Restart();

                    //If the online settings file exists
                    if (onlineStorageHelper.CheckFileExists("Settings.txt"))
                    {
                        //If the online settings file can be downloaded
                        if (onlineStorageHelper.DownloadFile("Settings.txt", "Settings.txt"))
                        {
                            //Opens a reader to read the downloaded settings file
                            StreamReader reader = new StreamReader("Settings.txt");

                            //Reads all the lines and stores them in the temp array
                            string[] temp = new string[7];
                            for (int i = 0; i < temp.Length; i++)
                            {
                                temp[i] = reader.ReadLine();
                            }

                            //Closes the reader since the file is finished being read
                            reader.Close();

                            //If a change in the settings was made, abort the current thread to start again with the new settings
                            if (this.learnRate != Convert.ToDouble(temp[0]) || this.momentum != Convert.ToDouble(temp[1]) || this.errInterval != Convert.ToDouble(temp[2]) ||
                                this.maxEpochs != Convert.ToInt32(temp[3]) || this.filePath != temp[4] || this.loadFilePath != temp[5] || this.fileNum != Convert.ToInt32(temp[6]))
                            {
                                Thread.CurrentThread.Abort();
                            }
                        }
                        //If the settings file can't be downloaded, show an error
                        else
                        {
                            AddError("Unable to download file within training method");
                            Console.WriteLine("Error Logged: " + "Unable to download file within training method");
                        }
                    }
                    //If the settings file can't be found, show an error
                    else
                    {
                        AddError("Unable to find Settings.txt file within App42 for training method");
                        Console.WriteLine("Error Logged: " + "Unable to find Settings.txt file within App42 for training method");
                    }
                }
            }
        }


        //Method for adding a new header for the errors that will come for the current file
        public void AddNewError(string filePath)
        {
            //Checks for an exisiting error file
            if (!onlineStorageHelper.DownloadFile("Errors.txt", "Errors.txt"))
            {
                Console.WriteLine("Unable to download Errors.txt file");
            }

            //Updates the error file with the new header and uploads the new file online
            StreamWriter writer = new StreamWriter("Errors.txt", true);
            writer.WriteLine("\n********************************************************************************");
            writer.WriteLine("**************************************" + filePath + "**************************************");
            writer.WriteLine("********************************************************************************\n");
            writer.Close();
            onlineStorageHelper.UpdateFile("Errors.txt", "Errors.txt", "TXT", "Errors");
        }


        //Method for adding a new error to the online errors file
        public void AddError(string newError)
        {
            //If the error file exists
            if (onlineStorageHelper.CheckFileExists("Errors.txt"))
            {
                //If the error file can be downloaded
                if (onlineStorageHelper.DownloadFile("Errors.txt", "Errors.txt"))
                {
                    //Opens the error file, writes a new error line within it, and closes the file
                    StreamWriter writer = new StreamWriter("Errors.txt", true);
                    writer.WriteLine(DateTime.Now.ToString() + "\n " + newError + "\n");
                    writer.Close();

                    //Uploads the new file online, if failed tell the user
                    if (!onlineStorageHelper.UpdateFile("Errors.txt", "Errors.txt", "TXT", "Errors"))
                    {
                        Console.WriteLine("Unable to update errors file with new information");
                    }
                }
                //If the error file can't be downloaded, tell the user
                else
                {
                    Console.WriteLine("Errors file not found on App42");
                }
            }
            //If the error file can't be found, tell the user
            else
            {
                Console.WriteLine("Errors file not found on App42");
            }
        }


        //Method for adding a new header for the current info file that is used to store the current information for the network
        public void AddNewCurrInfo(string filePath, string message)
        {
            //Opens the file, writes the message given, and closes the file
            StreamWriter writer = new StreamWriter(filePath + ".txt");
            writer.WriteLine(message);
            writer.Close();

            //Uploads the new file online
            onlineStorageHelper.UpdateFile(filePath + ".txt", filePath + ".txt", "TXT", "Current Info");
        }


        //Method for adding new information online for the current neural network
        public void AddCurrInfo(string filePath, string message)
        {
            //If the file can be found
            if (onlineStorageHelper.CheckFileExists(filePath + ".txt"))
            {
                //If the file can be downloaded
                if (onlineStorageHelper.DownloadFile(filePath + ".txt", filePath + ".txt"))
                {
                    //Opens the file for writing, writes the new information to the file, and closes the file
                    StreamWriter writer = new StreamWriter(filePath + ".txt", true);
                    writer.WriteLine(message);
                    writer.Close();

                    //Uploads the file online, if failed shows the error
                    if (!onlineStorageHelper.UpdateFile(filePath + ".txt", filePath + ".txt", "TXT", "Current Info"))
                    {
                        AddError("Unable to update current info file with new information");
                        Console.WriteLine("Error Logged: Unable to update current info file with new information");
                    }
                }
                //If the fail can't be downloaded, show the error
                else
                {
                    AddError("Unable to find current info file");
                    Console.WriteLine("Error Logged: Unable to find current info file");
                }
            }
            //If the file can't be found, show the error
            else
            {
                AddError("Unable to find current info file");
                Console.WriteLine("Error Logged: Unable to find current info file");
            }
        }
    }
}
