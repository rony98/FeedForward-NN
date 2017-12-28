using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using com.shephertz.app42.paas.sdk.csharp;
using com.shephertz.app42.paas.sdk.csharp.upload;
using System.Net;

namespace NeuralNetworkV2
{
    class Program
    {
        //Node amount values for the neural network
        private const int NUM_INPUT = 4;
        private const int NUM_HIDDEN = 5;
        private const int NUM_OUTPUT = 3;

        //If randomly generating values
        private const int NUM_ROWS = 1000;
        private const int SEED = 2;

        //Current State
        private static int currState = 0;

        //Saving info
        private static string filePath = "";
        private static string loadFilePath = "";
        private static int fileNum = 0;

        static void Main(string[] args)
        {
            while (true)
            {
                while (true)
                {
                    Console.WriteLine("What do you want to do?\n1. Train using new weights\n2. Train using existing weights\n3. Check accuracy for weights\n4. Check Error for weights");
                    string temp = Convert.ToString(Console.ReadLine());

                    if (Int32.TryParse(temp, out currState) && Convert.ToInt32(temp) > 0 && Convert.ToInt32(temp) < 5)
                    {
                        break;
                    }

                    Console.Clear();
                }

                if (currState == 1 || currState == 2)
                {
                    if (currState == 2)
                    {
                        Console.WriteLine("What folder would you like to load the weights from?");
                        filePath = Convert.ToString(Console.ReadLine());

                        while (true)
                        {
                            Console.WriteLine("What file number would you like to load the weights from?");
                            fileNum = Convert.ToInt16(Console.ReadLine());

                            if (File.Exists(filePath + "/hidden-weights-" + fileNum + ".txt") && File.Exists(filePath + "/output-weights-" + fileNum + ".txt"))
                            {
                                break;
                            }

                            Console.Clear();
                        }
                    }
                    else
                    {
                        Console.WriteLine("What folder would you like to save the weights to?");
                        filePath = Convert.ToString(Console.ReadLine());

                        if (System.IO.Directory.Exists(filePath))
                        {
                            System.IO.Directory.Delete(filePath, true);
                        }

                        System.IO.Directory.CreateDirectory(filePath);
                    }

                    NeuralNetwork network;
                    Console.WriteLine("Creating a " + 784 + "-" + 300 + "-" + 10 + " neural network");
                    network = new NeuralNetwork(784, 300, 10, SEED, currState, filePath, fileNum);
                    string temp = "";

                    if (currState == 2)
                    {
                        Console.WriteLine("Where would you like to save the weights to? Type same to keep unchanged");
                        temp = Console.ReadLine();

                        if (temp.ToLower() != "same")
                        {
                            filePath = Convert.ToString(temp);
                        }

                        if (System.IO.Directory.Exists(filePath))
                        {
                            System.IO.Directory.Delete(filePath, true);
                        }

                        System.IO.Directory.CreateDirectory(filePath);
                    }

                    double[][] data = LoadTrainingData();

                    Console.WriteLine("\nCreating train (80%) and test (20%) matrices");
                    double[][] trainData;
                    double[][] testData;
                    SplitTrainTest(data, 0.80, SEED, out trainData, out testData);
                    Console.WriteLine("Done\n");

                    //Setting the maximum number of training iterations
                    int maxEpochs = 1000;

                    //Setting the learning rate, controls how fast training works
                    double learnRate = 0.005;
                    //double learnRate = 0.05;

                    //Setting the momentum rate, the momentum rate is an optional parameter which is used to increase the speed of training
                    double momentum = 0.5;

                    double errInterval = 1;

                    Console.WriteLine("\nSetting maxEpochs (Max training iterations) = " + maxEpochs);
                    Console.WriteLine("Setting learnRate = " + learnRate.ToString());
                    Console.WriteLine("Setting momentum  = " + momentum.ToString());

                    Console.WriteLine("Setting up server...");

                    int currEpoch = 0;

                    while (currEpoch < maxEpochs)
                    {
                        network.AddNewError(filePath);

                        if (network.onlineStorageHelper.CheckFileExists("Settings.txt"))
                        {
                            if (network.onlineStorageHelper.DownloadFile("Settings.txt", "Settings.txt"))
                            {
                                StreamReader reader = new StreamReader("Settings.txt");

                                learnRate = Convert.ToDouble(reader.ReadLine());
                                momentum = Convert.ToDouble(reader.ReadLine());
                                errInterval = Convert.ToDouble(reader.ReadLine());
                                maxEpochs = Convert.ToInt32(reader.ReadLine());
                                filePath = reader.ReadLine();
                                loadFilePath = reader.ReadLine();
                                fileNum = Convert.ToInt32(reader.ReadLine());

                                reader.Close();
                            }
                            else
                            {
                                network.AddError("Unable to download Settings.txt file");
                                Console.WriteLine("Error Logged: " + "Unable to download Settings.txt file");
                            }
                        }
                        else
                        {
                            network.AddError("Unable to find Settings.txt file within App42");
                            Console.WriteLine("Error Logged: " + "Unable to find Settings.txt file within App42");

                            loadFilePath = "";
                            fileNum = 0;

                            StreamWriter writer = new StreamWriter("Settings.txt");

                            writer.WriteLine(learnRate);
                            writer.WriteLine(momentum);
                            writer.WriteLine(errInterval);
                            writer.WriteLine(maxEpochs);
                            writer.WriteLine(filePath);
                            writer.WriteLine(loadFilePath);
                            writer.WriteLine(fileNum);

                            writer.WriteLine("********************************************************************************");
                            writer.WriteLine("Learn Rate/Momentum/Error Interval/Max Epoch/Save Folder/Load Folder/File Number");
                            writer.WriteLine("********************************************************************************");

                            writer.Close();

                            if (!network.onlineStorageHelper.UpdateFile("Settings.txt", "Settings.txt", "TXT", "Settings"))
                            {
                                network.AddError("Unable to update settings file");
                                Console.WriteLine("Error Logged: " + "Unable to update settings file");
                            }
                        }

                        if (System.IO.Directory.Exists(filePath))
                        {
                            System.IO.Directory.Delete(filePath, true);
                        }

                        System.IO.Directory.CreateDirectory(filePath);

                        if (loadFilePath != "" && fileNum != 0)
                        {
                            network = new NeuralNetwork(784, 300, 10, SEED, 2, loadFilePath, fileNum);
                        }

                        if (currEpoch == 0)
                        {
                            network.AddNewCurrInfo(filePath, "Starting training");
                            Console.WriteLine("\nStarting training");
                        }
                        else
                        {
                            network.AddNewCurrInfo(filePath, "Restarting training");
                            Console.WriteLine("\nRestarting training");
                        }

                        double[] weights = network.Train(trainData, maxEpochs, learnRate, momentum, filePath, errInterval, out currEpoch, loadFilePath, fileNum);
                    }

                    Console.WriteLine("Training Done");

                    double trainAcc = network.Accuracy(trainData);
                    Console.WriteLine("\nFinal accuracy on training data = " +
                      trainAcc.ToString("F4"));

                    double testAcc = network.Accuracy(testData);
                    Console.WriteLine("Final accuracy on test data     = " +
                      testAcc.ToString("F4"));

                    Console.WriteLine("\nEnd back-propagation demo\n");
                    Console.WriteLine("Saving Data");

                    network.SaveData(filePath, maxEpochs);

                    Console.WriteLine("Saving finished, press any key to close");

                    Console.ReadLine();
                }
                else if (currState == 3 || currState == 4)
                {
                    while (true)
                    {
                        Console.WriteLine("What folder would you like to load the weights from?");
                        filePath = Convert.ToString(Console.ReadLine());

                        if (Directory.Exists(filePath))
                        {
                            break;
                        }

                        Console.Clear();
                    }

                    while (true)
                    {
                        Console.WriteLine("What file number would you like to load the weights from?");
                        fileNum = Convert.ToInt16(Console.ReadLine());

                        if (File.Exists(filePath + "/hidden-weights-" + fileNum + ".txt") && File.Exists(filePath + "/output-weights-" + fileNum + ".txt"))
                        {
                            break;
                        }

                        Console.Clear();
                    }

                    NeuralNetwork neural = new NeuralNetwork(784, 300, 10, SEED, currState, filePath, fileNum);

                    Console.WriteLine("Loading Data...");

                    neural.LoadData(filePath, fileNum);

                    Console.WriteLine("Loading Training and Testing Data...");

                    double[][] data = LoadTrainingData();
                    double[][] trainData;
                    double[][] testData;
                    SplitTrainTest(data, 0.80, SEED, out trainData, out testData);

                    if (currState == 3)
                    {
                        Console.WriteLine("Calculating Accuracy...");

                        double accuracy = neural.Accuracy(testData);

                        Console.WriteLine("Correct: " + neural.numCorrect + "\nWrong: " + neural.numWrong + "\nAccuracy: " + accuracy);
                    }
                    else
                    {
                        Console.WriteLine("Calculating Error...");

                        double error = neural.Error(trainData);

                        Console.WriteLine("Error: " + error);
                    }
                }

                Console.WriteLine("Type in YES to continue, any other key to close");
                string response = Console.ReadLine();

                if (response.ToLower() != "yes" && response.ToLower() != "ye" && response.ToLower() != "y")
                {
                    break;
                }
            }
        }


        //Splits the data into a training set and test set
        static void SplitTrainTest(double[][] allData, double trainPct, int seed, out double[][] trainData, out double[][] testData)
        {
            //Random object for generating values
            Random random = new Random(seed);

            //Sets the total rows
            int totalRows = allData.Length;

            //Calculates the amount of training set rows and test set rows there are
            int numTrainRows = (int)(totalRows * trainPct); // usually 0.80
            int numTestRows = totalRows - numTrainRows;

            //Initializes the train and test data arrays
            trainData = new double[numTrainRows][];
            testData = new double[numTestRows][];

            //Copies the allData array to copy
            double[][] copy = new double[allData.Length][]; // ref copy of data
            for (int i = 0; i < copy.Length; i++)
            {
                copy[i] = allData[i];
            }

            //Loops through the copy array in order to scramble the order of it
            for (int i = 0; i < copy.Length; i++)
            {
                //Scrambles the order of the copy array
                int r = random.Next(i, copy.Length);
                double[] tmp = copy[r];
                copy[r] = copy[i];
                copy[i] = tmp;
            }

            //Sets all the training set data
            for (int i = 0; i < numTrainRows; i++)
            {
                trainData[i] = copy[i];
            }

            //Sets all the test set data
            for (int i = 0; i < numTestRows; i++)
            {
                testData[i] = copy[i + numTrainRows];
            }
        }

        static double[][] LoadTrainingData()
        {
            double[][] data = new double[42000][];
            StreamReader reader = new StreamReader("train.csv");

            reader.ReadLine();

            string[] tempArray;
            int count = 0;
            string temp;

            while (!reader.EndOfStream)
            {
                temp = reader.ReadLine();
                tempArray = temp.Split(',');

                data[count] = new double[tempArray.Length + 9];

                for (int i = 1; i < tempArray.Length; i++)
                {
                    data[count][i - 1] = Convert.ToDouble(tempArray[i]);
                    data[count][i - 1] /= 255.0;
                    data[count][i - 1] = 2.0 * (data[count][i - 1]) - 1.0;
                }

                for (int i = tempArray.Length - 1; i < data[count].Length; i++)
                {
                    if (Convert.ToDouble(tempArray[0]) == i - (tempArray.Length - 1))
                    {
                        data[count][i] = 1;
                    }
                    else
                    {
                        data[count][i] = 0;
                    }
                }

                count++;
            }

            reader.Close();

            return data;
        }
    }
}