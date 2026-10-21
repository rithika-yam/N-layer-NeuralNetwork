import java.io.*;
import static java.lang.System.currentTimeMillis;

/**
 * Network trains and outputs the desired values for four test cases in an N Layer network
 * configuration. It is a feed forward network that works toward minimizing error for the outputs
 * by training using backpropagation. Network includes input, hidden, and output activation layers.
 * The network can change logical operations, change the number of activation nodes, change the number of hidden
 * layers, change to manual weights, and switch between regular running and training.
 *
 * Alterable Content:
 * 1. To change all configuration parameters go to control file.
 * 2. Load inputs, truths, and weights from files. Save weights to weight file.
 * 3. To change accuracy (number of decimals for output). go to reportResults().
 *
 * Table of Contents:
 * public void echoConfiguration()
 * public void allocateArrays()
 * public double randomize(double low, double high)
 * public void populateWeightArrays()
 * public void runOneCase(int currentCase)
 * public void run()
 * public double thresholdFunction(double value)
 * public double thresholdDerivative(double value)
 * public double sigmoidFunction(double value)
 * private double sigmoidDerivative(double value)
 * public void setUpInputsForCase(int currentCase)
 * public double computeErrorForCase(int currentCase)
 * public void runForTrain(int currentCase)
 * public void train()
 * public boolean saveWeights()
 * public boolean loadWeights(String fileName)
 * public boolean loadConfiguration(String fileName)
 * public boolean loadInputs(String fileName)
 * public boolean loadTruths(String fileName)
 * public void reportResults()
 * public static void main(String[] args)
 *
 * @author Rithika Yamasani
 * @version 04-21-2024
 */
public class NNImage
{
   double[][] activations;          // activations for input, hidden, and output layers
   double[][] outputs;              // saved outputs only to print results for each case
   int numInputs;                   // number of input layer activations
   int numHiddenLayers;             // number of hidden layers
   int numLayers;                   // number of total activation layers
   int[] numActivationNodes;        // number of activation nodes in each layer
   int numOutputs;                  // number of output nodes in output layer
   int outputLayerIndex;            // index of the output layer
   int numCases;                    // number of cases for inputs
   double[][] cases;                // input activations with the four cases
   double[][] truths;               // the desired outputs for each case and each output node
   double dRandMin;                 // lower weight bound
   double dRandMax;                 // upper weight bound
   double lambda;                   // learning factor controls how far down the hill per step
   double[][][] weights;            // weights array for all layers
   double[][] theta;                // theta before activation function
   double[][] psi;                  // psi array
   int maxEpochs;                   // number of max runs through training algorithm
   double errorThresh;              // minimum error reached for the output before ending training
   boolean isTraining;              // whether training or running
   boolean randomWeight;            // option of manual or random weights when running
   double error;                    // current error
   int epochs;                      // current number of iterations
   long timeElapsedTrain;           // time in milliseconds it took to train
   String weightsFileName;          // file name for the weights file to be loaded
   String truthsFileName;           // file name for the truths file to be loaded
   String inputsFileName;           // file name for the inputs file to be loaded
   boolean saveWeights;             // choose to save weights after training
   int saveFrequency;               // after how many training loops to save weights
   int keepAliveFrequency;          // how frequently to display a progress message during training

   private static final int INPUT_LAYER = 0;          // layer index of input layer
   private static final int FIRST_HIDDEN_LAYER = 1;   // layer index of first hidden layer
   private static final String DEFAULT_CONTROL_FILENAME = "control_2-B-C-3.txt";  // Default control file

   /**
    * Prints out all the necessary information to understand the processes in the network.
    * Prints type of network, weight value bounds, maximum iterations for training, error threshold for
    * training, and training state.
    */
   public void echoConfiguration()
   {
      int n;

      // print network configuration
      System.out.print(numInputs);
      for (n = FIRST_HIDDEN_LAYER; n < numHiddenLayers + 1; n++) // first hidden layer begins at n = 1
      {
         System.out.print("-" + numActivationNodes[n]);
      }
      System.out.println("-" + numOutputs + " Network");

      if (isTraining)
      {
         if (randomWeight)
         {
            System.out.println("Random weights between [" + dRandMin + ", " + dRandMax + "]");
         }
         else
         {
            System.out.println("Weights file = " + weightsFileName);
         }

         System.out.println("Lambda = " + lambda);
         System.out.println("Maximum iterations = " + maxEpochs);
         System.out.println("Error Threshold = " + errorThresh);
         System.out.println("Training...");
      } // if (isTraining)
      else
      {
         System.out.println("Running...");
      }

      System.out.println("Inputs: " + inputsFileName);
      System.out.println("Truths: " + truthsFileName);
      System.out.println("Weights: " + weightsFileName);

      if (!randomWeight) // if loading weights from a file
      {
         System.out.println("Loading weights from " + weightsFileName);
      }
      if (saveWeights)
      {
         System.out.println("Saving weights to " + weightsFileName + " every " + saveFrequency + " iterations");
      }
   } // echoConfiguration() prints all necessary information for network

   /**
    * Allocates memory for all needed arrays. Separates training specific arrays from
    * run arrays to avoid taking up memory unnecessarily. Instantiates input values array,
    * hidden values array, outputs array, truths, and all weight arrays.
    */
   public void allocateArrays()
   {
      int wRows, wCols, tallestLayer, tallestHiddenLayer, n;

      tallestHiddenLayer = 0;
      for (n = FIRST_HIDDEN_LAYER; n < numHiddenLayers + FIRST_HIDDEN_LAYER; n++)
      {
         tallestHiddenLayer = Math.max(tallestHiddenLayer, numActivationNodes[n]);
      }
      tallestLayer = Math.max(Math.max(numInputs, numOutputs), tallestHiddenLayer);

      if (isTraining) // allocates specific arrays that are only needed for training
      {
         theta = new double[numHiddenLayers + 1][tallestHiddenLayer];
         psi = new double[numLayers][tallestLayer];
      }

      activations = new double[numHiddenLayers + 1 + 1][tallestLayer];
      cases = new double[numCases][numInputs];
      wRows = Math.max(numInputs, tallestHiddenLayer); // max # of nodes from input and hidden layers
      wCols = Math.max(tallestHiddenLayer, numOutputs); // max # of nodes from hidden and output layers
      weights = new double[numHiddenLayers + 1][wRows][wCols];
      truths = new double[numCases][numOutputs];
      outputs = new double[numCases][numOutputs];
   } // allocateArrays() makes memory space for arrays

   /**
    * Returns a random weight between the low and high bounds for the initial training weights.
    *
    * @return random weight as a double between low and high bounds
    */
   public double randomize(double low, double high)
   {
      return Math.random() * (high - low) + low;
   }

   /**
    * Sets the values of all weight arrays. Generates and puts in random or manual weights.
    */
   public void populateWeightArrays()
   {
      int n, m, k;

      if (randomWeight || !loadWeights(weightsFileName))
      {
         for (n = 0; n < numHiddenLayers + 1; n++) // iterates through connecting weights between all layers
         {
            for (m = 0; m < numActivationNodes[n]; m++)
            {
               for (k = 0; k < numActivationNodes[n + 1]; k++)
               {
                  weights[n][m][k] = randomize(dRandMin, dRandMax);
               }
            }
         }
      } // if (randomWeight) populate weight arrays with random weights
   } // populateWeightArrays() which are different for training and running

   /**
    * Only runs one case and calculates and saves hidden layer and output layer
    * values. Multiplies weights with node values then adds them. Puts that theta value
    * into the activation/thresholdFunction function in order to find the next layer's value.
    *
    * @param currentCase int indicating which of the input cases is being evaluated
    */
   public void runOneCase(int currentCase)
   {
      int n, m, k, j, i;
      double theta, fI;

      for (n = 0; n < numLayers - 1; n++)
      {
         for (k = 0; k < numActivationNodes[n + 1]; k++) // computes the theta and hidden values
         {
            theta = 0.0;
            for (m = 0; m < numActivationNodes[n]; m++)
            {
               theta += activations[n][m] * weights[n][m][k];
            }
            activations[n + 1][k] = thresholdFunction(theta);
         } // for (k = 0; k < numActivationNodes[n + 1]; k++)
      } // for (n = 0; n < numLayers - 1; n++)

      for (i = 0; i < numOutputs; i++)
      {
         outputs[currentCase][i] = activations[outputLayerIndex][i];
      } // for (i = 0; i < numOutputs; i++)
   } // runOneCase(int currentCase) calculates output for just one case

   /**
    * Without any training, calculates all layers' values including the output for all
    * four cases. Prints the inputs with their corresponding output.
    */
   public void run()
   {
      int currentCase;

      for (currentCase = 0; currentCase < numCases; currentCase++) // iterates through all four cases to find output
      {
         setUpInputsForCase(currentCase);
         runOneCase(currentCase);
      }
   } // run() calculates output for all four cases

   /**
    * The activation function takes in a value and gives an output.
    *
    * @param value double theta calculated value
    * @return double result of the activation function
    */
   public double thresholdFunction(double value)
   {
      return sigmoidFunction(value);
   }

   /**
    * Calculates the derivative of the activation/threshold function.
    *
    * @param value double activation value
    * @return double calculated derivative value
    */
   public double thresholdDerivative(double value)
   {
      return sigmoidDerivative(value);
   }

   /**
    * The sigmoid function takes the calculated theta values of nodes and turns
    * them into numbers between 0 and 1.
    *
    * @param value double theta calculated value
    * @return number between 0 and 1
    */
   public double sigmoidFunction(double value)
   {
      return 1.0 / (1.0 + StrictMath.exp(-value));
   }

   /**
    * Takes the derivative of the sigmoid function to help in training calculations.
    *
    * @param value double to be put into derivative of sigmoid function
    * @return double value after put through derivative of sigmoid function
    */
   private double sigmoidDerivative(double value)
   {
      double sigmoid = sigmoidFunction(value);
      return sigmoid * (1.0 - sigmoid);
   }

   /**
    * Puts in inputs from the corresponding case.
    * @param currentCase int indicating which of the input cases is being evaluated
    */
   public void setUpInputsForCase(int currentCase)
   {
      int m;

      for (m = 0; m < numInputs; m++) // copy one case of inputs into input node values
      {
         activations[INPUT_LAYER][m] = cases[currentCase][m];
      }
   }

   /**
    * Calculates the error for given case for all outputs.
    * @param currentCase int indicating which of the input cases is being evaluated
    * @return error for output for currentCase
    */
   public double computeErrorForCase(int currentCase)
   {
      int i;
      double outputError = 0.0;

      for (i = 0; i < numOutputs; i++)
      {
         outputError += (truths[currentCase][i] - outputs[currentCase][i])
                 * (truths[currentCase][i] - outputs[currentCase][i]);
      }
      return 0.5 * outputError;
   } // computeErrorForCase(int currentCase)

   /**
    * Helper for train method to perform run for train calculations of
    * thetaj, hj, and output values.
    *
    * @precondition inputs array and truths array already populated for currentCase
    * @param currentCase uses the current case from training loop
    */
   public void runForTrain(int currentCase)
   {
      int n, m, k, j, i;
      double thetaSum, omegaI, fI;

      for (n = 0; n < numLayers - 2; n++)
      {
         for (k = 0; k < numActivationNodes[n + 1]; k++) // computes the theta and hidden values
         {
            thetaSum = 0.0;
            for (m = 0; m < numActivationNodes[n]; m++) // theta summation
            {
               thetaSum += activations[n][m] * weights[n][m][k];
            }
            theta[n + 1][k] = thetaSum;
            activations[n + 1][k] = thresholdFunction(thetaSum);
         }
      } // for (n = 0; n < numLayers - 1; n++)

      for (i = 0; i < numOutputs; i++)
      {
         thetaSum = 0.0;
         for (j = 0; j < numActivationNodes[numLayers - 2]; j++)
         {
            thetaSum += activations[numLayers - 2][j] * weights[numLayers - 2][j][i];
         }
         activations[outputLayerIndex][i] = thresholdFunction(thetaSum);
         omegaI = truths[currentCase][i] - activations[outputLayerIndex][i];
         psi[outputLayerIndex][i] = omegaI * thresholdDerivative(thetaSum);
      } // for (i = 0; i < numOutputs; i++)
   } // runForTrain(int currentCase)

   /**
    * Train breaks down training into several steps to calculate weights that would yield
    * a smaller error when calculating the output. It uses the previous output to calculate error.
    * Calculates the delta weights which are the changes in weight needed to minimize error and get
    * closer to the desired output. Keeps track of error and number of iterations.
    */
   public void train()
   {
      int currentCase, n, x, m, k, j, i;
      double omega, psiLocal, sumError;
      double deltaW;
      long startTimeTrain, endTimeTrain;

      startTimeTrain = currentTimeMillis();

      error = 1.0; // start at 1.0 so it goes into the while loop
      while (epochs < maxEpochs && error > errorThresh) // condition for ending training
      {
         sumError = 0.0;

         for (currentCase = 0; currentCase < numCases; currentCase++) // iterates through all four cases
         {
            setUpInputsForCase(currentCase);

            runForTrain(currentCase);
            for (n = outputLayerIndex - 1; n > 1; n--)
            {
               for (j = 0; j < numActivationNodes[n]; j++)
               {
                  omega = 0.0;
                  for (i = 0; i < numActivationNodes[n + 1]; i++)
                  {
                     omega += psi[n + 1][i] * weights[n][j][i];
                     deltaW = activations[n][j] * psi[n + 1][i] * lambda;
                     weights[n][j][i] += deltaW;
                  }
                  psi[n][j] = omega * thresholdDerivative(theta[n][j]);
               } // for (j = 0; j < numActivationNodes[n]; j++)
            } // for (n = outputLayerIndex - 1; n > 1; n--))

            for (m = 0; m < numActivationNodes[FIRST_HIDDEN_LAYER]; m++)
            {
               omega = 0.0;
               for (k = 0; k < numActivationNodes[FIRST_HIDDEN_LAYER + 1]; k++)
               {
                  omega += psi[FIRST_HIDDEN_LAYER + 1][k] * weights[FIRST_HIDDEN_LAYER][m][k];
                  deltaW = activations[FIRST_HIDDEN_LAYER][m] * psi[FIRST_HIDDEN_LAYER + 1][k] * lambda;
                  weights[FIRST_HIDDEN_LAYER][m][k] += deltaW;
               }

               psiLocal = omega * thresholdDerivative(theta[FIRST_HIDDEN_LAYER][m]);
               for (x = 0; x < numInputs; x++)
               {
                  weights[INPUT_LAYER][x][m] += lambda * activations[INPUT_LAYER][x] * psiLocal;
               }

            } // for (m = 0; m < numActivationNodes[FIRST_HIDDEN_LAYER]; m++)

            runOneCase(currentCase);
            sumError += computeErrorForCase(currentCase);
         } // for (currentCase = 0; currentCase < numCases; currentCase++) iterates through all four cases

         error = sumError / ((double) numCases); // save the average error
         epochs++;
         if (saveWeights && (epochs % saveFrequency) == 0) // save the weights periodically
         {
            saveWeights();
         }
         if (keepAliveFrequency != 0 && (epochs % keepAliveFrequency) == 0) // print a message
         {
            System.out.println("Iteration " + epochs + ", Error = " + error);
         }
      } // while (epochs < maxEpochs && error > errorThresh)

      endTimeTrain = currentTimeMillis();
      timeElapsedTrain = endTimeTrain - startTimeTrain;
   } // train() calculates weights to minimize error for output

   /**
    * Saves weight arrays into a file.
    * @return true if able to save into file
    *         false if there was an issue saving into file
    */
   public boolean saveWeights()
   {
      int n, m, k;

      try
      {
         FileWriter fw = new FileWriter(weightsFileName);

         fw.write(numLayers + "\n");
         for (n = 0; n < numLayers - 1; n++)
         {
            fw.write(numActivationNodes[n] + " ");
         }
         fw.write(numActivationNodes[outputLayerIndex] + "\n");

         for (n = 0; n < numHiddenLayers + 1; n++)
         {
            for (m = 0; m < numActivationNodes[n]; m++)
            {
               for (k = 0; k < numActivationNodes[n + 1]; k++)
               {
                  fw.write(weights[n][m][k] + "\n");
               }
            }
         }

         fw.close();
         return true; // saving into file worked
      } // try saving weight arrays into a file
      catch (IOException e)
      {
         System.out.println("Could not save into file " + weightsFileName);
         return false; // saving into file did not work
      }
   } // public boolean saveWeights(String fileName)

   /**
    * Loads weight array into the network from a file.
    * @param fileName file weight array is coming from
    * @return true if file was loaded successfully for expected configuration
    *         false if file is not compatible or doesn't exist
    */
   public boolean loadWeights(String fileName)
   {
      int n, m, k;
      String line;
      String[] values;

      try
      {
         FileReader fr = new FileReader(fileName);
         BufferedReader br = new BufferedReader(fr);

         line = br.readLine();
         if (!(numHiddenLayers == Integer.parseInt(line) - 2))
         {
            System.out.println("Number of layers does not match weights file.");
            return false;
         }

         values = br.readLine().split(" +", 5); // network type
         for (n = 0; n < numLayers; n++)
         {
            if (numActivationNodes[n] != Integer.parseInt(values[n]))
            {
               System.out.println("Configuration does not match weights file.");
               return false;
            }
         }

         for (n = 0; n < numHiddenLayers + 1; n++) // to read and set weights
         {
            for (m = 0; m < numActivationNodes[n]; m++)
            {
               for (k = 0; k < numActivationNodes[n + 1]; k++)
               {
                  line = br.readLine();
                  weights[n][m][k] = Double.parseDouble(line);
               }
            }
         }

         br.close();
         return true; // loading from file worked
      } // try loading weight from a file
      catch (Throwable e)
      {
         System.out.println("Could not load from file " + fileName + " Using random weights instead");
         return false; // loading from file did not work
      }
   } // public boolean loadWeights(String fileName)

   /**
    * Loads configuration parameters into the network from a file. Contains the names of other files.
    * @param fileName control file with all configuration parameters
    * @return true if reading and loading information from file worked
    *         false if file is not compatible or doesn't exist
    */
   public boolean loadConfiguration(String fileName)
   {
      String[] values;
      int n;

      try
      {
         FileReader fr = new FileReader(fileName);
         BufferedReader br = new BufferedReader(fr);

         values = br.readLine().split(" +"); // network type
         numActivationNodes = new int[values.length];
         for (n = 0; n < values.length; n++)
         {
            numActivationNodes[n] = Integer.parseInt(values[n]);
         }
         numInputs = numActivationNodes[0];
         numHiddenLayers = values.length - 2;
         numLayers = values.length;
         outputLayerIndex = numHiddenLayers + 1;
         numOutputs = numActivationNodes[outputLayerIndex];

         values = br.readLine().split(" +", 2); // inputsFileNamePrefix
         inputsFileName = values[0];

         values = br.readLine().split(" +", 2); // isTraining
         isTraining = Boolean.parseBoolean(values[0]);

         values = br.readLine().split(" +", 2); // randomWeight
         randomWeight = Boolean.parseBoolean(values[0]);

         values = br.readLine().split(" +", 2); // weightsFileName
         weightsFileName = values[0];

         values = br.readLine().split(" +", 2); // saveWeights after training
         saveWeights = Boolean.parseBoolean(values[0]);

         values = br.readLine().split(" +", 2); // save frequency during training
         saveFrequency = Integer.parseInt(values[0]);

         values = br.readLine().split(" +", 2); // keepAlive display frequency during training
         keepAliveFrequency = Integer.parseInt(values[0]);

         values = br.readLine().split(" +", 2); // dRandMin
         dRandMin = Double.parseDouble(values[0]);

         values = br.readLine().split(" +", 2); // dRandMax
         dRandMax = Double.parseDouble(values[0]);

         values = br.readLine().split(" +", 2); // lambda
         lambda = Double.parseDouble(values[0]);

         values = br.readLine().split(" +", 2); // maxEpochs
         maxEpochs = Integer.parseInt(values[0]);

         values = br.readLine().split(" +", 2); // errorThreshold
         errorThresh = Double.parseDouble(values[0]);

         values = br.readLine().split(" +", 2); // truthsFileName
         truthsFileName = values[0];

         values = br.readLine().split(" +", 2); // numCases
         numCases = Integer.parseInt(values[0]);

         br.close();
         return true;
      } // try reading control file
      catch (IOException e)
      {
         System.out.println(e);
         return false;
      }
   } // loadConfiguration(String fileName)

   /**
    * Loads inputs into the network from a file for a certain configuration.
    * @param fileName text file with inputs
    * @return true if reading and loading information from file worked
    *         false if file is not compatible or doesn't exist
    */
   public boolean loadInputs(String fileName)
   {
      String[] values;
      int currentCase, m;

      try
      {
         FileReader fr = new FileReader(fileName);
         BufferedReader br = new BufferedReader(fr);

         for (currentCase = 0; currentCase < numCases; currentCase++)
         {
            values = br.readLine().split(" +", numInputs);
            for (m = 0; m < numInputs; m++)
            {
               cases[currentCase][m] = Double.parseDouble(values[m]);
            }
         } // for (currentCase = 0; currentCase < numCases; currentCase++)

         br.close();
         return true;
      } // try reading inputs file
      catch (IOException e)
      {
         System.out.println(e);
         return false;
      }
   } // loadInputs(String fileName)

   /**
    * Loads truths into the network from a file for a certain configuration.
    * @param fileName text file with truths
    * @return true if reading and loading information from file worked
    *         false if file is not compatible or doesn't exist
    */
   public boolean loadTruths(String fileName)
   {
      String[] values;
      int currentCase, i;

      try
      {
         FileReader fr = new FileReader(fileName);
         BufferedReader br = new BufferedReader(fr);

         for (currentCase = 0; currentCase < numCases; currentCase++) // read and copy truths from file
         {
            values = br.readLine().split(" +", numOutputs);
            for (i = 0; i < numOutputs; i++)
            {
               truths[currentCase][i] = Double.parseDouble(values[i]);
            }
         } // for (currentCase = 0; currentCase < numCases; currentCase++)

         br.close();
         return true;
      } // try reading truths file
      catch (IOException e)
      {
         System.out.println(e);
         return false;
      }
   } // loadTruths(String fileName)

   /**
    * Prints report information after training or running. Includes truth table, outputs, error,
    * number of iterations, reason for ending training, time to train, and weights file saved to.
    */
   public void reportResults()
   {
      int currentCase, m, i;

      System.out.println("Inputs     Truths     F");
      System.out.println("------    --------    ----");
      for (currentCase = 0; currentCase < numCases; currentCase++) // prints truth table
      {
         for (m = 0; m < numInputs; m++)
         {
            System.out.print(" " + Math.round(cases[currentCase][m]));
         }

         System.out.print("    ");
         for (i = 0; i < numOutputs; i++)
         {
            System.out.print(Math.round(truths[currentCase][i]) + "    ");
         }

         for (i = 0; i < numOutputs; i++)
         {
            System.out.print(String.format("%1.18f   ", outputs[currentCase][i])); // change decimals of outputs
         }

         System.out.println();
      } // for (currentCase = 0; currentCase < numCases; currentCase++)

      if (isTraining) // only needed for training print iterations, error, and reason for end training
      {
         if (epochs >= maxEpochs) // prints if max iterations reached
         {
            System.out.println("Maximum iterations reached!");
         }

         if (error <= errorThresh) // prints if min error reached
         {
            System.out.println("Minimum error reached!");
         }

         System.out.println("Average Error: " + error);
         System.out.println("Number of Iterations: " + epochs);
         System.out.println("Train Time: " + timeElapsedTrain + " milliseconds");
      } // if(isTraining)
   } // reportResults() prints info after training/running

   /**
    * Tests the training and running of the network with all corresponding configurations.
    *
    * @param args allows command-line arguments to be passed to the program
    */
   public static void main(String[] args)
   {
      NNImage n = new NNImage();
      String controlFileName;

      if (args.length < 1)
      {
         System.out.println("Control file not specified, using default control_2-B-C-3.txt");
         controlFileName = DEFAULT_CONTROL_FILENAME;
      }
      else
      {
         controlFileName = args[0];
      }

      if (!n.loadConfiguration(controlFileName)) // load config from file
      {
         System.err.println("loadConfiguration failed.");
      }
      else
      {
         n.echoConfiguration();
         n.allocateArrays();

         if (!n.loadTruths(n.truthsFileName)) // load truths from file
         {
            System.err.println("loadTruths failed.");
         }
         else
         {
            if (!n.loadInputs(n.inputsFileName)) // load inputs from file
            {
               System.err.println("loadInputs failed.");
            }
            else
            {
               n.populateWeightArrays();

               if (n.isTraining) // if training the network
               {
                  n.train();
                  n.run(); // recompute the outputs for the final weights

                  if (n.saveWeights)
                  {
                     n.saveWeights();
                  }
               }
               else // if running the network without training
               {
                  n.run();
               }

               n.reportResults();
            } // else loaded inputs
         } // else loaded truths
      } // else proceed when loadConfiguration works
   } // main(String[] args) calls all the methods to test training and running
} // NNImage class trains an N layer network using backpropagation