import java.util.ArrayList;
import java.util.random.*;

import javax.swing.SwingUtilities;

public class Main {
    public static int INPUT_SIZE = 784; // 28x28 pixels for MNIST
    public static int OUTPUT_SIZE = 10;
    public static java.util.Random RANDOM = new java.util.Random();
    public static void main(String[] args) {

        

/**
        //BUILD NETWORK & PRINT OUTPUT
        Network network = buildNetwork(INPUT_SIZE, 2, 10, OUTPUT_SIZE);
        ArrayList<Float> output = network.calculateOutput(example);
        System.out.println("output: " + output);
        System.out.println("cache: " + network.getCache(example));
        //END BUILD NETWORK & PRINT OUTPUT

        //TEST COST
        ArrayList<ArrayList<Float>> examples = new ArrayList<>();
        examples.add(example);
        ArrayList<ArrayList<Float>> expectedOutputs = new ArrayList<>();
        expectedOutputs.add(expectedOutput);
        System.out.println("cost: " + network.cost(examples, expectedOutputs));
        //END TEST COST

    }
*/
        //LOAD MNIST DATA
        ArrayList<ArrayList<Float>> trainImages = null;
        ArrayList<ArrayList<Float>> trainLabels = null;

    try {
        int maxExamples = 10000; // for quick testing
        trainImages = MNISTLoader.loadImages("/Users/austin/IdeaProjects/Basic Neural Network/data/train-images.idx3-ubyte", maxExamples);
        trainLabels = MNISTLoader.loadLabels("/Users/austin/IdeaProjects/Basic Neural Network/data/train-labels.idx1-ubyte", maxExamples, OUTPUT_SIZE);
    } 
    catch (Exception e) {
        e.printStackTrace();
    }


        //BUILD NETWORK
        Network network = buildNetwork(INPUT_SIZE, 2, 50, OUTPUT_SIZE);

        //TRAIN NETWORK/
        int epochs = 5; // Lower for faster testing
        float learningRate = 0.01f;
        for (int epoch = 0; epoch < epochs; epoch++) {
            float totalCost = 0f;
            for (int i = 0; i < trainImages.size(); i++) {
                ArrayList<Float> example = trainImages.get(i);
                ArrayList<Float> expectedOutput = trainLabels.get(i);

                // Backpropagation: compute gradients and update weights/biases
                Gradient gradient = network.createGradient(example, expectedOutput);
                network.updateWeightsAndBiases(gradient, learningRate);

                // Only calculate cost once per example (after update)
                float cost = network.cost(
                    new ArrayList<>(java.util.List.of(example)),
                    new ArrayList<>(java.util.List.of(expectedOutput))
                );
                totalCost += cost;
            }
            if (epoch % 2 == 0) { // Print every 2 epochs
                System.out.println("Epoch " + epoch + " average cost: " + (totalCost / trainImages.size()));
            }
        }
        //END TRAIN NETWORK

        SwingUtilities.invokeLater(() -> {
            new DigitDrawFrame(network).setVisible(true);
        });

    }

    private static Network buildNetwork (int inputSize, int numHiddenLayers, int hiddenLayerSize, int outputSize) {
        Network network = new Network();
        //First Hidden Layer
        network.addNewLayer(inputSize, hiddenLayerSize);
        //Other Hidden Layers
        for (int i = 2; i <= numHiddenLayers; i++) {
            network.addNewLayer(hiddenLayerSize, hiddenLayerSize);
        }
        //Output Layer
        network.addNewLayer(hiddenLayerSize, outputSize);
        return network;
    }


}
