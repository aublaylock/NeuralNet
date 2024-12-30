import java.util.ArrayList;
import java.util.Random;

public class Main {
    public static int INPUT_SIZE = 16;
    public static int OUTPUT_SIZE = 10;
    public static Random RANDOM = new Random();

    public static void main(String[] args) {
        Random random = new Random();

        //CREATE TESTCASES
//        ArrayList<ArrayList<Float>> examples = new ArrayList<>();
//        ArrayList<ArrayList<Float>> expectedOutputs = new ArrayList<>();
//        for (int i = 1; i <= 10; i++) {
//            ArrayList<Float> example = new ArrayList<>();
//            ArrayList<Float> expectedOutputs = new ArrayList<>();
//            for (int j = 1; j <= INPUT_SIZE; j++) {
//                example.add(1.0f / ((float)j));
//            }
//            examples.add(example);
//        }
        //END CREATE TESTCASE

        //Create a single test case
        ArrayList<Float> example = new ArrayList<>();
        ArrayList<Float> expectedOutput = new ArrayList<>();
        for (int j = 1; j <= INPUT_SIZE; j++) {
            example.add(1.0f / ((float)j));
            expectedOutput.add(1f);
        }
        //END Create a single test case


        //BUILD NETWORK & PRINT OUTPUT
        Network network = buildNetwork(INPUT_SIZE, 1, 10, OUTPUT_SIZE);
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
