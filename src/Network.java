import java.util.ArrayList;

public class Network {

    private ArrayList<Layer> layers;
    private String activationFunction;

    public Network() {
        this.layers = new ArrayList<>();
        this.activationFunction = "sigmoid";
    }
    public Network(String activationFunction) {
        this.layers = new ArrayList<>();
        this.activationFunction = activationFunction;
    }

    public void addLayer(Layer layer){
        layers.add(layer);
    }

    public void addNewLayer(int inputSize, int layerSize) {
        Layer layer = new Layer(this, inputSize);
        ArrayList<ArrayList<Float>> weightMatrix = new ArrayList<>();
        ArrayList<Float> biases = new ArrayList<>();
        for (int i = 1; i <= layerSize; i++) {
            ArrayList<Float> nodeWeights = new ArrayList<>();
            float bias = (float)(Main.RANDOM.nextFloat() - 0.5);
            for (int j = 1; j <= inputSize; j++) {
                nodeWeights.add((float)(Main.RANDOM.nextFloat() - 0.5));
            }

            weightMatrix.add(nodeWeights);
            biases.add(bias);
            layer.setWeightMatrix(weightMatrix);
            layer.setBiases(biases);
        }
        this.addLayer(layer);
    }

    public float cost(ArrayList<ArrayList<Float>> examples, ArrayList<ArrayList<Float>> expectedOutputs) {
        if (examples.size() != expectedOutputs.size()) {
            throw new IllegalArgumentException("Examples size must equal expectedOutputs size. Examples size: " + examples.size() + ". expectedOutputs Size: " + expectedOutputs.size());
        }
        ArrayList<Float> output;
        float allExamplesSum = 0f;
        for (int i = 0; i < examples.size(); i++) {
            float singleExampleSum = 0f;
            output = calculateOutput(examples.get(i));
            for (int j = 0; j < output.size(); j++) {
                float difference = (output.get(j) - expectedOutputs.get(i).get(j));
                singleExampleSum += difference*difference;
            }
            allExamplesSum += singleExampleSum;
        }
        return allExamplesSum/((float)(examples.size()));
    }

    public ArrayList<ArrayList<Float>> getCache(ArrayList<Float> input) {
        ArrayList<ArrayList<Float>> output = new ArrayList<>();
        ArrayList<Float> currentOutput = input;
        for (Layer layer : layers) {
            output.add(currentOutput);
            currentOutput = layer.calculateActivations(currentOutput);
        }
        output.add(currentOutput);
        return output;
    }
    public ArrayList<Float> calculateOutput(ArrayList<Float> input) {
        //CHECK FOR CORRECT INPUT SIZE
        ArrayList<Float> currentOutput = input;
        for (Layer layer : layers) {
            currentOutput = layer.calculateActivations(currentOutput);
        }
        return currentOutput;
    }

//    public ArrayList<Float> calculateGradient(ArrayList<Float> input) {
//
//    }

    public float activation(float num) {
        if (activationFunction.equals("sigmoid")) {
            return (float)(1.0 / (1 + Math.exp(-num)));
        }
        else if (activationFunction.equals("relu")) {
            return Math.max(0f, num);
        }
        throw new IllegalArgumentException("Activation function: " + activationFunction + ". But needs to be either 'relu' or 'sigmoid'.");
    }

    public float dActivation(float num) {
        if (activationFunction.equals("sigmoid")) {
            return (float)(activation(num)*(1-activation(num)));
        }
        else if (activationFunction.equals("relu")) {
            return (num>0) ? 1 : 0;
        }
        throw new IllegalArgumentException("Activation function: " + activationFunction + ". But needs to be either 'relu' or 'sigmoid'.");
    }

    
}