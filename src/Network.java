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
        for (int i = 0; i < layerSize; i++) {
            ArrayList<Float> nodeWeights = new ArrayList<>();
            float bias = (float)(Main.RANDOM.nextFloat() - 0.5);
            for (int j = 0; j < inputSize; j++) {
                nodeWeights.add((float)(Main.RANDOM.nextFloat() - 0.5));
            }

            weightMatrix.add(nodeWeights);
            biases.add(bias);
        }
        layer.setWeightMatrix(weightMatrix);
        layer.setBiases(biases);
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
            float activation = activation(num);
            //Derivative of sigmoid is sigmoid * (1 - sigmoid)
            return (float)(activation * (1 - activation));
        }
        else if (activationFunction.equals("relu")) {
            return (num>0) ? 1 : 0;
        }
        throw new IllegalArgumentException("Activation function: " + activationFunction + ". But needs to be either 'relu' or 'sigmoid'.");
    }

    public Gradient createGradient(ArrayList<Float> input, ArrayList<Float> expectedOutput) {
        calculateOutput(input);

        //Layer --> Node --> Weight
        ArrayList<ArrayList<ArrayList<Float>>> weightGradients = new ArrayList<>();
        //Layer --> Bias (belongs to a single node))
        ArrayList<ArrayList<Float>> biasGradients = new ArrayList<>();
        ArrayList<Float> dout = new ArrayList<>();

        //Calculate dout for the last layer
        Layer lastLayer = layers.get(layers.size() - 1);
        ArrayList<Float> lastActivations = lastLayer.getLastActivations();
        ArrayList<Float> lastLogits = lastLayer.getLastLogits();
        for (int i = 0; i < lastActivations.size(); i++) {
            float dCostWRTActivation = 2f * (lastActivations.get(i) - expectedOutput.get(i));
            dout.add(dCostWRTActivation * this.dActivation(lastLogits.get(i)));
        }

        //Calculate gradients for each layer
        for (int i = layers.size() - 1; i >= 0; i--) {
            Layer layer = layers.get(i);
            ArrayList<Float> prevActivations = (i == 0) ? input : layers.get(i - 1).getLastActivations();
            ArrayList<Float> prevLogits = (i == 0) ? input : layers.get(i - 1).getLastLogits();
            // Calculate gradients for this layer
            ArrayList<ArrayList<Float>> dWeights = layer.dCostWRTWeights(prevActivations, dout);
            ArrayList<Float> dBiases = layer.dCostWRTBiases(dout);

            weightGradients.add(0, dWeights);
            biasGradients.add(0, dBiases);

            // Calculate dout for the next layer
            dout = layer.dCostWRTPrevActivations(dout, prevActivations, prevLogits);
        }

        return new Gradient(weightGradients, biasGradients);
    }

    public void updateWeightsAndBiases(Gradient gradient, float learningRate) {
        for (int i = 0; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            ArrayList<ArrayList<Float>> weightGradients = gradient.getWeightGradients().get(i);
            ArrayList<Float> biasGradients = gradient.getBiasGradients().get(i);

            // Update weights
            ArrayList<ArrayList<Float>> newWeights = new ArrayList<>();
            for (int j = 0; j < layer.getWeightMatrix().size(); j++) {
                ArrayList<Float> newWeightsForNode = new ArrayList<>();
                for (int k = 0; k < layer.getWeightMatrix().get(j).size(); k++) {
                    float newWeight = layer.getWeightMatrix().get(j).get(k) - learningRate * weightGradients.get(j).get(k);
                    newWeightsForNode.add(newWeight);
                }
                newWeights.add(newWeightsForNode);
            }
            layer.setWeightMatrix(newWeights);

            // Update biases
            ArrayList<Float> newBiases = new ArrayList<>();
            for (int j = 0; j < layer.getBiases().size(); j++) {
                float newBias = layer.getBiases().get(j) - learningRate * biasGradients.get(j);
                newBiases.add(newBias);
            }
            layer.setBiases(newBiases);
        }
    }
    
}