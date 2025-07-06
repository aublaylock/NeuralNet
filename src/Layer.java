import java.util.ArrayList;

public class Layer {
    private final int inputSize;

    private ArrayList<ArrayList<Float>> weightMatrix;

    private ArrayList<Float> biases;
    private final Network network;

    private ArrayList<Float> lastActivations; // after activation
    private ArrayList<Float> lastLogits;      // before activation 

    public Layer(Network network, int inputSize) {
        this.inputSize = inputSize;
        this.network = network;
    }

    public ArrayList<Float> calculateOutput(ArrayList<Float> inputs) {
        if (inputs.size() != inputSize) {
            throw new IllegalArgumentException("Input size must be " + inputSize + ", but got " + inputs.size());
        }
        ArrayList<Float> logits = new ArrayList<>();
        ArrayList<Float> outputs = new ArrayList<>();
        int nodeIndex = 0;
        for (ArrayList<Float> nodeWeights : weightMatrix) {
            int inputIndex = 0;
            float sum = 0f;
            for (Float input : inputs) {
                sum += input * nodeWeights.get(inputIndex);
                inputIndex++;
            }
            float correspondingBias = biases.get(nodeIndex);
            sum += correspondingBias;
            logits.add(sum);
            //Apply activation function to the sum
            outputs.add(getNetwork().activation(sum));
            nodeIndex++;
        }
        // System.out.println("Shape: " + this.weightMatrix.size() + " x " + this.weightMatrix.getFirst().size());
        this.lastLogits = logits;
        this.lastActivations = outputs;
        return outputs;
    }

    //INITIAL DERIVATIVES
    public ArrayList<Float> dCostWRTPrevActivations(ArrayList<Float> dout, ArrayList<Float> prevActivations, ArrayList<Float> prevLogits, ArrayList<ArrayList<Float>> weightMatrix) {
        if (prevLogits.isEmpty()) {
            // No previous logits for the input layer, return zeros or skip
            return new ArrayList<>();
        }

        ArrayList<Float> output = new ArrayList<>();
        //For each neuron in previous layer
        for (int i = 0; i < prevActivations.size(); i++) {
            //Calculate the derivative of the cost with respect to this neuron's activation
            // sum of influences through each node in this layer
            float sum = 0f;
            //iterate over each weight connecting node in prevLayer to this layer
            for (int j = 0; j < this.size(); j++) {
                //Add the derivative (chain rule)

                //Same as just the weight we are talking about
                float dLogitWRTPrevActivation = weightMatrix.get(j).get(i);
                //Same as just derivative of the activation function of the logit of the node we are deriving
                float dActivationWRTLogit = network.dActivation(prevLogits.get(i));
                //Same as just dout
                float dCostWRTActivation = dout.get(j);


                sum += dLogitWRTPrevActivation * dActivationWRTLogit * dCostWRTActivation;
            }
            output.add(sum);
        }
        return output;
    }

    public ArrayList<ArrayList<Float>> dCostWRTWeights(ArrayList<Float> prevActivations, ArrayList<Float> douts) {
        ArrayList<ArrayList<Float>> gradients = new ArrayList<>();
        for (int nodeIndex = 0; nodeIndex < weightMatrix.size(); nodeIndex++) {
            ArrayList<Float> nodeGradient = new ArrayList<>();
            for (int inputIndex = 0; inputIndex < prevActivations.size(); inputIndex++) {
                float partial = prevActivations.get(inputIndex) * douts.get(nodeIndex);
                nodeGradient.add(partial);
            }
            gradients.add(nodeGradient);
        }
        return gradients;
    }

    public ArrayList<Float> dCostWRTBiases(ArrayList<Float> prevActivations, ArrayList<Float> douts) {
        ArrayList<Float> gradient = new ArrayList<>();
        for (int nodeIndex = 0; nodeIndex < biases.size(); nodeIndex++) {
            float partial = douts.get(nodeIndex);
            gradient.add(partial);
        }
        return gradient;
    }



    public Network getNetwork() {
        return network;
    }
    public void setWeightMatrix(ArrayList<ArrayList<Float>> weightMatrix) {
        this.weightMatrix = weightMatrix;
    }
    public void setBiases(ArrayList<Float> biases) {
        this.biases = biases;
    }
    public int size() {
        return this.biases.size();
    }
    public ArrayList<ArrayList<Float>> getWeightMatrix() {
        return weightMatrix;
    }
    public ArrayList<Float> getBiases() {
        return biases;
    }
    public ArrayList<Float> getLastActivations() {
        return lastActivations;
    }
    public ArrayList<Float> getLastLogits() {
        return lastLogits;
    }
}
