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

    public ArrayList<Float> calculateActivations(ArrayList<Float> inputs) {
        ArrayList<Float> logits = calculateLogits(inputs);
        ArrayList<Float> activations = new ArrayList<>();
        for (float logit : logits) {
            activations.add(getNetwork().activation(logit));
        }
        this.lastActivations = activations;
        return activations;
    }

    public ArrayList<Float> calculateLogits(ArrayList<Float> inputs) {
    if (inputs.size() != inputSize) {
        throw new IllegalArgumentException("Input size must be " + inputSize + ", but got " + inputs.size());
    }
    ArrayList<Float> logits = new ArrayList<>();
    for (ArrayList<Float> nodeWeights : weightMatrix) {
        float sum = 0f;
        for (int i = 0; i < inputs.size(); i++) {
            sum += inputs.get(i) * nodeWeights.get(i);
        }
        sum += biases.get(logits.size());
        logits.add(sum);
    }
    this.lastLogits = logits;
    return logits;
    }

    //INITIAL DERIVATIVES
    public ArrayList<Float> dCostWRTPrevActivations(ArrayList<Float> dout, ArrayList<Float> prevActivations, ArrayList<Float> logits) {
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
                float dLogitWRTPrevActivation = this.weightMatrix.get(j).get(i);
                //Same as just derivative of the activation function of the logit of the node we are deriving
                float dActivationWRTLogit = network.dActivation(logits.get(j));
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

    public ArrayList<Float> dCostWRTBiases(ArrayList<Float> douts) {
        ArrayList<Float> gradient = new ArrayList<>();
        for (int nodeIndex = 0; nodeIndex < biases.size(); nodeIndex++) {
                gradient.add(douts.get(nodeIndex));
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
