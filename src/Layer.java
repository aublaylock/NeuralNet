import java.util.ArrayList;

public class Layer {
    private final int inputSize;

    private ArrayList<ArrayList<Float>> weightMatrix;

    private ArrayList<Float> biases;
    private final Network network;

    public Layer(Network network, int inputSize) {
        this.inputSize = inputSize;
        this.network = network;
    }

    public ArrayList<Float> calculateActivations(ArrayList<Float> inputs) {
        ArrayList<Float> logits = calculateLogits(inputs);
        ArrayList<Float> activations = new ArrayList<>();
        for (int i = 0; i < logits.size(); i++) {
            activations.add(getNetwork().activation(logits.get(i)));
        }
        //System.out.println("Shape: " + this.weightMatrix.size() + " x " + this.weightMatrix.get(0).size());
        return activations;
    }

    public ArrayList<Float> calculateLogits(ArrayList<Float> inputs) {
        if (inputs.size() != inputSize) {
            throw new IllegalArgumentException("Input size must be " + inputSize + ", but got " + inputs.size());
        }
        ArrayList<Float> outputs = new ArrayList<>();
        int nodeIndex = 0;
        for (ArrayList<Float> nodeWeights : weightMatrix) {
            int inputIndex = 0;
            float sum = 0f;
            float correspondingWeight = nodeWeights.get(inputIndex);
            for (Float input : inputs) {
                sum += input * correspondingWeight;
                inputIndex++;
            }
            float correspondingBias = biases.get(nodeIndex);
            sum += correspondingBias;
            outputs.add(sum);
            nodeIndex++;
        }
//        System.out.println("Shape: " + this.weightMatrix.size() + " x " + this.weightMatrix.get(0).size());
        return outputs;
    }

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

    public ArrayList<ArrayList<Float>> dCostWRTWeights(ArrayList<Float> dout, ArrayList<Float> prevActivations, ArrayList<Float> logits) {
        ArrayList<ArrayList<Float>> output = new ArrayList<>();
        //For each neuron in previous layer
        for (int i = 0; i < prevActivations.size(); i++) {
            // stores the derivatives wrt to each weight attached to a node in this layer
            ArrayList<Float> nodeWeightDerivatives = new ArrayList<>();
            //iterate over each weight connecting node in prevLayer to this layer
            for (int j = 0; j < this.size(); j++) {
                //calculate the derivative (chain rule) for each weight connected to this node

                //Same as just the weight we are talking about
                float dLogitWRTWeight = prevActivations.get(i);
                //Same as just derivative of the activation function of the logit of the node we are deriving
                float dActivationWRTLogit = network.dActivation(logits.get(j));
                //Same as just dout
                float dCostWRTActivation = dout.get(j);

                float dCostWRTWeight = dLogitWRTWeight * dActivationWRTLogit * dCostWRTActivation;
                nodeWeightDerivatives.add(dCostWRTWeight);
            }
            output.add(nodeWeightDerivatives);
        }
        return output;
    }

    public ArrayList<Float> dCostWRTBiases(ArrayList<Float> dout, ArrayList<Float> logits) {
        ArrayList<Float> output = new ArrayList<>();

        //iterate over each weight connecting node in prevLayer to this layer
        for (int j = 0; j < this.size(); j++) {

            //Same as just derivative of the activation function of the logit of the node we are deriving
            float dActivationWRTLogit = network.dActivation(logits.get(j));
            //Same as just dout
            float dCostWRTActivation = dout.get(j);

            float dCostWRTBias = dActivationWRTLogit * dCostWRTActivation;

            output.add(dCostWRTBias);
        }
        return output;
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
    public float size() {
        return this.biases.size();
    }
}
