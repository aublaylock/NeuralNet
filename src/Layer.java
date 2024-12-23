import java.util.ArrayList;

public class Layer {
    private final int inputSize;

    private ArrayList<ArrayList<Float>> weights;

    private ArrayList<Float> biases;
    private final Network network;

    public Layer(Network network, int inputSize) {
        this.inputSize = inputSize;
        this.network = network;
    }

    public void drawOutput() {
        //draw each node
    }

    public ArrayList<Float> calculateOutput(ArrayList<Float> inputs) {
        if (inputs.size() != inputSize) {
            throw new IllegalArgumentException("Input size must be " + inputSize + ", but got " + inputs.size());
        }
        ArrayList<Float> outputs = new ArrayList<>();
        int nodeIndex = 0;
        for (ArrayList<Float> nodeWeights : weights) {
            int inputIndex = 0;
            float sum = 0f;
            float correspondingWeight = nodeWeights.get(inputIndex);
            for (Float input : inputs) {
                sum += input * correspondingWeight;
                inputIndex++;
            }
            float correspondingBias = biases.get(nodeIndex);
            sum += correspondingBias;
            outputs.add(getNetwork().activation(sum));
            nodeIndex++;
        }
        return outputs;
    }

    public int getInputSize() {
        return inputSize;
    }

    public Network getNetwork() {
        return network;
    }
    public void setWeights(ArrayList<ArrayList<Float>> weights) {
        this.weights = weights;
    }
    public void setBiases(ArrayList<Float> biases) {
        this.biases = biases;
    }
}
