import java.util.ArrayList;

public class Layer {
    private final int inputSize;
    private ArrayList<Node> nodes;

    private final Network network;

    public Layer(Network network, int inputSize) {
        this.inputSize = inputSize;
        this.network = network;
        this.nodes = new ArrayList<>();
    }

    public void addNode(Node node) {
        nodes.add(node);
    }

    public void drawOutput() {
        //draw each node
    }

    public ArrayList<Float> calculateOutput(ArrayList<Float> input) {
        if (input.size() != inputSize) {
            throw new IllegalArgumentException("Input size must be " + inputSize + ", but got " + input.size());
        }
        ArrayList<Float> output = new ArrayList<>();
        for (Node node : nodes) {
            output.add(node.calculateOutput(input));
        }
        return output;
    }

    public int getInputSize() {
        return inputSize;
    }

    public int size() {
        return nodes.size();
    }

    public Network getNetwork() {
        return network;
    }
}
