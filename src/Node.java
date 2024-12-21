import java.awt.*;
import java.util.ArrayList;

public class Node {
    private final Layer layer;
    private ArrayList<Float> weights;
    private float bias;

    public Node(Layer layer) {
        this.layer = layer;
    }

    public void setWeights(ArrayList<Float> weights) {
        if (weights.size() != layer.getInputSize()) {
            throw new IllegalArgumentException("Weights size must be " + layer.getInputSize() + ", but was " + weights.size());
        }
        this.weights = weights;
    }

    public void setBias(float bias) {
        this.bias = bias;
    }

    public Float calculateOutput(ArrayList<Float> input) {
        if (input.size() != layer.getInputSize()) {
            throw new IllegalArgumentException("Input size must be " + layer.getInputSize() + ", but was " + input.size());
        }
        float sum = 0.0f;
        int index = 0;
        for (Float inputValue : input) {
            sum += weights.get(index) * inputValue;
            index++;
        }
        sum += bias;
        return layer.getNetwork().relu(sum);
    }
}
