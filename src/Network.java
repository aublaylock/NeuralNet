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

    public void drawOutput(/* ArrayList<Float> input */) {
        //Draw input layer
        //Draw hidden layers
    }

    //NOT SURE IF THIS WORKS, I HAVEN'T TESTED IT
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

//    public ArrayList<Float> calculateGradient(ArrayList<ArrayList<Float>> examples, ArrayList<ArrayList<Float>> expectedOutputs,) {
//
//    }

    public ArrayList<Float> calculateOutput(ArrayList<Float> input) {
        //CHECK FOR CORRECT INPUT SIZE
        ArrayList<Float> currentOutput = input;
        for (Layer layer : layers) {
            currentOutput = layer.calculateOutput(currentOutput);
        }
        return currentOutput;
    }

    public float activation(float num) {
        if (activationFunction.equals("sigmoid")) {
            return (float)(1.0 / (1 + Math.exp(-num)));
        }
        else if (activationFunction.equals("relu")) {
            return Math.max(0f, num);
        }
        throw new IllegalArgumentException("Activation function: " + activationFunction + ". But needs to be either 'relu' or 'sigmoid'.");
    }
    
}