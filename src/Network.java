import java.util.ArrayList;

public class Network {

    private ArrayList<Layer> layers;

    public Network() {
        this.layers = new ArrayList<>();
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
        int index = 0;
        ArrayList<Float> currentOutput = input;
        for (Layer layer : layers) {
            currentOutput = layer.calculateOutput(currentOutput);
        }
        float total = 0f;
        for (float activation : currentOutput) {
            total += activation;
        }
        for(int i = 0; i < currentOutput.size(); i++) {
            currentOutput.set(i, currentOutput.get(i)/total);
        }

        return currentOutput;
    }

    public float sigmoid(float num) {
        return (float)(1.0 / (1 + Math.exp(-num)));
    }
    public float relu(float num) {
        return Math.max(0f, num);
    }
}