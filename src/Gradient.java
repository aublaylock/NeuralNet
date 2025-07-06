import java.util.ArrayList;

public class Gradient {
    private ArrayList<ArrayList<ArrayList<Float>>> weightGradients;
    private ArrayList<ArrayList<Float>> biasGradients;

    public Gradient(ArrayList<ArrayList<ArrayList<Float>>> weightGradients, ArrayList<ArrayList<Float>> biasGradients) {
        this.weightGradients = weightGradients;
        this.biasGradients = biasGradients;
    }

    public ArrayList<ArrayList<ArrayList<Float>>> getWeightGradients() {
        return weightGradients;
    }

    public ArrayList<ArrayList<Float>> getBiasGradients() {
        return biasGradients;
    }

    public void setWeightGradients(ArrayList<ArrayList<ArrayList<Float>>> weightGradients) {
        this.weightGradients = weightGradients;
    }

    public void setBiasGradients(ArrayList<ArrayList<Float>> biasGradients) {
        this.biasGradients = biasGradients;
    }
}