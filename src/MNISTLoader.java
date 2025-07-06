import java.io.*;
import java.util.ArrayList;

public class MNISTLoader {
    public static ArrayList<ArrayList<Float>> loadImages(String file, int max) throws IOException {
        DataInputStream dis = new DataInputStream(new FileInputStream(file));
        int magic = dis.readInt();
        int numImages = dis.readInt();
        int numRows = dis.readInt();
        int numCols = dis.readInt();
        ArrayList<ArrayList<Float>> images = new ArrayList<>();
        for (int i = 0; i < Math.min(numImages, max); i++) {
            ArrayList<Float> image = new ArrayList<>();
            for (int j = 0; j < numRows * numCols; j++) {
                int pixel = dis.readUnsignedByte();
                image.add(pixel / 255.0f); // normalize to [0,1]
            }
            images.add(image);
        }
        dis.close();
        return images;
    }

    public static ArrayList<ArrayList<Float>> loadLabels(String file, int max, int numClasses) throws IOException {
        DataInputStream dis = new DataInputStream(new FileInputStream(file));
        int magic = dis.readInt();
        int numLabels = dis.readInt();
        ArrayList<ArrayList<Float>> labels = new ArrayList<>();
        for (int i = 0; i < Math.min(numLabels, max); i++) {
            int label = dis.readUnsignedByte();
            ArrayList<Float> oneHot = new ArrayList<>();
            for (int j = 0; j < numClasses; j++) {
                oneHot.add(j == label ? 1.0f : 0.0f);
            }
            labels.add(oneHot);
        }
        dis.close();
        return labels;
    }
}