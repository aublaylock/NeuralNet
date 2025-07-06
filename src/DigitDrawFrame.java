import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.util.ArrayList;

public class DigitDrawFrame extends JFrame {
    private final int gridSize = 28;
    private final float[][] pixels = new float[gridSize][gridSize];
    private final JPanel drawPanel;
    private final int pixelSize = 16; // Size of each pixel on screen
    private final JLabel predictionLabel = new JLabel("Prediction: ");
    private boolean isErasing = false;

    public DigitDrawFrame(Network network) {
        setTitle("Draw a Digit (28x28 Grid)");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(gridSize * pixelSize + 20, gridSize * pixelSize + 100);

        drawPanel = new JPanel() {
            @Override
            protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                for (int y = 0; y < gridSize; y++) {
                    for (int x = 0; x < gridSize; x++) {
                        float value = pixels[y][x];
                        int gray = 255 - (int)(value * 255);
                        g.setColor(new Color(gray, gray, gray));
                        g.fillRect(x * pixelSize, y * pixelSize, pixelSize, pixelSize);
                        g.setColor(Color.LIGHT_GRAY);
                        g.drawRect(x * pixelSize, y * pixelSize, pixelSize, pixelSize);
                    }
                }
            }
        };
        drawPanel.setPreferredSize(new Dimension(gridSize * pixelSize, gridSize * pixelSize));

        // Mouse listeners for drawing/erasing and live prediction
        drawPanel.addMouseListener(new MouseAdapter() {
            public void mousePressed(MouseEvent e) {
                int x = e.getX() / pixelSize;
                int y = e.getY() / pixelSize;
                if (x >= 0 && x < gridSize && y >= 0 && y < gridSize) {
                    isErasing = pixels[y][x] > 0.5f; // If already drawn, erase
                    updatePixel(x, y);
                }
            }
            public void mouseReleased(MouseEvent e) {
                updatePrediction(network);
            }
        });
        drawPanel.addMouseMotionListener(new MouseMotionAdapter() {
            public void mouseDragged(MouseEvent e) {
                int x = e.getX() / pixelSize;
                int y = e.getY() / pixelSize;
                if (x >= 0 && x < gridSize && y >= 0 && y < gridSize) {
                    updatePixel(x, y);
                }
            }
        });

        JButton clearBtn = new JButton("Clear");
        clearBtn.addActionListener(e -> {
            for (int y = 0; y < gridSize; y++)
                for (int x = 0; x < gridSize; x++)
                    pixels[y][x] = 0.0f;
            drawPanel.repaint();
            predictionLabel.setText("Prediction: ");
        });

        JPanel btnPanel = new JPanel();
        btnPanel.add(clearBtn);

        JPanel topPanel = new JPanel(new BorderLayout());
        topPanel.add(predictionLabel, BorderLayout.CENTER);

        add(topPanel, BorderLayout.NORTH);
        add(drawPanel, BorderLayout.CENTER);
        add(btnPanel, BorderLayout.SOUTH);

        clearBtn.doClick();
        pack();
        setLocationRelativeTo(null);
    }

    private void updatePixel(int x, int y) {
        pixels[y][x] = isErasing ? 0.0f : 1.0f;
        drawPanel.repaint();
    }

    private void updatePrediction(Network network) {
        ArrayList<Float> input = getInput();
        ArrayList<Float> output = network.calculateOutput(input);
        int prediction = 0;
        float max = output.get(0);
        for (int i = 1; i < output.size(); i++) {
            if (output.get(i) > max) {
                max = output.get(i);
                prediction = i;
            }
        }
        predictionLabel.setText("Prediction: " + prediction);
    }

    // Convert the 28x28 grid to a 784-length input vector
    private ArrayList<Float> getInput() {
        ArrayList<Float> input = new ArrayList<>(gridSize * gridSize);
        for (int y = 0; y < gridSize; y++)
            for (int x = 0; x < gridSize; x++)
                input.add(pixels[y][x]);
        return input;
    }
}