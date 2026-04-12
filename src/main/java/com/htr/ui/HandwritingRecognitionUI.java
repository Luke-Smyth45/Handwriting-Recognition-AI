package com.htr.ui;

import com.htr.data.ImagePreprocessor;
import com.htr.model.HTRModel;
import com.htr.model.ModelConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import javax.swing.*;
import javax.swing.filechooser.FileNameExtensionFilter;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

/**
 * Main application window.
 *
 * Layout:
 * ┌──────────────────────────────────────────────────┐
 * │  Toolbar: [Load Model] [Load Image] [Clear]      │
 * ├──────────────────────────────────────────────────┤
 * │  DrawingPanel  (free-hand canvas, top section)   │
 * ├──────────────────────────────────────────────────┤
 * │  Preview of loaded image (if any)                │
 * ├──────────────────────────────────────────────────┤
 * │  [Recognise] button                              │
 * ├──────────────────────────────────────────────────┤
 * │  ResultPanel  (prediction output)                │
 * └──────────────────────────────────────────────────┘
 */
public class HandwritingRecognitionUI extends JFrame {

    private static final Logger log = LoggerFactory.getLogger(HandwritingRecognitionUI.class);

    // ── UI components ─────────────────────────────────────────────────────────
    private final DrawingPanel    drawingPanel  = new DrawingPanel();
    private final ResultPanel     resultPanel   = new ResultPanel();
    private final JLabel          imagePreview  = new JLabel("No image loaded", SwingConstants.CENTER);
    private final JButton         btnRecognise  = new JButton("Recognise");
    private final JButton         btnClear      = new JButton("Clear");
    private final JButton         btnLoadImage  = new JButton("Load Image...");
    private final JButton         btnLoadModel  = new JButton("Load Model...");
    private final JLabel          modelStatus   = new JLabel("No model loaded");

    // ── State ─────────────────────────────────────────────────────────────────
    private HTRModel        model       = null;
    private BufferedImage   loadedImage = null;
    private final ImagePreprocessor preprocessor = new ImagePreprocessor();

    // ── Constructor ───────────────────────────────────────────────────────────

    public HandwritingRecognitionUI() {
        super("Handwriting Recognition AI");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(760, 600);
        setMinimumSize(new Dimension(640, 500));
        setLocationRelativeTo(null);

        buildUI();
        wireListeners();

        // Try to auto-load the default model zip if it exists
        if (Files.exists(Paths.get(ModelConfig.MODEL_SAVE_DIR))) {
            loadModelFromPath(ModelConfig.MODEL_SAVE_DIR);
        } else {
            resultPanel.showStatus("Load a trained model to begin");
        }
    }

    // ── UI construction ───────────────────────────────────────────────────────

    private void buildUI() {
        setLayout(new BorderLayout(6, 6));
        getRootPane().setBorder(BorderFactory.createEmptyBorder(8, 8, 8, 8));

        // ── Toolbar ───────────────────────────────────────────────────────────
        JPanel toolbar = new JPanel(new FlowLayout(FlowLayout.LEFT, 6, 0));
        toolbar.add(btnLoadModel);
        toolbar.add(btnLoadImage);
        toolbar.add(btnClear);
        toolbar.add(Box.createHorizontalStrut(20));
        modelStatus.setForeground(Color.GRAY);
        modelStatus.setFont(modelStatus.getFont().deriveFont(Font.ITALIC));
        toolbar.add(modelStatus);
        add(toolbar, BorderLayout.NORTH);

        // ── Centre: drawing canvas + optional image preview ───────────────────
        JPanel centre = new JPanel(new BorderLayout(4, 4));

        JPanel drawSection = new JPanel(new BorderLayout(4, 4));
        drawSection.setBorder(BorderFactory.createTitledBorder("Draw Here (or load an image)"));
        drawSection.add(drawingPanel, BorderLayout.CENTER);
        centre.add(drawSection, BorderLayout.CENTER);

        // Image preview (shown only when an image is loaded)
        imagePreview.setPreferredSize(new Dimension(200, 80));
        imagePreview.setBorder(BorderFactory.createTitledBorder("Loaded Image"));
        imagePreview.setVisible(false);
        centre.add(imagePreview, BorderLayout.EAST);

        add(centre, BorderLayout.CENTER);

        // ── Bottom: recognise button + result ─────────────────────────────────
        JPanel bottom = new JPanel(new BorderLayout(4, 4));

        btnRecognise.setFont(btnRecognise.getFont().deriveFont(Font.BOLD, 14f));
        btnRecognise.setPreferredSize(new Dimension(0, 40));
        bottom.add(btnRecognise, BorderLayout.NORTH);
        bottom.add(resultPanel,  BorderLayout.CENTER);

        add(bottom, BorderLayout.SOUTH);
    }

    // ── Event wiring ──────────────────────────────────────────────────────────

    private void wireListeners() {
        btnLoadModel.addActionListener(e -> onLoadModel());
        btnLoadImage.addActionListener(e -> onLoadImage());
        btnClear.addActionListener(e -> onClear());
        btnRecognise.addActionListener(e -> onRecognise());
    }

    // ── Action handlers ───────────────────────────────────────────────────────

    private void onLoadModel() {
        JFileChooser fc = new JFileChooser("models");
        fc.setFileFilter(new FileNameExtensionFilter("DL4J Model (*.zip)", "zip"));
        fc.setDialogTitle("Select trained model (.zip)");
        if (fc.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
            loadModelFromPath(fc.getSelectedFile().getAbsolutePath());
        }
    }

    private void loadModelFromPath(String path) {
        try {
            if (model != null) model.close();
            model = new HTRModel();
            model.load(path);
            modelStatus.setText("Model: " + Paths.get(path).getFileName());
            modelStatus.setForeground(new Color(0, 140, 0));
            resultPanel.showStatus("Model loaded — draw or load an image, then click Recognise");
            log.info("Model loaded from {}", path);
        } catch (Exception ex) {
            modelStatus.setText("Model load failed");
            modelStatus.setForeground(Color.RED);
            resultPanel.showStatus("Failed to load model: " + ex.getMessage());
            log.error("Failed to load model from {}", path, ex);
            model = null;
        }
    }

    private void onLoadImage() {
        JFileChooser fc = new JFileChooser();
        fc.setFileFilter(new FileNameExtensionFilter("Images (PNG, JPEG, BMP)", "png", "jpg", "jpeg", "bmp"));
        if (fc.showOpenDialog(this) != JFileChooser.APPROVE_OPTION) return;

        File file = fc.getSelectedFile();
        try {
            loadedImage = ImageIO.read(file);
            if (loadedImage == null) throw new IOException("Could not decode image");

            // Show thumbnail in the preview panel
            Image thumb = loadedImage.getScaledInstance(
                    imagePreview.getPreferredSize().width - 8, -1, Image.SCALE_SMOOTH);
            imagePreview.setIcon(new ImageIcon(thumb));
            imagePreview.setText(null);
            imagePreview.setVisible(true);

            // Clear the drawing canvas since we now have an image
            drawingPanel.clear();
            resultPanel.showStatus("Image loaded — click Recognise");
            log.info("Image loaded: {}", file.getAbsolutePath());
        } catch (IOException ex) {
            resultPanel.showStatus("Could not load image: " + ex.getMessage());
            log.error("Image load failed", ex);
        }
        revalidate();
        repaint();
    }

    private void onClear() {
        drawingPanel.clear();
        loadedImage = null;
        imagePreview.setIcon(null);
        imagePreview.setText("No image loaded");
        imagePreview.setVisible(false);
        resultPanel.clear();
        revalidate();
        repaint();
    }

    private void onRecognise() {
        if (model == null) {
            JOptionPane.showMessageDialog(this,
                    "Please load a trained model first.\n\nUse the 'Load Model...' button to select a saved model directory.",
                    "No model loaded", JOptionPane.WARNING_MESSAGE);
            return;
        }

        resultPanel.showProcessing();
        btnRecognise.setEnabled(false);

        // Run inference on a background thread so the UI stays responsive
        SwingWorker<String, Void> worker = new SwingWorker<>() {
            @Override
            protected String doInBackground() throws Exception {
                BufferedImage source = loadedImage != null
                        ? loadedImage
                        : drawingPanel.getCanvasImage();

                float[][] tensor = preprocessor.preprocess(source);
                return model.predict(tensor);
            }

            @Override
            protected void done() {
                btnRecognise.setEnabled(true);
                try {
                    String text = get();
                    resultPanel.showResult(text);
                    log.info("Prediction: '{}'", text);
                } catch (Exception ex) {
                    resultPanel.showStatus("Error: " + ex.getCause().getMessage());
                    log.error("Inference failed", ex);
                }
            }
        };

        worker.execute();
    }
}
