package com.htr.data;

import org.junit.jupiter.api.Test;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;

import static org.junit.jupiter.api.Assertions.*;

class ImagePreprocessorTest {

    private final ImagePreprocessor preprocessor = new ImagePreprocessor();

    @Test
    void preprocess_outputHasCorrectDimensions() {
        // Create a small dummy grayscale image
        BufferedImage img = new BufferedImage(200, 50, BufferedImage.TYPE_BYTE_GRAY);
        float[][] tensor = preprocessor.preprocess(img);

        assertEquals(ImagePreprocessor.MODEL_HEIGHT, tensor.length,       "Height mismatch");
        assertEquals(ImagePreprocessor.MODEL_WIDTH,  tensor[0].length,    "Width mismatch");
    }

    @Test
    void preprocess_valuesInZeroOneRange() {
        BufferedImage img = makeColorImage(100, 40);
        float[][] tensor = preprocessor.preprocess(img);

        for (float[] row : tensor) {
            for (float v : row) {
                assertTrue(v >= 0f && v <= 1f, "Pixel value out of [0,1]: " + v);
            }
        }
    }

    @Test
    void preprocess_whiteBackgroundBecomesZero() {
        // All-white image → after invert, should be 0.0
        BufferedImage white = new BufferedImage(64, 32, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g = white.createGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, 64, 32);
        g.dispose();

        float[][] tensor = preprocessor.preprocess(white);
        for (float[] row : tensor) {
            for (float v : row) {
                assertEquals(0f, v, 0.01f, "White pixel should map to ~0.0 after inversion");
            }
        }
    }

    @Test
    void flatten_correctLength() {
        float[][] t = new float[ImagePreprocessor.MODEL_HEIGHT][ImagePreprocessor.MODEL_WIDTH];
        float[] flat = ImagePreprocessor.flatten(t);
        assertEquals(ImagePreprocessor.MODEL_HEIGHT * ImagePreprocessor.MODEL_WIDTH, flat.length);
    }

    @Test
    void expandDims_correctShape() {
        float[][] t = new float[ImagePreprocessor.MODEL_HEIGHT][ImagePreprocessor.MODEL_WIDTH];
        float[][][][] expanded = ImagePreprocessor.expandDims(t);
        assertEquals(1,                              expanded.length);
        assertEquals(ImagePreprocessor.MODEL_HEIGHT, expanded[0].length);
        assertEquals(ImagePreprocessor.MODEL_WIDTH,  expanded[0][0].length);
        assertEquals(1,                              expanded[0][0][0].length);
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private BufferedImage makeColorImage(int w, int h) {
        BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(Color.LIGHT_GRAY);
        g.fillRect(0, 0, w, h);
        g.setColor(Color.BLACK);
        g.drawString("Hello", 10, 20);
        g.dispose();
        return img;
    }
}
