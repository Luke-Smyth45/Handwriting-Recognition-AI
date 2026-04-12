package com.htr.data;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

/**
 * Preprocesses raw IAM images into normalised float tensors ready for the CNN.
 *
 * Pipeline per image:
 *   1. Load PNG from disk
 *   2. Convert to grayscale
 *   3. Resize to MODEL_HEIGHT, preserving aspect ratio
 *   4. Pad width to MODEL_WIDTH with white (or crop if wider)
 *   5. Normalise pixel values to [0.0, 1.0]  (255 → 1.0)
 *   6. Invert: dark ink becomes high activation (1.0), white background → 0.0
 *
 * Output: float[MODEL_HEIGHT][MODEL_WIDTH]
 */
public class ImagePreprocessor {

    private static final Logger log = LoggerFactory.getLogger(ImagePreprocessor.class);

    /** Target image height fed to the CNN (must match ModelConfig). */
    public static final int MODEL_HEIGHT = 32;

    /** Target image width fed to the CNN (must match ModelConfig). */
    public static final int MODEL_WIDTH  = 128;

    // ── Public API ────────────────────────────────────────────────────────────

    /**
     * Load an image from a file path and preprocess it.
     *
     * @param imagePath absolute or relative path to the PNG/JPEG
     * @return float[MODEL_HEIGHT][MODEL_WIDTH] normalised + inverted pixel values
     * @throws IOException if the file cannot be read
     */
    public float[][] preprocessFromPath(String imagePath) throws IOException {
        BufferedImage raw = ImageIO.read(new File(imagePath));
        if (raw == null) {
            throw new IOException("Could not decode image: " + imagePath);
        }
        return preprocess(raw);
    }

    /**
     * Preprocess an already-loaded BufferedImage.
     *
     * @param image source image (any type / size)
     * @return float[MODEL_HEIGHT][MODEL_WIDTH]
     */
    public float[][] preprocess(BufferedImage image) {
        BufferedImage gray    = toGrayscale(image);
        BufferedImage resized = resizeHeight(gray, MODEL_HEIGHT);
        BufferedImage padded  = padOrCropWidth(resized, MODEL_WIDTH);
        return normaliseAndInvert(padded);
    }

    // ── Pipeline steps ────────────────────────────────────────────────────────

    /** Convert any image to TYPE_BYTE_GRAY. */
    private BufferedImage toGrayscale(BufferedImage src) {
        if (src.getType() == BufferedImage.TYPE_BYTE_GRAY) return src;

        BufferedImage gray = new BufferedImage(
                src.getWidth(), src.getHeight(), BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g = gray.createGraphics();
        g.drawImage(src, 0, 0, null);
        g.dispose();
        return gray;
    }

    /**
     * Resize image to targetHeight while preserving aspect ratio.
     * Width is scaled proportionally.
     */
    private BufferedImage resizeHeight(BufferedImage src, int targetHeight) {
        int srcW = src.getWidth();
        int srcH = src.getHeight();

        if (srcH == targetHeight) return src;

        double scale    = (double) targetHeight / srcH;
        int    newWidth = Math.max(1, (int) (srcW * scale));

        BufferedImage resized = new BufferedImage(newWidth, targetHeight, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g = resized.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING,  RenderingHints.VALUE_ANTIALIAS_ON);
        g.drawImage(src, 0, 0, newWidth, targetHeight, null);
        g.dispose();
        return resized;
    }

    /**
     * Pad image on the right with white to reach targetWidth,
     * or crop on the right if wider than targetWidth.
     */
    private BufferedImage padOrCropWidth(BufferedImage src, int targetWidth) {
        int srcW = src.getWidth();
        if (srcW == targetWidth) return src;

        BufferedImage result = new BufferedImage(targetWidth, src.getHeight(), BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g = result.createGraphics();

        // Fill with white background (grayscale 255)
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, targetWidth, src.getHeight());

        // Draw source (will be clipped if wider than targetWidth)
        g.drawImage(src, 0, 0, null);
        g.dispose();

        if (srcW > targetWidth) {
            log.debug("Image width {} > target {}, cropped", srcW, targetWidth);
        }

        return result;
    }

    /**
     * Convert pixel values from [0, 255] to [0.0, 1.0] and invert
     * so that dark ink → 1.0 and white background → 0.0.
     */
    private float[][] normaliseAndInvert(BufferedImage src) {
        int h = src.getHeight();
        int w = src.getWidth();
        float[][] tensor = new float[h][w];

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                // getRGB returns packed ARGB; for grayscale the R/G/B channels are equal
                int gray = src.getRGB(x, y) & 0xFF;
                // Normalise to [0,1] then invert
                tensor[y][x] = 1.0f - (gray / 255.0f);
            }
        }

        return tensor;
    }

    // ── Utility ───────────────────────────────────────────────────────────────

    /**
     * Flatten a 2-D float tensor to a 1-D array (row-major order).
     * Useful for feeding into TensorFlow as a flat buffer.
     */
    public static float[] flatten(float[][] tensor) {
        int h = tensor.length;
        int w = tensor[0].length;
        float[] flat = new float[h * w];
        for (int y = 0; y < h; y++) {
            System.arraycopy(tensor[y], 0, flat, y * w, w);
        }
        return flat;
    }

    /**
     * Expand to shape [1, H, W, 1] as required by the CNN input (batch=1, channels=1).
     */
    public static float[][][][] expandDims(float[][] tensor) {
        int h = tensor.length;
        int w = tensor[0].length;
        float[][][][] result = new float[1][h][w][1];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                result[0][y][x][0] = tensor[y][x];
            }
        }
        return result;
    }
}
