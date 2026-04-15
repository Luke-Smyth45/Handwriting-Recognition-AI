package com.htr.data;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

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

    // ── Augmentation ──────────────────────────────────────────────────────────

    private static final ThreadLocal<Random> THREAD_RANDOM = ThreadLocal.withInitial(Random::new);

    /**
     * Apply random augmentation to a preprocessed float[H][W] image.
     * Only called during training — never during inference.
     *
     * Transforms applied:
     *   - Random rotation        ±7°
     *   - Random width stretch   [0.85, 1.15]
     *   - Random brightness      [0.75, 1.25]
     *   - Gaussian noise         σ = 0.03
     */
    public static float[][] augment(float[][] img) {
        ThreadLocalRandom rng = ThreadLocalRandom.current();

        // Random rotation ±7°
        img = rotate(img, Math.toRadians(rng.nextDouble(-7.0, 7.0)));

        // Random width stretch 0.85–1.15
        img = stretchWidth(img, rng.nextDouble(0.85, 1.15));

        // Random brightness jitter ±25%
        float brightScale = (float) rng.nextDouble(0.75, 1.25);
        int H = img.length, W = img[0].length;
        for (int r = 0; r < H; r++)
            for (int c = 0; c < W; c++)
                img[r][c] = Math.min(1f, img[r][c] * brightScale);

        // Gaussian noise σ=0.03
        Random rand = THREAD_RANDOM.get();
        for (int r = 0; r < H; r++)
            for (int c = 0; c < W; c++)
                img[r][c] = Math.min(1f, Math.max(0f,
                        img[r][c] + (float) (rand.nextGaussian() * 0.03)));

        return img;
    }

    private static float[][] stretchWidth(float[][] src, double factor) {
        int H = src.length, W = src[0].length;
        int newW = Math.max(1, (int) Math.round(W * factor));
        float[][] stretched = new float[H][newW];
        for (int r = 0; r < H; r++)
            for (int c = 0; c < newW; c++) {
                double srcX = c / factor;
                int x0 = (int) Math.floor(srcX);
                double wx = srcX - x0;
                stretched[r][c] = (float) ((1 - wx) * pixel(src, r, x0,     H, W)
                                         +      wx  * pixel(src, r, x0 + 1, H, W));
            }
        // Centre the stretched image in the original W, padding/cropping as needed
        float[][] result = new float[H][W];
        if (newW >= W) {
            int offset = (newW - W) / 2;
            for (int r = 0; r < H; r++)
                for (int c = 0; c < W; c++)
                    result[r][c] = stretched[r][Math.min(offset + c, newW - 1)];
        } else {
            int offset = (W - newW) / 2;
            for (int r = 0; r < H; r++)
                for (int c = 0; c < newW; c++)
                    result[r][offset + c] = stretched[r][c];
        }
        return result;
    }

    private static float[][] rotate(float[][] src, double angle) {
        int H = src.length, W = src[0].length;
        float cx = (W - 1) / 2f, cy = (H - 1) / 2f;
        // Inverse rotation: map destination pixel back to source
        double cos = Math.cos(-angle), sin = Math.sin(-angle);
        float[][] dst = new float[H][W];
        for (int r = 0; r < H; r++) {
            for (int c = 0; c < W; c++) {
                double dx = c - cx, dy = r - cy;
                double srcX = cos * dx - sin * dy + cx;
                double srcY = sin * dx + cos * dy + cy;
                dst[r][c] = bilinear(src, srcX, srcY, H, W);
            }
        }
        return dst;
    }

    private static float bilinear(float[][] img, double x, double y, int H, int W) {
        int x0 = (int) Math.floor(x), y0 = (int) Math.floor(y);
        double wx = x - x0, wy = y - y0;
        return (float) (
            (1 - wy) * ((1 - wx) * pixel(img, y0,     x0,     H, W)
                      +      wx  * pixel(img, y0,     x0 + 1, H, W)) +
                 wy  * ((1 - wx) * pixel(img, y0 + 1, x0,     H, W)
                      +      wx  * pixel(img, y0 + 1, x0 + 1, H, W))
        );
    }

    private static float pixel(float[][] img, int r, int c, int H, int W) {
        if (r < 0 || r >= H || c < 0 || c >= W) return 0f;
        return img[r][c];
    }

    // ── Utility ───────────────────────────────────────────────────────────────

    /**
     * Reshape a flat float[] (row-major) back to float[H][W].
     */
    public static float[][] unflatten(float[] flat, int H, int W) {
        float[][] img = new float[H][W];
        for (int r = 0; r < H; r++)
            System.arraycopy(flat, r * W, img[r], 0, W);
        return img;
    }

    /**
     * Flatten a 2-D float tensor to a 1-D array (row-major order).
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
