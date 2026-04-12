package com.htr.model;

import com.htr.data.ImagePreprocessor;

/**
 * Central configuration for the HTR model architecture and training.
 */
public class ModelConfig {

    // ── Input dimensions ──────────────────────────────────────────────────────
    public static final int IMG_HEIGHT   = ImagePreprocessor.MODEL_HEIGHT; // 32
    public static final int IMG_WIDTH    = ImagePreprocessor.MODEL_WIDTH;  // 128
    public static final int IMG_CHANNELS = 1;                              // grayscale

    // ── CNN ───────────────────────────────────────────────────────────────────
    // 5 blocks: filters, 3×3 kernels, pool sizes (0 = no pool)
    // Pool schedule {2,2,2,0,2} reduces H: 32→2, W: 128→8
    public static final int[] CNN_FILTERS      = {32, 64, 128, 128, 256};
    public static final int[] CNN_KERNEL_SIZES = {3, 3, 3, 3, 3};
    public static final int[] POOL_SIZES       = {2, 2, 2, 0, 2};

    // ── RNN ───────────────────────────────────────────────────────────────────
    public static final int RNN_UNITS  = 256; // units per direction in BiLSTM
    public static final int RNN_LAYERS = 2;

    // ── Output ────────────────────────────────────────────────────────────────
    /** All printable ASCII characters + blank token for CTC */
    public static final String CHARSET =
            " !\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    public static final int NUM_CLASSES = CHARSET.length() + 1; // +1 for CTC blank
    public static final int BLANK_INDEX = NUM_CLASSES - 1;

    // ── Training ──────────────────────────────────────────────────────────────
    public static final int    BATCH_SIZE    = 32;
    public static final int    EPOCHS        = 50;
    public static final double LEARNING_RATE = 1e-4;

    // ── CTC Decoding ──────────────────────────────────────────────────────────
    public static final int BEAM_WIDTH = 10;

    // ── Paths ─────────────────────────────────────────────────────────────────
    public static final String DATASET_ROOT   = "data/raw/archive/iam_words";
    public static final String MODEL_SAVE_DIR = "models/htr_model.zip";

    private ModelConfig() {}
}
