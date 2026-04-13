package com.htr.training;

import com.htr.data.*;
import com.htr.model.ModelConfig;
import com.htr.model.ModelGraph;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.nio.file.*;
import java.util.*;

/**
 * Trains the CNN + LSTM model on the IAM word dataset using DL4J.
 *
 * Loss: multi-class cross-entropy with "stretched" labels.
 * Each word's characters are linearly distributed across TIME_STEPS time
 * positions. The network learns to predict the right character at each
 * position; at inference time CTCDecoder merges duplicates into the final word.
 */
public class ModelTrainer {

    private static final Logger log = LoggerFactory.getLogger(ModelTrainer.class);
    private static final int TIME_STEPS = ModelGraph.TIME_STEPS; // 8

    private final IAMDataLoader     dataLoader     = new IAMDataLoader();
    private final ImagePreprocessor preprocessor   = new ImagePreprocessor();
    private final CharsetEncoder    charsetEncoder = new CharsetEncoder();

    /** All preprocessed images cached as flat float[] to eliminate per-batch disk I/O. */
    private final Map<String, float[]> imageCache = new HashMap<>();

    // ── Entry point ───────────────────────────────────────────────────────────

    public void train() {
        log.info("=== HTR Training started (DL4J) ===");
        log.info("TIME_STEPS={}", TIME_STEPS);

        DatasetSplit split = loadDataset();
        log.info("Dataset: {}", split);
        preloadImages(split);

        ComputationGraph model = loadOrBuild();
        log.info("Model ready  ({} parameters)", model.numParams());

        double bestValLoss = Double.MAX_VALUE;

        for (int epoch = 1; epoch <= ModelConfig.EPOCHS; epoch++) {
            double trainLoss = runEpoch(model, split.getTrainSamples(), true);
            double valLoss   = runEpoch(model, split.getValSamples(),   false);

            log.info("Epoch [{}/{}]  train_loss={}  val_loss={}", epoch, ModelConfig.EPOCHS,
                    String.format("%.4f", trainLoss), String.format("%.4f", valLoss));

            if (valLoss < bestValLoss) {
                bestValLoss = valLoss;
                saveModel(model);
            }
        }

        log.info("=== Training complete.  Best val_loss={} ===", String.format("%.4f", bestValLoss));
    }

    // ── Dataset ───────────────────────────────────────────────────────────────

    private DatasetSplit loadDataset() {
        List<IAMSample> all = dataLoader.loadWords(ModelConfig.DATASET_ROOT);
        if (all.isEmpty()) {
            throw new IllegalStateException(
                    "No samples found in: " + ModelConfig.DATASET_ROOT +
                    "\nRun --validate to diagnose.");
        }
        Path trainFile = Paths.get(ModelConfig.DATASET_ROOT, "trainset.txt");
        if (Files.exists(trainFile)) {
            try {
                return dataLoader.loadSplitFromFiles(all, ModelConfig.DATASET_ROOT);
            } catch (IOException e) {
                log.warn("Could not load official split files, using random split", e);
            }
        }
        return dataLoader.splitRandom(all, 42L);
    }

    // ── Image cache ───────────────────────────────────────────────────────────

    /**
     * Pre-loads every image in the dataset into RAM as flat float[] arrays.
     * This eliminates per-batch disk I/O and is the primary fix for low GPU
     * utilisation — the GPU was starved waiting for the CPU to read files.
     */
    private void preloadImages(DatasetSplit split) {
        List<IAMSample> all = new java.util.ArrayList<>(split.getTrainSamples());
        all.addAll(split.getValSamples());

        log.info("Pre-loading {} images into RAM cache...", all.size());
        int loaded = 0;
        int failed = 0;
        for (IAMSample sample : all) {
            String path = sample.getImagePath();
            if (imageCache.containsKey(path)) continue;
            try {
                float[][] img = preprocessor.preprocessFromPath(path);
                imageCache.put(path, ImagePreprocessor.flatten(img));
                loaded++;
                if (loaded % 5000 == 0) {
                    log.info("  cached {}/{} images", loaded, all.size());
                }
            } catch (Exception e) {
                failed++;
                log.debug("Could not preload {}: {}", path, e.getMessage());
            }
        }
        log.info("Image cache ready: {} loaded, {} skipped/failed", loaded, failed);
    }

    // ── Training loop ─────────────────────────────────────────────────────────

    private double runEpoch(ComputationGraph model, List<IAMSample> samples, boolean isTraining) {
        if (isTraining) Collections.shuffle(samples);

        double totalLoss  = 0.0;
        int    batchCount = 0;

        int totalBatches = (int) Math.ceil((double) samples.size() / ModelConfig.BATCH_SIZE);

        for (int start = 0; start < samples.size(); start += ModelConfig.BATCH_SIZE) {
            int end   = Math.min(start + ModelConfig.BATCH_SIZE, samples.size());
            List<IAMSample> batch = samples.subList(start, end);
            try {
                double loss = runBatch(model, batch, isTraining);
                totalLoss += loss;
                batchCount++;
                if (batchCount % 100 == 0) {
                    log.info("  batch {}/{}  avg_loss={}", batchCount, totalBatches,
                            String.format("%.4f", totalLoss / batchCount));
                }
            } catch (Exception e) {
                log.warn("Skipping batch at index {}: {}", start, e.getMessage());
            }
        }

        return batchCount > 0 ? totalLoss / batchCount : Double.NaN;
    }

    private double runBatch(ComputationGraph model, List<IAMSample> batch, boolean isTraining)
            throws IOException {
        int B = batch.size();

        // features: [B, 1, H, W]  (NCHW — CNN input)
        INDArray features = Nd4j.zeros(B, 1, ModelConfig.IMG_HEIGHT, ModelConfig.IMG_WIDTH);

        // labels: [B, NUM_CLASSES, TIME_STEPS]  (one-hot per time step)
        INDArray labels = Nd4j.zeros(B, ModelConfig.NUM_CLASSES, TIME_STEPS);

        for (int b = 0; b < B; b++) {
            IAMSample sample = batch.get(b);

            // Use cached flat array; fall back to disk if not cached.
            float[] flat = imageCache.get(sample.getImagePath());
            if (flat == null) {
                float[][] img = preprocessor.preprocessFromPath(sample.getImagePath());
                flat = ImagePreprocessor.flatten(img);
            }
            // Assign the whole [H×W] slice in one call — much faster than putScalar loops.
            features.get(NDArrayIndex.point(b), NDArrayIndex.point(0),
                         NDArrayIndex.all(), NDArrayIndex.all())
                    .assign(Nd4j.create(flat, new int[]{ModelConfig.IMG_HEIGHT, ModelConfig.IMG_WIDTH}));

            // CTC label packing: length at [b,0,0], char indices at [b,i+1,0].
            // Max label length = TIME_STEPS - 1 (col 0 reserved for the length value).
            int[] encoded = charsetEncoder.encode(sample.getTranscription());
            int L = Math.min(encoded.length, TIME_STEPS - 1);
            if (L < encoded.length) {
                log.warn("Truncating '{}' ({} chars) to {} for CTC (TIME_STEPS={})",
                        sample.getTranscription(), encoded.length, L, TIME_STEPS);
            }
            labels.putScalar(new int[]{b, 0, 0}, (float) L);
            for (int i = 0; i < L; i++) {
                labels.putScalar(new int[]{b, i + 1, 0}, (float) encoded[i]);
            }
        }

        MultiDataSet mds = new MultiDataSet(
                new INDArray[]{features},
                new INDArray[]{labels});

        if (isTraining) {
            model.fit(mds);
            return model.score();
        } else {
            return model.score(mds);
        }
    }

    // ── Load or build ─────────────────────────────────────────────────────────

    private ComputationGraph loadOrBuild() {
        File saveFile = new File(ModelConfig.MODEL_SAVE_DIR);
        if (saveFile.exists()) {
            try {
                log.info("Resuming from existing model: {}", ModelConfig.MODEL_SAVE_DIR);
                ComputationGraph model = ModelSerializer.restoreComputationGraph(saveFile, true);
                log.info("Loaded model with {} parameters", model.numParams());
                return model;
            } catch (Exception e) {
                log.warn("Could not load existing model ({}). " +
                         "If you changed the loss function, delete {} and retrain from scratch.",
                         e.getMessage(), ModelConfig.MODEL_SAVE_DIR);
            }
        }
        log.info("No saved model found — initialising new model");
        ComputationGraph model = ModelGraph.build();
        model.init();
        return model;
    }

    // ── Save ──────────────────────────────────────────────────────────────────

    private void saveModel(ComputationGraph model) {
        try {
            File saveFile = new File(ModelConfig.MODEL_SAVE_DIR);
            saveFile.getParentFile().mkdirs();
            ModelSerializer.writeModel(model, saveFile, true);
            log.info("Best model saved → {}", ModelConfig.MODEL_SAVE_DIR);
        } catch (IOException e) {
            log.error("Failed to save model", e);
        }
    }
}
