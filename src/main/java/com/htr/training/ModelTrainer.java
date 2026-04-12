package com.htr.training;

import com.htr.data.*;
import com.htr.model.ModelConfig;
import com.htr.model.ModelGraph;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.nio.file.*;
import java.util.*;

/**
 * Trains the CNN + BiLSTM model on the IAM word dataset using DL4J.
 *
 * Loss: multi-class cross-entropy with "stretched" labels.
 * Each word's characters are linearly distributed across TIME_STEPS time
 * positions. The network learns to predict the right character at each
 * position; at inference time CTCDecoder merges duplicates into the final word.
 */
public class ModelTrainer {

    private static final Logger log = LoggerFactory.getLogger(ModelTrainer.class);
    private static final int TIME_STEPS = ModelGraph.TIME_STEPS; // IMG_WIDTH / 16 = 8

    private final IAMDataLoader     dataLoader     = new IAMDataLoader();
    private final ImagePreprocessor preprocessor   = new ImagePreprocessor();
    private final CharsetEncoder    charsetEncoder = new CharsetEncoder();

    // ── Entry point ───────────────────────────────────────────────────────────

    public void train() {
        log.info("=== HTR Training started (DL4J) ===");
        log.info("TIME_STEPS={}", TIME_STEPS);

        DatasetSplit split = loadDataset();
        log.info("Dataset: {}", split);

        ComputationGraph model = ModelGraph.build();
        model.init();
        log.info("Model initialised  ({} parameters)", model.numParams());

        double bestValLoss = Double.MAX_VALUE;

        for (int epoch = 1; epoch <= ModelConfig.EPOCHS; epoch++) {
            double trainLoss = runEpoch(model, split.getTrainSamples(), true);
            double valLoss   = runEpoch(model, split.getValSamples(),   false);

            log.info("Epoch [{}/{}]  train_loss={:.4f}  val_loss={:.4f}",
                    epoch, ModelConfig.EPOCHS, trainLoss, valLoss);

            if (valLoss < bestValLoss) {
                bestValLoss = valLoss;
                saveModel(model);
            }
        }

        log.info("=== Training complete.  Best val_loss={:.4f} ===", bestValLoss);
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
                    log.info("  batch {}/{}  avg_loss={:.4f}",
                            batchCount, totalBatches, totalLoss / batchCount);
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

        // labels: [B, NUM_CLASSES, TIME_STEPS]  (one-hot per time step — RnnOutputLayer)
        INDArray labels = Nd4j.zeros(B, ModelConfig.NUM_CLASSES, TIME_STEPS);

        for (int b = 0; b < B; b++) {
            IAMSample sample = batch.get(b);

            // Load and preprocess image → float[32][128]
            float[][] img = preprocessor.preprocessFromPath(sample.getImagePath());
            for (int y = 0; y < ModelConfig.IMG_HEIGHT; y++) {
                for (int x = 0; x < ModelConfig.IMG_WIDTH; x++) {
                    features.putScalar(new int[]{b, 0, y, x}, img[y][x]);
                }
            }

            // Encode transcription and stretch across TIME_STEPS
            int[] encoded = charsetEncoder.encode(sample.getTranscription());
            for (int t = 0; t < TIME_STEPS; t++) {
                int charIdx  = encoded.length > 0
                        ? Math.min((int) ((long) t * encoded.length / TIME_STEPS), encoded.length - 1)
                        : 0;
                int classIdx = encoded.length > 0 ? encoded[charIdx] : ModelConfig.BLANK_INDEX;
                labels.putScalar(new int[]{b, classIdx, t}, 1.0f);
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
