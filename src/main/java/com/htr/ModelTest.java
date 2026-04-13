package com.htr;

import com.htr.data.IAMDataLoader;
import com.htr.data.IAMSample;
import com.htr.data.ImagePreprocessor;
import com.htr.data.CharsetEncoder;
import com.htr.model.ModelConfig;
import com.htr.model.ModelGraph;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;

/**
 * Quick smoke test for the CNN → LSTM pipeline.
 * Run with --test. Exits 0 on success, 1 on failure.
 *
 * Checks:
 *   1. Model builds without error
 *   2. Forward pass on dummy data produces correct output shape
 *   3. One real training batch completes without error
 */
public class ModelTest {

    private static final Logger log = LoggerFactory.getLogger(ModelTest.class);

    public static void run() {
        int failures = 0;

        // ── Test 1: model builds ──────────────────────────────────────────────
        log.info("[ TEST 1 ] Building model...");
        ComputationGraph model;
        try {
            model = ModelGraph.build();
            model.init();
            log.info("  PASS  Model built — {} parameters", model.numParams());
        } catch (Exception e) {
            log.error("  FAIL  Model build threw: {}", e.getMessage());
            System.exit(1);
            return;
        }

        // ── Test 2: forward pass shape ────────────────────────────────────────
        log.info("[ TEST 2 ] Forward pass on dummy input [2, 1, {}, {}]...",
                ModelConfig.IMG_HEIGHT, ModelConfig.IMG_WIDTH);
        try {
            INDArray dummy = Nd4j.randn(2, 1, ModelConfig.IMG_HEIGHT, ModelConfig.IMG_WIDTH);
            INDArray[] out = model.output(dummy);
            long[] shape = out[0].shape();
            long[] expected = {2, ModelConfig.NUM_CLASSES, ModelGraph.TIME_STEPS};

            log.info("  Output shape: {}", Arrays.toString(shape));
            log.info("  Expected:     {}", Arrays.toString(expected));

            if (Arrays.equals(shape, expected)) {
                log.info("  PASS  Output shape is correct");
            } else {
                log.error("  FAIL  Shape mismatch — CNN→LSTM reshape is still broken");
                failures++;
            }
        } catch (Exception e) {
            log.error("  FAIL  Forward pass threw: {}", e.getMessage());
            failures++;
        }

        // ── Test 3: one real training batch ───────────────────────────────────
        log.info("[ TEST 3 ] One real training batch from dataset...");
        try {
            List<IAMSample> samples = new IAMDataLoader()
                    .loadWords(ModelConfig.DATASET_ROOT);

            if (samples.isEmpty()) {
                log.warn("  SKIP  No samples found at {} — skipping real-data test",
                        ModelConfig.DATASET_ROOT);
            } else {
                List<IAMSample> batch = samples.subList(0, Math.min(4, samples.size()));
                int B = batch.size();

                ImagePreprocessor  preprocessor   = new ImagePreprocessor();
                CharsetEncoder     charsetEncoder = new CharsetEncoder();
                int                T              = ModelGraph.TIME_STEPS;

                INDArray features = Nd4j.zeros(B, 1, ModelConfig.IMG_HEIGHT, ModelConfig.IMG_WIDTH);
                INDArray labels   = Nd4j.zeros(B, ModelConfig.NUM_CLASSES, T);

                for (int b = 0; b < B; b++) {
                    float[][] img = preprocessor.preprocessFromPath(batch.get(b).getImagePath());
                    for (int y = 0; y < ModelConfig.IMG_HEIGHT; y++)
                        for (int x = 0; x < ModelConfig.IMG_WIDTH; x++)
                            features.putScalar(new int[]{b, 0, y, x}, img[y][x]);

                    // CTC packed format: length at [b,0,0], char indices at [b,i+1,0]
                    int[] encoded = charsetEncoder.encode(batch.get(b).getTranscription());
                    int L = Math.min(encoded.length, T - 1);
                    labels.putScalar(new int[]{b, 0, 0}, (float) L);
                    for (int i = 0; i < L; i++) {
                        labels.putScalar(new int[]{b, i + 1, 0}, (float) encoded[i]);
                    }
                }

                MultiDataSet mds = new MultiDataSet(new INDArray[]{features}, new INDArray[]{labels});
                model.fit(mds);
                double loss = model.score();
                log.info("  PASS  Batch loss = {}", String.format("%.4f", loss));
            }
        } catch (Exception e) {
            log.error("  FAIL  Real batch threw: {}", e.getMessage());
            failures++;
        }

        // ── Summary ───────────────────────────────────────────────────────────
        log.info("─────────────────────────────────────");
        if (failures == 0) {
            log.info("All tests passed — CNN→LSTM pipeline is working. Run --train.");
        } else {
            log.error("{} test(s) failed — do not run --train yet.", failures);
        }
        System.exit(failures == 0 ? 0 : 1);
    }
}
