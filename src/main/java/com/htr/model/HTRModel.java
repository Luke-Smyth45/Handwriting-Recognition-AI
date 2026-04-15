package com.htr.model;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

/**
 * Loads a trained DL4J model and runs inference on preprocessed images.
 */
public class HTRModel implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(HTRModel.class);

    private ComputationGraph model;
    private final CTCDecoder decoder = new CTCDecoder(ModelConfig.BEAM_WIDTH);

    // ── Load ──────────────────────────────────────────────────────────────────

    public void load(String modelPath) throws IOException {
        File file = new File(modelPath);
        if (!file.exists()) {
            throw new IllegalArgumentException(
                    "Model file not found: " + modelPath +
                    "\nTrain the model first with --train, or use Load Model to select a .zip file.");
        }
        log.info("Loading model from: {}", modelPath);
        model = ModelSerializer.restoreComputationGraph(file, true);
        log.info("Model loaded successfully ({} parameters)", model.numParams());
    }

    // ── Inference ─────────────────────────────────────────────────────────────

    public String predict(float[][] imageData) {
        if (model == null) {
            throw new IllegalStateException("Model not loaded. Call load() first.");
        }

        int h = imageData.length;
        int w = imageData[0].length;

        // Build input tensor [1, 1, H, W] (NCHW format for CNN)
        INDArray input = Nd4j.zeros(1, 1, h, w);
        for (int row = 0; row < h; row++) {
            for (int col = 0; col < w; col++) {
                input.putScalar(new int[]{0, 0, row, col}, imageData[row][col]);
            }
        }

        // Forward pass → output[0] shape: [1, NUM_CLASSES, TIME_STEPS]
        INDArray[] outputs = model.output(input);
        INDArray logits = outputs[0];

        int timeSteps  = (int) logits.shape()[2];
        int numClasses = (int) logits.shape()[1];

        float[][] logitMatrix = new float[timeSteps][numClasses];
        for (int t = 0; t < timeSteps; t++) {
            for (int c = 0; c < numClasses; c++) {
                logitMatrix[t][c] = logits.getFloat(0, c, t);
            }
        }

        return decoder.beamSearchDecode(logitMatrix);
    }

    @Override
    public void close() {}
}
