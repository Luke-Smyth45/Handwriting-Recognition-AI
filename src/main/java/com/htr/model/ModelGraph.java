package com.htr.model;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Builds the shared CNN + BiLSTM model architecture used by both
 * ModelTrainer (training) and HTRModel (inference).
 *
 * Architecture:
 *   Input  [B, 1, 32, 128]  (NCHW)
 *   → 5 CNN blocks           → [B, 256, 2, 8]
 *   → CnnToRnn (auto)        → [B, 512, 8]  (features=512, time=8)
 *   → BiLSTM × 2             → [B, 512, 8]
 *   → Softmax output         → [B, NUM_CLASSES, 8]
 */
public class ModelGraph {

    /** Number of time steps after CNN downsampling (IMG_WIDTH / 16). */
    public static final int TIME_STEPS = ModelConfig.IMG_WIDTH / 16; // 8

    private ModelGraph() {}

    /**
     * Build and return an uninitialised ComputationGraph ready for
     * {@code model.init()} or {@code ModelSerializer.restoreComputationGraph()}.
     */
    public static ComputationGraph build() {
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(42)
                .updater(new Adam(ModelConfig.LEARNING_RATE))
                .weightInit(WeightInit.XAVIER)
                .graphBuilder()
                .addInputs("input")

                // ── CNN Block 1: [B,1,32,128] → pool → [B,32,16,64] ──────────
                .addLayer("conv1", new ConvolutionLayer.Builder(3, 3)
                        .nOut(32).padding(1, 1).activation(Activation.RELU).build(), "input")
                .addLayer("pool1", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2).stride(2, 2).build(), "conv1")

                // ── CNN Block 2: → pool → [B,64,8,32] ────────────────────────
                .addLayer("conv2", new ConvolutionLayer.Builder(3, 3)
                        .nOut(64).padding(1, 1).activation(Activation.RELU).build(), "pool1")
                .addLayer("pool2", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2).stride(2, 2).build(), "conv2")

                // ── CNN Block 3: → pool → [B,128,4,16] ───────────────────────
                .addLayer("conv3", new ConvolutionLayer.Builder(3, 3)
                        .nOut(128).padding(1, 1).activation(Activation.RELU).build(), "pool2")
                .addLayer("pool3", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2).stride(2, 2).build(), "conv3")

                // ── CNN Block 4: no pool → [B,128,4,16] ──────────────────────
                .addLayer("conv4", new ConvolutionLayer.Builder(3, 3)
                        .nOut(128).padding(1, 1).activation(Activation.RELU).build(), "pool3")

                // ── CNN Block 5: → pool → [B,256,2,8] ────────────────────────
                .addLayer("conv5", new ConvolutionLayer.Builder(3, 3)
                        .nOut(256).padding(1, 1).activation(Activation.RELU).build(), "conv4")
                .addLayer("pool5", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2).stride(2, 2).build(), "conv5")

                // ── BiLSTM (DL4J auto-inserts CnnToRnnPreProcessor here) ──────
                // pool5 [B,256,2,8] → [B,512,8] after preprocessor
                .addLayer("bilstm1", new Bidirectional(Bidirectional.Mode.CONCAT,
                        new LSTM.Builder()
                                .nOut(ModelConfig.RNN_UNITS)
                                .activation(Activation.TANH)
                                .build()), "pool5")

                .addLayer("bilstm2", new Bidirectional(Bidirectional.Mode.CONCAT,
                        new LSTM.Builder()
                                .nOut(ModelConfig.RNN_UNITS)
                                .activation(Activation.TANH)
                                .build()), "bilstm1")

                // ── Output: softmax over NUM_CLASSES at each time step ─────────
                .addLayer("output", new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nOut(ModelConfig.NUM_CLASSES)
                        .activation(Activation.SOFTMAX)
                        .build(), "bilstm2")

                .setOutputs("output")
                // Tell DL4J the input type so it can infer nIn and insert preprocessors
                .setInputTypes(InputType.convolutional(
                        ModelConfig.IMG_HEIGHT, ModelConfig.IMG_WIDTH, ModelConfig.IMG_CHANNELS))
                .build();

        return new ComputationGraph(conf);
    }
}
