package com.htr.model;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.ReshapeVertex;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import com.htr.model.CTCLossFunction;

/**
 * Builds the CNN + LSTM model as a ComputationGraph.
 *
 * Architecture:
 *   Input  [B, 1, 32, 128]    (NCHW)
 *   → 5 CNN blocks             → [B, 256, 2, 16]
 *   → ReshapeVertex (C-order)  → [B, 512, 16]  channels×height collapsed into features,
 *                                               width preserved as time steps
 *   → LSTM × 2                 → [B, 256, 16]
 *   → CTC output               → [B, NUM_CLASSES, 16]
 *
 * ReshapeVertex is a primitive Nd4j reshape — no semantic preprocessing logic.
 * A C-order reshape of [B, C, H, W] → [B, C*H, W] correctly maps each CNN
 * column (width position) to one time step, equivalent to CnnToRnnPreProcessor
 * but without that class's DL4J 1.0.0-M2.1 bug that collapses time steps to 1.
 *
 * setInputTypes() is intentionally omitted: it runs after the builder and
 * silently overwrites manually configured vertices / preprocessors. All nIn
 * values are set explicitly so DL4J needs no inference.
 */
public class ModelGraph {

    /** Width columns kept as time steps: 3× width-halving pools (128 / 8 = 16). */
    public static final int TIME_STEPS = ModelConfig.IMG_WIDTH / 8; // 16

    // CNN output after pool5: [B, 256, 2, 16]
    private static final int CNN_OUT_HEIGHT   = ModelConfig.IMG_HEIGHT / 16; // 2
    private static final int CNN_OUT_CHANNELS = ModelConfig.CNN_FILTERS[4];  // 256

    /** Feature size per time step after ReshapeVertex: channels × height. */
    public static final int RNN_INPUT_SIZE = CNN_OUT_CHANNELS * CNN_OUT_HEIGHT; // 512

    private ModelGraph() {}

    public static ComputationGraph build() {
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(42)
                .updater(new Adam(ModelConfig.LEARNING_RATE))
                .weightInit(WeightInit.XAVIER)
                // Clip gradients so LSTM weights can't explode.
                // ClipElementWiseAbsoluteValue caps each gradient value at ±1.0,
                // which is the standard fix for RNN gradient explosion.
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(1.0)
                .graphBuilder()
                .addInputs("input")

                // ── CNN Block 1: [B,1,32,128] → bn → pool → [B,32,16,64] ─────
                .addLayer("conv1", new ConvolutionLayer.Builder(3, 3)
                        .nIn(ModelConfig.IMG_CHANNELS).nOut(32)
                        .padding(1, 1).activation(Activation.RELU).build(), "input")
                .addLayer("bn1", new BatchNormalization.Builder().nIn(32).nOut(32).build(), "conv1")
                .addLayer("pool1", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2).stride(2, 2).build(), "bn1")

                // ── CNN Block 2: → bn → pool → [B,64,8,32] ───────────────────
                .addLayer("conv2", new ConvolutionLayer.Builder(3, 3)
                        .nIn(32).nOut(64)
                        .padding(1, 1).activation(Activation.RELU).build(), "pool1")
                .addLayer("bn2", new BatchNormalization.Builder().nIn(64).nOut(64).build(), "conv2")
                .addLayer("pool2", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2).stride(2, 2).build(), "bn2")

                // ── CNN Block 3: → bn → pool → [B,128,4,16] ──────────────────
                .addLayer("conv3", new ConvolutionLayer.Builder(3, 3)
                        .nIn(64).nOut(128)
                        .padding(1, 1).activation(Activation.RELU).build(), "pool2")
                .addLayer("bn3", new BatchNormalization.Builder().nIn(128).nOut(128).build(), "conv3")
                .addLayer("pool3", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2).stride(2, 2).build(), "bn3")

                // ── CNN Block 4: → bn → no pool → [B,128,4,16] ───────────────
                .addLayer("conv4", new ConvolutionLayer.Builder(3, 3)
                        .nIn(128).nOut(128)
                        .padding(1, 1).activation(Activation.RELU).build(), "pool3")
                .addLayer("bn4", new BatchNormalization.Builder().nIn(128).nOut(128).build(), "conv4")

                // ── CNN Block 5: → bn → pool (height only) → [B,256,2,16] ───
                .addLayer("conv5", new ConvolutionLayer.Builder(3, 3)
                        .nIn(128).nOut(256)
                        .padding(1, 1).activation(Activation.RELU).build(), "bn4")
                .addLayer("bn5", new BatchNormalization.Builder().nIn(256).nOut(256).build(), "conv5")
                .addLayer("pool5", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 1).stride(2, 1).build(), "bn5")

                // ── Reshape: [B,256,2,16] → [B,512,16] ───────────────────────
                // C-order reshape: element [b,c,h,w] maps to [b, c*H+h, w].
                // Each width column w becomes one time step; channels and height
                // are flattened into the feature dimension. This is numerically
                // equivalent to CnnToRnnPreProcessor without its DL4J 1.0.0-M2.1
                // bug that always collapses the time dimension to 1.
                .addVertex("reshape", new ReshapeVertex(-1, RNN_INPUT_SIZE, TIME_STEPS), "pool5")

                // ── LSTM 1: [B,512,16] → [B,256,16] ─────────────────────────
                .addLayer("lstm1", new LSTM.Builder()
                        .nIn(RNN_INPUT_SIZE)
                        .nOut(ModelConfig.RNN_UNITS)
                        .activation(Activation.TANH)
                        .dropOut(0.5)
                        .build(), "reshape")

                // ── LSTM 2: [B,256,16] → [B,NUM_CLASSES,16] ─────────────────
                // nOut = NUM_CLASSES so the RnnOutputLayer input feature size
                // matches the label feature size. DL4J 1.0.0-M2.1 validates
                // input.size(1) == label.size(1) before applying output weights,
                // so the final LSTM must project directly to NUM_CLASSES.
                .addLayer("lstm2", new LSTM.Builder()
                        .nIn(ModelConfig.RNN_UNITS)
                        .nOut(ModelConfig.NUM_CLASSES)
                        .activation(Activation.TANH)
                        .build(), "lstm1")

                // ── Output: [B,NUM_CLASSES,8] → CTC loss (softmax inside CTCLossFunction) ──
                .addLayer("output", new RnnOutputLayer.Builder(new CTCLossFunction())
                        .nIn(ModelConfig.NUM_CLASSES)
                        .nOut(ModelConfig.NUM_CLASSES)
                        .activation(Activation.IDENTITY)  // softmax applied inside CTCLossFunction
                        .build(), "lstm2")

                .setOutputs("output")
                .build();

        return new ComputationGraph(conf);
    }
}
