package com.htr.model;

import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.common.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

/**
 * True CTC (Connectionist Temporal Classification) loss for DL4J.
 *
 * Implements ILossFunction so it can be dropped into RnnOutputLayer.
 * The output layer must use Activation.IDENTITY — this class applies
 * softmax internally and returns the combined CTC+Softmax gradient
 * (grad[t][c] = softmax[t][c] - gamma[t][c]).
 *
 * DL4J's reshape3dTo2d applies permute([0,2,1]) then F-order reshape,
 * mapping labels3d[b, c, t] → labels2d[b + B*t, c].
 *
 * Label packing in labels3d[B, NUM_CLASSES, T] (set by ModelTrainer):
 *   labels3d[b, 0,   0] = L           → labels2d[b, 0]
 *   labels3d[b, i+1, 0] = charIdx[i]  → labels2d[b, i+1]
 *   all other cells = 0.0
 *
 * So in labels2d we read the label for sample b from row b (NOT b*T).
 * Similarly, preOutput2d[b + B*t, c] = preOutput3d[b, c, t], so
 * gradients for sample b at time t go to row b + B*t.
 *
 * Serialization: ILossFunction carries @JsonTypeInfo(use=Id.CLASS), so Jackson
 * writes the fully-qualified class name and reconstructs via zero-arg constructor.
 */
public class CTCLossFunction implements ILossFunction {

    private static final long serialVersionUID = 1L;
    private static final Logger log = LoggerFactory.getLogger(CTCLossFunction.class);
    private static final double NEG_INF = Double.NEGATIVE_INFINITY;

    // ── ILossFunction ─────────────────────────────────────────────────────────

    @Override
    public double computeScore(INDArray labels, INDArray preOutput,
                               IActivation activationFn, INDArray mask, boolean average) {
        int totalRows = (int) preOutput.shape()[0];
        int C         = (int) preOutput.shape()[1];
        int T         = ModelGraph.TIME_STEPS;
        int B         = totalRows / T;

        double[][] lsm = logSoftmax(preOutput, totalRows, C);

        double totalLoss = 0.0;
        int    counted   = 0;
        for (int b = 0; b < B; b++) {
            int[] labelSeq = extractLabel(labels, b);
            if (labelSeq.length == 0 || labelSeq.length > T) continue;

            double[][] logProbs = sliceLogProbs(lsm, b, B, T, C);
            double     logP     = ctcForwardOnly(logProbs, labelSeq);

            if (!Double.isInfinite(logP)) {
                totalLoss -= logP;
                counted++;
            }
        }
        return average && counted > 0 ? totalLoss / counted : totalLoss;
    }

    @Override
    public INDArray computeScoreArray(INDArray labels, INDArray preOutput,
                                      IActivation activationFn, INDArray mask) {
        int totalRows = (int) preOutput.shape()[0];
        int C         = (int) preOutput.shape()[1];
        int T         = ModelGraph.TIME_STEPS;
        int B         = totalRows / T;

        double[][] lsm   = logSoftmax(preOutput, totalRows, C);
        INDArray   scores = Nd4j.zeros(totalRows, 1);

        for (int b = 0; b < B; b++) {
            int[] labelSeq = extractLabel(labels, b);
            if (labelSeq.length == 0 || labelSeq.length > T) continue;

            double[][] logProbs = sliceLogProbs(lsm, b, B, T, C);
            double     logP     = ctcForwardOnly(logProbs, labelSeq);
            double     loss     = Double.isInfinite(logP) ? 0.0 : -logP;

            // Distribute per-sample loss evenly across T rows so DL4J's
            // time-axis reduction yields the correct per-sample total.
            float perRow = (float) (loss / T);
            for (int t = 0; t < T; t++) {
                scores.putScalar(b + B * t, 0, perRow);
            }
        }
        return scores;
    }

    @Override
    public INDArray computeGradient(INDArray labels, INDArray preOutput,
                                    IActivation activationFn, INDArray mask) {
        int totalRows = (int) preOutput.shape()[0];
        int C         = (int) preOutput.shape()[1];
        int T         = ModelGraph.TIME_STEPS;
        int B         = totalRows / T;

        double[][] lsm  = logSoftmax(preOutput, totalRows, C);
        INDArray   grad = Nd4j.zeros(totalRows, C);

        for (int b = 0; b < B; b++) {
            int[] labelSeq = extractLabel(labels, b);
            if (labelSeq.length == 0 || labelSeq.length > T) continue;

            double[][] logProbs = sliceLogProbs(lsm, b, B, T, C);
            double[][] gamma    = new double[T][C];

            double logP = ctcForwardBackward(logProbs, labelSeq, gamma);
            if (Double.isInfinite(logP)) {
                log.debug("CTC: degenerate alignment for sample {} (label len {}), skipping gradient",
                        b, labelSeq.length);
                continue;
            }

            // Combined CTC + Softmax gradient: softmax[t][c] - gamma[t][c]
            // Gradient for sample b at time t goes to row b + B*t.
            for (int t = 0; t < T; t++) {
                for (int c = 0; c < C; c++) {
                    double softmax = Math.exp(logProbs[t][c]);
                    grad.putScalar(new int[]{b + B * t, c}, (float) (softmax - gamma[t][c]));
                }
            }
        }

        // Activation.IDENTITY backprop is a no-op — return grad directly.
        return grad;
    }

    @Override
    public Pair<Double, INDArray> computeGradientAndScore(INDArray labels, INDArray preOutput,
            IActivation activationFn, INDArray mask, boolean average) {
        int totalRows = (int) preOutput.shape()[0];
        int C         = (int) preOutput.shape()[1];
        int T         = ModelGraph.TIME_STEPS;
        int B         = totalRows / T;

        double[][] lsm  = logSoftmax(preOutput, totalRows, C);
        INDArray   grad = Nd4j.zeros(totalRows, C);

        double totalLoss = 0.0;
        int    counted   = 0;

        for (int b = 0; b < B; b++) {
            int[] labelSeq = extractLabel(labels, b);
            if (labelSeq.length == 0 || labelSeq.length > T) continue;

            double[][] logProbs = sliceLogProbs(lsm, b, B, T, C);
            double[][] gamma    = new double[T][C];

            double logP = ctcForwardBackward(logProbs, labelSeq, gamma);
            if (Double.isInfinite(logP)) {
                log.debug("CTC: degenerate alignment for sample {} (label len {}), skipping",
                        b, labelSeq.length);
                continue;
            }

            totalLoss -= logP;
            counted++;

            for (int t = 0; t < T; t++) {
                for (int c = 0; c < C; c++) {
                    double softmax = Math.exp(logProbs[t][c]);
                    grad.putScalar(new int[]{b + B * t, c}, (float) (softmax - gamma[t][c]));
                }
            }
        }

        double score = average && counted > 0 ? totalLoss / counted : totalLoss;
        return Pair.of(score, grad);
    }

    @Override
    public String name() {
        return "CTC";
    }

    // ── CTC forward only (for score computation) ──────────────────────────────

    /**
     * Runs the CTC forward algorithm and returns log P(label | logProbs).
     *
     * @param logProbs  [T][C] log-softmax probabilities
     * @param labelSeq  character index sequence, length L
     * @return log probability of the label sequence; NEGATIVE_INFINITY if
     *         the label cannot be aligned (L > T)
     */
    private double ctcForwardOnly(double[][] logProbs, int[] labelSeq) {
        int T     = logProbs.length;
        int L     = labelSeq.length;
        int blank = ModelConfig.BLANK_INDEX;
        int S     = 2 * L + 1;
        int[] lp  = buildLPrime(labelSeq, blank, S);

        double[][] logAlpha = new double[T][S];
        for (double[] row : logAlpha) Arrays.fill(row, NEG_INF);

        logAlpha[0][0] = logProbs[0][blank];
        if (S > 1) logAlpha[0][1] = logProbs[0][labelSeq[0]];

        for (int t = 1; t < T; t++) {
            for (int s = 0; s < S; s++) {
                double a = logAlpha[t-1][s];
                if (s > 0) a = logSumExp(a, logAlpha[t-1][s-1]);
                if (s > 1 && lp[s] != lp[s-2]) a = logSumExp(a, logAlpha[t-1][s-2]);
                logAlpha[t][s] = a + logProbs[t][lp[s]];
            }
        }

        return logSumExp(logAlpha[T-1][S-1], S >= 2 ? logAlpha[T-1][S-2] : NEG_INF);
    }

    // ── CTC forward + backward (for gradient computation) ────────────────────

    /**
     * Runs the full CTC forward-backward algorithm.
     *
     * @param logProbs  [T][C] log-softmax probabilities
     * @param labelSeq  character index sequence, length L
     * @param gamma     output array [T][C] filled with CTC posteriors
     * @return log P(label | logProbs); NEGATIVE_INFINITY on degenerate input
     */
    private double ctcForwardBackward(double[][] logProbs, int[] labelSeq, double[][] gamma) {
        int T     = logProbs.length;
        int C     = logProbs[0].length;
        int L     = labelSeq.length;
        int blank = ModelConfig.BLANK_INDEX;
        int S     = 2 * L + 1;
        int[] lp  = buildLPrime(labelSeq, blank, S);

        // ── Forward ──────────────────────────────────────────────────────────
        double[][] logAlpha = new double[T][S];
        for (double[] row : logAlpha) Arrays.fill(row, NEG_INF);

        logAlpha[0][0] = logProbs[0][blank];
        if (S > 1) logAlpha[0][1] = logProbs[0][labelSeq[0]];

        for (int t = 1; t < T; t++) {
            for (int s = 0; s < S; s++) {
                double a = logAlpha[t-1][s];
                if (s > 0) a = logSumExp(a, logAlpha[t-1][s-1]);
                if (s > 1 && lp[s] != lp[s-2]) a = logSumExp(a, logAlpha[t-1][s-2]);
                logAlpha[t][s] = a + logProbs[t][lp[s]];
            }
        }

        double logP = logSumExp(logAlpha[T-1][S-1], S >= 2 ? logAlpha[T-1][S-2] : NEG_INF);
        if (Double.isInfinite(logP)) return logP;

        // ── Backward ─────────────────────────────────────────────────────────
        double[][] logBeta = new double[T][S];
        for (double[] row : logBeta) Arrays.fill(row, NEG_INF);

        logBeta[T-1][S-1] = 0.0;
        if (S >= 2) logBeta[T-1][S-2] = 0.0;

        for (int t = T - 2; t >= 0; t--) {
            for (int s = 0; s < S; s++) {
                double b = logBeta[t+1][s] + logProbs[t+1][lp[s]];
                if (s < S-1)
                    b = logSumExp(b, logBeta[t+1][s+1] + logProbs[t+1][lp[s+1]]);
                if (s < S-2 && lp[s] != lp[s+2])
                    b = logSumExp(b, logBeta[t+1][s+2] + logProbs[t+1][lp[s+2]]);
                logBeta[t][s] = b;
            }
        }

        // ── Posteriors gamma[t][c] ────────────────────────────────────────────
        for (int t = 0; t < T; t++) {
            double[] logAB = new double[C];
            Arrays.fill(logAB, NEG_INF);

            for (int s = 0; s < S; s++) {
                int c = lp[s];
                logAB[c] = logSumExp(logAB[c], logAlpha[t][s] + logBeta[t][s]);
            }

            for (int c = 0; c < C; c++) {
                gamma[t][c] = Double.isInfinite(logAB[c]) ? 0.0 : Math.exp(logAB[c] - logP);
            }
        }

        return logP;
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /**
     * Build the CTC expanded label l' = [blank, c0, blank, c1, ..., cL-1, blank].
     * Even positions = blank; odd positions = label character.
     */
    private static int[] buildLPrime(int[] labelSeq, int blank, int S) {
        int[] lp = new int[S];
        for (int s = 0; s < S; s++) {
            lp[s] = (s % 2 == 0) ? blank : labelSeq[(s - 1) / 2];
        }
        return lp;
    }

    /**
     * Row-wise log-softmax applied to a [rows, C] INDArray.
     * Returns a Java double[rows][C] for fast inner-loop access.
     */
    private static double[][] logSoftmax(INDArray preOutput, int rows, int C) {
        double[][] out = new double[rows][C];
        for (int r = 0; r < rows; r++) {
            double max = Double.NEGATIVE_INFINITY;
            for (int c = 0; c < C; c++) {
                double v = preOutput.getDouble(r, c);
                if (v > max) max = v;
            }
            double sumExp = 0.0;
            for (int c = 0; c < C; c++) sumExp += Math.exp(preOutput.getDouble(r, c) - max);
            double logSumE = max + Math.log(sumExp);
            for (int c = 0; c < C; c++) {
                out[r][c] = preOutput.getDouble(r, c) - logSumE;
            }
        }
        return out;
    }

    /**
     * Extract the [T][C] log-prob slice for batch item b.
     *
     * DL4J reshape: preOutput2d[b + B*t, c] = preOutput3d[b, c, t].
     * So time step t for sample b is at row b + B*t in the 2D array.
     */
    private static double[][] sliceLogProbs(double[][] lsm, int b, int B, int T, int C) {
        double[][] slice = new double[T][C];
        for (int t = 0; t < T; t++) {
            slice[t] = Arrays.copyOf(lsm[b + B * t], C);
        }
        return slice;
    }

    /**
     * Extract the label sequence for batch item b from the packed labels2d tensor.
     *
     * DL4J reshape: labels3d[b, c, 0] → labels2d[b, c].
     * Packing (set by ModelTrainer):
     *   labels2d[b, 0]   = L           (label length)
     *   labels2d[b, i+1] = charIdx[i]  (for i = 0..L-1)
     *
     * Returns empty array if L is 0 or invalid.
     */
    private static int[] extractLabel(INDArray labels2d, int b) {
        int L = (int) labels2d.getFloat(b, 0);
        if (L <= 0 || L >= (int) labels2d.shape()[1]) return new int[0];
        int[] seq = new int[L];
        for (int i = 0; i < L; i++) {
            seq[i] = (int) labels2d.getFloat(b, i + 1);
        }
        return seq;
    }

    /** Numerically stable log(exp(a) + exp(b)). */
    private static double logSumExp(double a, double b) {
        if (a == NEG_INF) return b;
        if (b == NEG_INF) return a;
        double m = Math.max(a, b);
        return m + Math.log(Math.exp(a - m) + Math.exp(b - m));
    }
}
