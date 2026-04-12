package com.htr.model;

import com.htr.data.CharsetEncoder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * Decodes the raw CTC output (log-probability matrix) into a text string.
 *
 * Two strategies:
 *   1. Greedy best-path  — fast O(T * C), slightly less accurate
 *   2. Beam search       — O(T * beamWidth * C), more accurate (default for inference)
 *
 * Input matrix shape: float[timeSteps][numClasses]
 * The last class index (ModelConfig.BLANK_INDEX) is the CTC blank token.
 */
public class CTCDecoder {

    private static final Logger log = LoggerFactory.getLogger(CTCDecoder.class);

    private final int            beamWidth;
    private final CharsetEncoder charset;

    public CTCDecoder(int beamWidth) {
        this.beamWidth = beamWidth;
        this.charset   = new CharsetEncoder();
    }

    // ── Greedy best-path decoding ─────────────────────────────────────────────

    /**
     * At each time step pick the highest-probability class, then collapse
     * repeated characters and remove blanks.
     *
     * @param logits float[timeSteps][numClasses]  (raw logits or log-probs)
     * @return decoded text string
     */
    public String greedyDecode(float[][] logits) {
        int timeSteps  = logits.length;
        int blankIndex = ModelConfig.BLANK_INDEX;

        int[] bestPath = new int[timeSteps];
        for (int t = 0; t < timeSteps; t++) {
            bestPath[t] = argmax(logits[t]);
        }

        // CTC collapse: remove blanks and consecutive duplicate labels
        List<Integer> collapsed = new ArrayList<>();
        int prev = -1;
        for (int label : bestPath) {
            if (label != blankIndex && label != prev) {
                collapsed.add(label);
            }
            prev = label;
        }

        return charset.decode(collapsed.stream().mapToInt(i -> i).toArray());
    }

    // ── Beam search decoding ──────────────────────────────────────────────────

    /**
     * Beam-search decoding — returns the top-1 hypothesis.
     *
     * @param logits float[timeSteps][numClasses]
     * @return decoded text string
     */
    public String beamSearchDecode(float[][] logits) {
        List<ScoredHypothesis> beams = beamSearchDecodeAll(logits);
        return beams.isEmpty() ? "" : beams.get(0).text();
    }

    /**
     * Beam-search decoding — returns all beams ranked by log-probability.
     *
     * Uses the standard prefix-beam-search algorithm:
     *   - Maintains beamWidth active prefixes.
     *   - Each prefix stores the total probability of ending with a non-blank
     *     and the total probability of ending with a blank separately.
     *
     * @param logits float[timeSteps][numClasses]
     * @return list of hypotheses, best first
     */
    public List<ScoredHypothesis> beamSearchDecodeAll(float[][] logits) {
        int timeSteps  = logits.length;
        int numClasses = logits[0].length;
        int blankIdx   = ModelConfig.BLANK_INDEX;

        // Convert logits to log-probabilities via log-softmax
        double[][] logProbs = logSoftmax(logits);

        // Beam: Map from prefix (int[]) → [logProbBlank, logProbNonBlank]
        // Use a string key for the prefix for easy map lookups
        Map<String, double[]> beams = new HashMap<>();
        beams.put("", new double[]{0.0, Double.NEGATIVE_INFINITY}); // start: p(blank)=1

        for (int t = 0; t < timeSteps; t++) {
            Map<String, double[]> newBeams = new HashMap<>();

            // Only extend the top beamWidth prefixes
            List<Map.Entry<String, double[]>> sorted = new ArrayList<>(beams.entrySet());
            sorted.sort((a, b) -> Double.compare(totalLogProb(b.getValue()), totalLogProb(a.getValue())));
            List<Map.Entry<String, double[]>> top = sorted.subList(0, Math.min(beamWidth, sorted.size()));

            for (Map.Entry<String, double[]> entry : top) {
                String prefix  = entry.getKey();
                double[] probs = entry.getValue(); // [logPb, logPnb]

                for (int c = 0; c < numClasses; c++) {
                    double lp = logProbs[t][c];

                    if (c == blankIdx) {
                        // Extend with blank → same prefix, update blank prob
                        String key = prefix;
                        double[] np = newBeams.getOrDefault(key, new double[]{Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY});
                        np[0] = logSumExp(np[0], totalLogProb(probs) + lp);
                        newBeams.put(key, np);
                    } else {
                        String label       = String.valueOf(charset.decode(new int[]{c}));
                        String newPrefix   = prefix + label;
                        double[] np        = newBeams.getOrDefault(newPrefix, new double[]{Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY});

                        // Check if the new label equals the last character of the prefix
                        char lastChar = prefix.isEmpty() ? 0 : prefix.charAt(prefix.length() - 1);
                        String newLabel = charset.decode(new int[]{c});
                        boolean isRepeat = !newLabel.isEmpty() && newLabel.charAt(0) == lastChar;

                        if (isRepeat) {
                            // Only non-blank ending can extend with repeat (requires blank in between)
                            np[1] = logSumExp(np[1], probs[0] + lp); // only blank-ending extends
                        } else {
                            np[1] = logSumExp(np[1], totalLogProb(probs) + lp);
                        }
                        newBeams.put(newPrefix, np);
                    }
                }
            }
            beams = newBeams;
        }

        // Collect and sort final beams
        List<ScoredHypothesis> results = new ArrayList<>();
        for (Map.Entry<String, double[]> entry : beams.entrySet()) {
            results.add(new ScoredHypothesis(entry.getKey(), totalLogProb(entry.getValue())));
        }
        results.sort((a, b) -> Double.compare(b.logProb(), a.logProb()));
        return results;
    }

    // ── Math helpers ──────────────────────────────────────────────────────────

    private int argmax(float[] arr) {
        int best = 0;
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > arr[best]) best = i;
        }
        return best;
    }

    /** Numerically stable log-softmax over each timestep. */
    private double[][] logSoftmax(float[][] logits) {
        double[][] out = new double[logits.length][logits[0].length];
        for (int t = 0; t < logits.length; t++) {
            double max = Double.NEGATIVE_INFINITY;
            for (float v : logits[t]) if (v > max) max = v;

            double sumExp = 0.0;
            for (float v : logits[t]) sumExp += Math.exp(v - max);
            double logSumE = max + Math.log(sumExp);

            for (int c = 0; c < logits[t].length; c++) {
                out[t][c] = logits[t][c] - logSumE;
            }
        }
        return out;
    }

    /** log(exp(a) + exp(b)) computed in a numerically stable way. */
    private double logSumExp(double a, double b) {
        if (a == Double.NEGATIVE_INFINITY) return b;
        if (b == Double.NEGATIVE_INFINITY) return a;
        double max = Math.max(a, b);
        return max + Math.log(Math.exp(a - max) + Math.exp(b - max));
    }

    /** Total log-probability of a prefix = logSumExp(blank, non-blank). */
    private double totalLogProb(double[] probs) {
        return logSumExp(probs[0], probs[1]);
    }

    // ── Types ─────────────────────────────────────────────────────────────────

    /** A decoded text hypothesis paired with its log-probability score. */
    public record ScoredHypothesis(String text, double logProb) {}
}
