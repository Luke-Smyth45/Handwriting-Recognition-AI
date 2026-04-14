package com.htr;

import com.htr.data.*;
import com.htr.model.HTRModel;
import com.htr.model.ModelConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

/**
 * Evaluates the saved model on the held-out test split.
 *
 * Metrics reported:
 *   - Word Error Rate  (WER): fraction of words predicted incorrectly (exact match)
 *   - Character Error Rate (CER): Levenshtein edit distance / ground-truth length
 */
public class Evaluator {

    private static final Logger log = LoggerFactory.getLogger(Evaluator.class);

    public static void run() {
        log.info("=== HTR Evaluation ===");

        // Load dataset and take the test split
        IAMDataLoader loader = new IAMDataLoader();
        List<IAMSample> all = loader.loadWords(ModelConfig.DATASET_ROOT);
        if (all.isEmpty()) {
            log.error("No samples found in: {}", ModelConfig.DATASET_ROOT);
            System.exit(1);
        }

        DatasetSplit split = loader.splitRandom(all, 42L);
        List<IAMSample> testSamples = split.getTestSamples();
        log.info("Test samples: {}", testSamples.size());

        // Load model
        HTRModel model = new HTRModel();
        try {
            model.load(ModelConfig.MODEL_SAVE_DIR);
        } catch (Exception e) {
            log.error("Could not load model: {}", e.getMessage());
            System.exit(1);
        }

        ImagePreprocessor preprocessor = new ImagePreprocessor();

        int totalWords  = 0;
        int wrongWords  = 0;
        long totalEdits = 0;
        long totalChars = 0;
        int  errors     = 0;

        for (int i = 0; i < testSamples.size(); i++) {
            IAMSample sample = testSamples.get(i);
            String ground = sample.getTranscription().toLowerCase();

            try {
                float[][] img = preprocessor.preprocessFromPath(sample.getImagePath());
                String pred   = model.predict(img).toLowerCase();

                totalWords++;
                if (!pred.equals(ground)) wrongWords++;

                int edit = levenshtein(pred, ground);
                totalEdits += edit;
                totalChars += ground.length();

            } catch (Exception e) {
                errors++;
                log.debug("Skipping {}: {}", sample.getImagePath(), e.getMessage());
            }

            if ((i + 1) % 500 == 0) {
                log.info("  evaluated {}/{}", i + 1, testSamples.size());
            }
        }

        double wer = totalWords > 0 ? 100.0 * wrongWords / totalWords : 0;
        double cer = totalChars > 0 ? 100.0 * totalEdits / totalChars : 0;

        log.info("─────────────────────────────────────");
        log.info("Results on {} test samples ({} skipped):", totalWords, errors);
        log.info("  Word Error Rate  (WER): {}", String.format("%.2f%%", wer));
        log.info("  Char Error Rate  (CER): {}", String.format("%.2f%%", cer));
        log.info("  Word Accuracy:          {}", String.format("%.2f%%", 100.0 - wer));
        log.info("─────────────────────────────────────");
    }

    /** Standard Levenshtein edit distance between two strings. */
    private static int levenshtein(String a, String b) {
        int m = a.length(), n = b.length();
        int[] prev = new int[n + 1];
        int[] curr = new int[n + 1];
        for (int j = 0; j <= n; j++) prev[j] = j;
        for (int i = 1; i <= m; i++) {
            curr[0] = i;
            for (int j = 1; j <= n; j++) {
                if (a.charAt(i - 1) == b.charAt(j - 1)) {
                    curr[j] = prev[j - 1];
                } else {
                    curr[j] = 1 + Math.min(prev[j - 1], Math.min(prev[j], curr[j - 1]));
                }
            }
            int[] tmp = prev; prev = curr; curr = tmp;
        }
        return prev[n];
    }
}
