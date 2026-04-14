package com.htr;

import com.htr.data.DatasetValidator;
import com.htr.model.ModelConfig;
import com.htr.ui.HandwritingRecognitionUI;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.*;

/**
 * Entry point for the Handwriting Recognition AI application.
 *
 * Usage:
 *   --train     Validate dataset then train the model
 *   --validate  Check dataset structure and exit
 *   --ui        Launch the graphical interface (default)
 */
public class Main {

    private static final Logger log = LoggerFactory.getLogger(Main.class);

    public static void main(String[] args) {
        String mode = args.length > 0 ? args[0] : "--ui";

        switch (mode) {
            case "--train"    -> runTraining();
            case "--validate" -> runValidation();
            case "--test"     -> ModelTest.run();
            case "--evaluate" -> Evaluator.run();
            case "--ui"       -> runUI();
            default -> {
                log.error("Unknown mode: '{}'. Use --train, --validate, --test, --evaluate, or --ui", mode);
                printUsage();
                System.exit(1);
            }
        }
    }

    // ── Modes ─────────────────────────────────────────────────────────────────

    private static void runValidation() {
        DatasetValidator validator = new DatasetValidator(ModelConfig.DATASET_ROOT);
        boolean ok = validator.validate();
        System.exit(ok ? 0 : 1);
    }

    private static void runTraining() {
        // Always validate first so training doesn't fail silently mid-way
        log.info("Checking dataset before training...");
        DatasetValidator validator = new DatasetValidator(ModelConfig.DATASET_ROOT);
        if (!validator.validate()) {
            log.error("Dataset validation failed. Fix the errors above, then re-run --train.");
            System.exit(1);
        }

        log.info("Starting training...");
        com.htr.training.ModelTrainer trainer = new com.htr.training.ModelTrainer();
        trainer.train();
    }

    private static void runUI() {
        log.info("Launching UI...");
        SwingUtilities.invokeLater(() -> {
            HandwritingRecognitionUI ui = new HandwritingRecognitionUI();
            ui.setVisible(true);
        });
    }

    // ── Help ──────────────────────────────────────────────────────────────────

    private static void printUsage() {
        System.out.println();
        System.out.println("Usage: java -jar handwriting-recognition.jar [mode]");
        System.out.println();
        System.out.println("  --ui        Launch the graphical interface (default)");
        System.out.println("  --train     Validate dataset then train the model");
        System.out.println("  --validate  Check dataset structure only and exit");
        System.out.println();
        System.out.println("Dataset should be placed at: " + ModelConfig.DATASET_ROOT);
        System.out.println("Download from: https://www.kaggle.com/datasets/nibinv23/iam-handwriting-word-database");
        System.out.println();
    }
}
