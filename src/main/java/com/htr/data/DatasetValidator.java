package com.htr.data;

import com.htr.model.ModelConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * Validates that the IAM dataset (Kaggle word-level download) is correctly
 * placed and structured before training or inference begins.
 *
 * Expected layout after extracting the Kaggle zip into data/raw/:
 *
 *   data/raw/
 *     words.txt                      ← transcription labels
 *     words/                         ← image directory
 *       a01/
 *         a01-000u/
 *           a01-000u-00-00.png
 *           a01-000u-00-01.png
 *           ...
 *       a02/
 *         ...
 *
 * Run via:  java -jar handwriting-recognition.jar --validate
 */
public class DatasetValidator {

    private static final Logger log = LoggerFactory.getLogger(DatasetValidator.class);

    private final String datasetRoot;
    private final List<String> errors   = new ArrayList<>();
    private final List<String> warnings = new ArrayList<>();

    public DatasetValidator(String datasetRoot) {
        this.datasetRoot = datasetRoot;
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /**
     * Run all checks. Prints a full report and returns true if valid.
     */
    public boolean validate() {
        log.info("Validating dataset at: {}", datasetRoot);

        checkRootExists();
        checkLabelFile();
        checkImageDirectory();
        checkSampleImages();

        printReport();
        return errors.isEmpty();
    }

    // ── Checks ────────────────────────────────────────────────────────────────

    private void checkRootExists() {
        Path root = Paths.get(datasetRoot);
        if (!Files.exists(root)) {
            errors.add("Dataset root directory not found: " + datasetRoot +
                    "\n  → Create it and extract the Kaggle zip contents inside.");
        } else if (!Files.isDirectory(root)) {
            errors.add(datasetRoot + " exists but is a file, not a directory.");
        }
    }

    private void checkLabelFile() {
        Path labelFile = Paths.get(datasetRoot, "words.txt");
        if (!Files.exists(labelFile)) {
            errors.add("Missing label file: " + labelFile +
                    "\n  → The Kaggle zip should contain 'words.txt'. Make sure you extracted it into data/raw/");
            return;
        }

        // Count valid and error lines
        int total = 0, valid = 0, errSeg = 0, comment = 0;
        try (BufferedReader reader = Files.newBufferedReader(labelFile)) {
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.strip();
                if (line.isEmpty())          continue;
                if (line.startsWith("#"))    { comment++; continue; }
                total++;
                String[] parts = line.split("\\s+");
                if (parts.length >= 9) {
                    if ("err".equalsIgnoreCase(parts[1])) errSeg++;
                    else valid++;
                }
            }
        } catch (IOException e) {
            errors.add("Could not read words.txt: " + e.getMessage());
            return;
        }

        log.info("words.txt: {} valid samples, {} segmentation errors, {} comment lines",
                valid, errSeg, comment);

        if (valid == 0) {
            errors.add("words.txt contains no valid samples.");
        } else if (valid < 1000) {
            warnings.add("words.txt has only " + valid + " valid samples — expected ~115,000 for the full IAM dataset.");
        }
    }

    private void checkImageDirectory() {
        Path imgDir = Paths.get(datasetRoot, "words");
        if (!Files.exists(imgDir)) {
            errors.add("Missing image directory: " + imgDir +
                    "\n  → The Kaggle zip should contain a 'words/' folder. Extract it into data/raw/");
            return;
        }

        // Count top-level writer subdirectories (a01, a02, ...)
        try {
            long writerDirs = Files.list(imgDir)
                    .filter(Files::isDirectory)
                    .count();
            if (writerDirs == 0) {
                errors.add("Image directory is empty: " + imgDir);
            } else {
                log.info("words/ directory: {} writer subdirectories found", writerDirs);
                if (writerDirs < 10) {
                    warnings.add("Only " + writerDirs + " writer directories found — expected ~100 for the full dataset.");
                }
            }
        } catch (IOException e) {
            errors.add("Could not read image directory: " + e.getMessage());
        }
    }

    private void checkSampleImages() {
        // Read the first few valid entries from words.txt and verify images exist
        Path labelFile = Paths.get(datasetRoot, "words.txt");
        Path imgRoot   = Paths.get(datasetRoot, "words");
        if (!Files.exists(labelFile) || !Files.exists(imgRoot)) return;

        int checked = 0, missing = 0;
        try (BufferedReader reader = Files.newBufferedReader(labelFile)) {
            String line;
            while ((line = reader.readLine()) != null && checked < 20) {
                line = line.strip();
                if (line.isEmpty() || line.startsWith("#")) continue;
                String[] parts = line.split("\\s+");
                if (parts.length < 9 || "err".equalsIgnoreCase(parts[1])) continue;

                String id        = parts[0];
                String imagePath = resolveImagePath(imgRoot, id);
                checked++;
                if (!Files.exists(Paths.get(imagePath))) missing++;
            }
        } catch (IOException e) {
            warnings.add("Could not verify sample images: " + e.getMessage());
            return;
        }

        if (checked == 0) return;

        if (missing == checked) {
            errors.add("None of the first " + checked + " image files were found." +
                    "\n  → Check that the 'words/' folder structure matches: words/a01/a01-000u/a01-000u-00-00.png");
        } else if (missing > 0) {
            warnings.add(missing + "/" + checked + " spot-checked images were missing.");
        } else {
            log.info("Image spot-check: {}/{} images verified OK", checked, checked);
        }
    }

    // ── Report ────────────────────────────────────────────────────────────────

    private void printReport() {
        System.out.println();
        System.out.println("══════════════════════════════════════════════");
        System.out.println("  Dataset Validation Report");
        System.out.println("  Root: " + datasetRoot);
        System.out.println("══════════════════════════════════════════════");

        if (errors.isEmpty() && warnings.isEmpty()) {
            System.out.println("  ✔  Dataset looks good! Ready to train.");
        }

        if (!errors.isEmpty()) {
            System.out.println("\n  ERRORS (" + errors.size() + "):");
            for (String e : errors) {
                System.out.println("  [ERROR] " + e);
            }
        }

        if (!warnings.isEmpty()) {
            System.out.println("\n  WARNINGS (" + warnings.size() + "):");
            for (String w : warnings) {
                System.out.println("  [WARN]  " + w);
            }
        }

        if (!errors.isEmpty()) {
            System.out.println();
            System.out.println("  How to fix:");
            System.out.println("  1. Download from https://www.kaggle.com/datasets/nibinv23/iam-handwriting-word-database");
            System.out.println("  2. Extract the zip");
            System.out.println("  3. Place contents so you have:");
            System.out.println("       " + datasetRoot + "/words.txt");
            System.out.println("       " + datasetRoot + "/words/a01/...");
        }

        System.out.println("══════════════════════════════════════════════");
        System.out.println();
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private String resolveImagePath(Path imgRoot, String id) {
        String[] seg = id.split("-");
        String writer = seg[0];
        String form   = seg[0] + "-" + seg[1];
        return imgRoot.resolve(writer).resolve(form).resolve(id + ".png").toString();
    }
}
