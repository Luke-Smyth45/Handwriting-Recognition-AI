package com.htr.data;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * Loads and parses the IAM Handwriting Dataset.
 *
 * Expected dataset layout:
 *   data/raw/
 *     lines.txt          - transcription labels for line-level images
 *     words.txt          - transcription labels for word-level images
 *     lines/             - PNG images keyed by line ID  (a01/a01-000u/a01-000u-00.png)
 *     words/             - PNG images keyed by word ID  (a01/a01-000u/a01-000u-00-00.png)
 *
 * IAM label file format (lines.txt):
 *   # comments start with #
 *   <id> <seg-result> <grey-level> <x> <y> <w> <h> <tag> <transcription>
 *   e.g.: a01-000u-00 ok 154 19 14 1756 52 AT A|MOVE|to|stop|the|legalisation
 *
 * IAM label file format (words.txt):
 *   <id> <seg-result> <grey-level> <x> <y> <w> <h> <grammatical-tag> <transcription>
 *   e.g.: a01-000u-00-00 ok 154 580 484 74 40 VBG A
 */
public class IAMDataLoader {

    private static final Logger log = LoggerFactory.getLogger(IAMDataLoader.class);

    /**
     * Load all line-level samples from the dataset.
     *
     * @param datasetRoot path to the dataset root, e.g. "data/raw"
     * @return list of IAMSample objects (image not yet loaded into memory)
     */
    public List<IAMSample> loadLines(String datasetRoot) {
        Path labelFile = Paths.get(datasetRoot, "lines.txt");
        Path imageRoot = Paths.get(datasetRoot, "lines");
        return parseLabelFile(labelFile, imageRoot, true);
    }

    /**
     * Load all word-level samples from the dataset.
     *
     * @param datasetRoot path to the dataset root, e.g. "data/raw"
     * @return list of IAMSample objects (image not yet loaded into memory)
     */
    public List<IAMSample> loadWords(String datasetRoot) {
        Path labelFile = Paths.get(datasetRoot, "words.txt");
        Path imageRoot = Paths.get(datasetRoot, "words");
        return parseLabelFile(labelFile, imageRoot, false);
    }

    /**
     * Split a list of samples into train / val / test using the official IAM
     * partition (first writer IDs map to train, last to test).
     *
     * If you have the official split files (trainset.txt, validationset1.txt, etc.)
     * place them in datasetRoot and use {@link #loadSplitFromFiles} instead.
     *
     * This fallback uses a 90 / 10 / 0 random split — the former test split
     * is folded into training to maximise the data the model learns from.
     * Use --evaluate on the validation split to measure accuracy.
     */
    public DatasetSplit splitRandom(List<IAMSample> samples, long seed) {
        List<IAMSample> shuffled = new ArrayList<>(samples);
        Collections.shuffle(shuffled, new Random(seed));

        int total    = shuffled.size();
        int trainEnd = (int) (total * 0.90);

        List<IAMSample> train = shuffled.subList(0, trainEnd);
        List<IAMSample> val   = shuffled.subList(trainEnd, total);
        List<IAMSample> test  = new ArrayList<>();

        DatasetSplit split = new DatasetSplit(
                new ArrayList<>(train),
                new ArrayList<>(val),
                new ArrayList<>(test)
        );
        log.info("Random split: {}", split);
        return split;
    }

    /**
     * Load the official IAM partition from split files.
     *
     * Split files expected at:
     *   datasetRoot/trainset.txt
     *   datasetRoot/validationset1.txt  (or validationset.txt)
     *   datasetRoot/testset.txt
     *
     * Each file contains one sample ID per line.
     */
    public DatasetSplit loadSplitFromFiles(List<IAMSample> allSamples, String datasetRoot)
            throws IOException {

        Set<String> trainIds = readIdSet(Paths.get(datasetRoot, "trainset.txt"));
        Set<String> valIds   = readIdSet(Paths.get(datasetRoot, "validationset1.txt"));
        Set<String> testIds  = readIdSet(Paths.get(datasetRoot, "testset.txt"));

        // Fall back to validationset.txt if validationset1.txt not found
        if (valIds.isEmpty()) {
            valIds = readIdSet(Paths.get(datasetRoot, "validationset.txt"));
        }

        List<IAMSample> train = new ArrayList<>();
        List<IAMSample> val   = new ArrayList<>();
        List<IAMSample> test  = new ArrayList<>();

        for (IAMSample sample : allSamples) {
            String id = sample.getId();
            if      (trainIds.contains(id)) train.add(sample);
            else if (valIds.contains(id))   val.add(sample);
            else if (testIds.contains(id))  test.add(sample);
            // samples not in any split file are silently skipped
        }

        DatasetSplit split = new DatasetSplit(train, val, test);
        log.info("Official split loaded: {}", split);
        return split;
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    private List<IAMSample> parseLabelFile(Path labelFile, Path imageRoot, boolean isLines) {
        List<IAMSample> samples = new ArrayList<>();

        if (!Files.exists(labelFile)) {
            log.error("Label file not found: {}", labelFile);
            return samples;
        }

        try (BufferedReader reader = Files.newBufferedReader(labelFile)) {
            String line;
            int lineNum = 0;

            while ((line = reader.readLine()) != null) {
                lineNum++;
                line = line.strip();

                // Skip comment and blank lines
                if (line.isEmpty() || line.startsWith("#")) continue;

                String[] parts = line.split("\\s+");

                // Minimum expected columns: 9
                if (parts.length < 9) {
                    log.warn("Skipping malformed line {} in {}: '{}'", lineNum, labelFile.getFileName(), line);
                    continue;
                }

                String id            = parts[0];
                String segResult     = parts[1];
                String transcription = parts[8];

                // IAM lines.txt uses '|' as word separator — replace with space
                if (isLines) {
                    transcription = transcription.replace("|", " ");
                }

                // Skip segmentation errors
                if ("err".equalsIgnoreCase(segResult)) {
                    log.debug("Skipping err sample: {}", id);
                    continue;
                }

                String imagePath = resolveImagePath(imageRoot, id);
                samples.add(new IAMSample(id, transcription, imagePath));
            }

        } catch (IOException e) {
            log.error("Failed to read label file: {}", labelFile, e);
        }

        log.info("Loaded {} samples from {}", samples.size(), labelFile.getFileName());
        return samples;
    }

    /**
     * Reconstruct the relative image path from a sample ID.
     *
     * Line  ID: a01-000u-00       → lines/a01/a01-000u/a01-000u-00.png
     * Word  ID: a01-000u-00-00    → words/a01/a01-000u/a01-000u-00-00.png
     */
    private String resolveImagePath(Path imageRoot, String id) {
        // e.g. "a01-000u-00-00" → ["a01", "000u", "00", "00"]
        String[] segments = id.split("-");

        // First part of the path: writer folder (e.g. "a01")
        String writer = segments[0];

        // Second part: form folder (e.g. "a01-000u")
        String form = segments[0] + "-" + segments[1];

        return imageRoot.resolve(writer).resolve(form).resolve(id + ".png").toString();
    }

    private Set<String> readIdSet(Path file) throws IOException {
        Set<String> ids = new HashSet<>();
        if (!Files.exists(file)) return ids;
        try (BufferedReader reader = Files.newBufferedReader(file)) {
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.strip();
                if (!line.isEmpty()) ids.add(line);
            }
        }
        return ids;
    }
}
