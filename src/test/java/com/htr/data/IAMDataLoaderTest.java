package com.htr.data;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class IAMDataLoaderTest {

    @TempDir
    Path tempDir;

    @Test
    void loaderInstantiates() {
        assertDoesNotThrow(IAMDataLoader::new);
    }

    @Test
    void loadLines_returnsEmptyListWhenLabelFileMissing() {
        IAMDataLoader loader = new IAMDataLoader();
        List<IAMSample> samples = loader.loadLines(tempDir.toString());
        assertTrue(samples.isEmpty(), "Should return empty list when lines.txt is absent");
    }

    @Test
    void loadLines_parsesValidLabelFile() throws IOException {
        // Write a minimal lines.txt
        Path labelFile = tempDir.resolve("lines.txt");
        Files.writeString(labelFile,
                "# IAM Database line list\n" +
                "a01-000u-00 ok 154 19 14 1756 52 AT A|MOVE|to|stop\n" +
                "a01-000u-01 ok 156 19 70 1756 48 AT The|move|is\n" +
                "a01-000u-02 err 157 19 119 1756 56 AT bad|line\n"
        );
        // Create the lines directory tree expected by resolveImagePath
        Files.createDirectories(tempDir.resolve("lines/a01/a01-000u"));

        IAMDataLoader loader = new IAMDataLoader();
        List<IAMSample> samples = loader.loadLines(tempDir.toString());

        // 'err' sample should be skipped
        assertEquals(2, samples.size());
        assertEquals("a01-000u-00", samples.get(0).getId());
        assertEquals("A MOVE to stop", samples.get(0).getTranscription());
        assertEquals("a01-000u-01", samples.get(1).getId());
        assertEquals("The move is",   samples.get(1).getTranscription());
    }

    @Test
    void splitRandom_respectsRatios() {
        IAMDataLoader loader = new IAMDataLoader();
        List<IAMSample> samples = new java.util.ArrayList<>();
        for (int i = 0; i < 1000; i++) {
            samples.add(new IAMSample("id-" + i, "text", "path"));
        }

        DatasetSplit split = loader.splitRandom(samples, 42L);
        assertEquals(1000, split.totalSize());
        assertEquals(900,  split.getTrainSamples().size());
        assertEquals(100,  split.getValSamples().size());
        assertEquals(0,    split.getTestSamples().size());
    }
}
