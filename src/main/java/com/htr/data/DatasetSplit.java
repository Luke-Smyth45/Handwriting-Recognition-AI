package com.htr.data;

import java.util.List;

/**
 * Holds the train / validation / test splits of the IAM dataset.
 *
 * IAM's official partition:
 *   Train : ~6,161 lines
 *   Val   :   966 lines
 *   Test  :  2,915 lines
 */
public class DatasetSplit {

    private final List<IAMSample> trainSamples;
    private final List<IAMSample> valSamples;
    private final List<IAMSample> testSamples;

    public DatasetSplit(List<IAMSample> trainSamples,
                        List<IAMSample> valSamples,
                        List<IAMSample> testSamples) {
        this.trainSamples = trainSamples;
        this.valSamples   = valSamples;
        this.testSamples  = testSamples;
    }

    public List<IAMSample> getTrainSamples() { return trainSamples; }
    public List<IAMSample> getValSamples()   { return valSamples; }
    public List<IAMSample> getTestSamples()  { return testSamples; }

    public int totalSize() {
        return trainSamples.size() + valSamples.size() + testSamples.size();
    }

    @Override
    public String toString() {
        return String.format("DatasetSplit{train=%d, val=%d, test=%d}",
                trainSamples.size(), valSamples.size(), testSamples.size());
    }
}
