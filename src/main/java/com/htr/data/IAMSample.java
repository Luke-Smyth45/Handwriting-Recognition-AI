package com.htr.data;

import java.awt.image.BufferedImage;

/**
 * Represents a single IAM dataset sample — an image paired with its ground-truth transcription.
 */
public class IAMSample {

    /** Unique IAM identifier, e.g. "a01-000u-00" */
    private final String id;

    /** Ground-truth transcription text */
    private final String transcription;

    /** Absolute path to the source image file */
    private final String imagePath;

    /** Loaded image (null until preprocessed) */
    private BufferedImage image;

    public IAMSample(String id, String transcription, String imagePath) {
        this.id = id;
        this.transcription = transcription;
        this.imagePath = imagePath;
    }

    public String getId()            { return id; }
    public String getTranscription() { return transcription; }
    public String getImagePath()     { return imagePath; }
    public BufferedImage getImage()  { return image; }
    public void setImage(BufferedImage image) { this.image = image; }

    @Override
    public String toString() {
        return "IAMSample{id='" + id + "', transcription='" + transcription + "'}";
    }
}
