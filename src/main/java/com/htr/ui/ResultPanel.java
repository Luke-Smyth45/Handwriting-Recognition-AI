package com.htr.ui;

import javax.swing.*;
import java.awt.*;

/**
 * Displays the model's predicted transcription and a confidence hint.
 */
public class ResultPanel extends JPanel {

    private final JLabel predictionLabel;
    private final JLabel statusLabel;

    public ResultPanel() {
        setLayout(new BorderLayout(8, 4));
        setBorder(BorderFactory.createTitledBorder("Recognition Result"));
        setBackground(new Color(245, 245, 250));

        predictionLabel = new JLabel(" ", SwingConstants.CENTER);
        predictionLabel.setFont(new Font("Serif", Font.BOLD, 26));
        predictionLabel.setForeground(new Color(30, 80, 160));

        statusLabel = new JLabel(" ", SwingConstants.CENTER);
        statusLabel.setFont(new Font("SansSerif", Font.PLAIN, 12));
        statusLabel.setForeground(Color.GRAY);

        add(predictionLabel, BorderLayout.CENTER);
        add(statusLabel,     BorderLayout.SOUTH);
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /** Show a recognised text result. */
    public void showResult(String text) {
        predictionLabel.setForeground(new Color(30, 80, 160));
        predictionLabel.setText(text.isEmpty() ? "(empty)" : text);
        statusLabel.setText("Recognised");
    }

    /** Show an error or status message. */
    public void showStatus(String message) {
        predictionLabel.setForeground(Color.GRAY);
        predictionLabel.setText(message);
        statusLabel.setText("");
    }

    /** Show a loading indicator while the model is running. */
    public void showProcessing() {
        predictionLabel.setForeground(Color.DARK_GRAY);
        predictionLabel.setText("Processing...");
        statusLabel.setText("");
    }

    /** Clear back to a blank state. */
    public void clear() {
        predictionLabel.setText(" ");
        statusLabel.setText(" ");
    }
}
