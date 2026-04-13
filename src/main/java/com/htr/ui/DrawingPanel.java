package com.htr.ui;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;

/**
 * A free-hand drawing canvas.
 * The user draws with the mouse; the result is captured as a BufferedImage
 * for preprocessing and model inference.
 */
public class DrawingPanel extends JPanel {

    private static final int PEN_THICKNESS = 4;
    private static final Color INK_COLOR   = Color.BLACK;
    private static final Color BG_COLOR    = Color.WHITE;

    private BufferedImage canvas;
    private Graphics2D    canvasG;
    private int           lastX, lastY;

    public DrawingPanel() {
        setBackground(BG_COLOR);
        // 480×120 = 4:1 aspect ratio, matching training images (128×32 = 4:1).
        // The preprocessor scales height to 32 (scale=32/120=0.267),
        // giving width=480*0.267=128 — exactly MODEL_WIDTH with no cropping.
        setPreferredSize(new Dimension(480, 120));
        setBorder(BorderFactory.createLineBorder(Color.GRAY, 1));

        initCanvas();
        addMouseListeners();
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /** Returns the current canvas image for model inference. */
    public BufferedImage getCanvasImage() {
        return canvas;
    }

    /** Wipe the canvas back to a white background. */
    public void clear() {
        canvasG.setColor(BG_COLOR);
        canvasG.fillRect(0, 0, canvas.getWidth(), canvas.getHeight());
        repaint();
    }

    // ── Painting ──────────────────────────────────────────────────────────────

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        // If panel is resized, recreate canvas
        if (canvas.getWidth() != getWidth() || canvas.getHeight() != getHeight()) {
            BufferedImage old = canvas;
            initCanvas();
            canvasG.drawImage(old, 0, 0, null);
        }
        g.drawImage(canvas, 0, 0, null);
    }

    // ── Internals ─────────────────────────────────────────────────────────────

    private void initCanvas() {
        int w = Math.max(getWidth(),  480);
        int h = Math.max(getHeight(), 120);
        canvas  = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        canvasG = canvas.createGraphics();
        canvasG.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        canvasG.setRenderingHint(RenderingHints.KEY_STROKE_CONTROL, RenderingHints.VALUE_STROKE_PURE);
        canvasG.setColor(BG_COLOR);
        canvasG.fillRect(0, 0, w, h);
    }

    private void addMouseListeners() {
        addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                lastX = e.getX();
                lastY = e.getY();
            }
        });

        addMouseMotionListener(new MouseMotionAdapter() {
            @Override
            public void mouseDragged(MouseEvent e) {
                canvasG.setColor(INK_COLOR);
                canvasG.setStroke(new BasicStroke(
                        PEN_THICKNESS,
                        BasicStroke.CAP_ROUND,
                        BasicStroke.JOIN_ROUND));
                canvasG.drawLine(lastX, lastY, e.getX(), e.getY());
                lastX = e.getX();
                lastY = e.getY();
                repaint();
            }
        });
    }
}
