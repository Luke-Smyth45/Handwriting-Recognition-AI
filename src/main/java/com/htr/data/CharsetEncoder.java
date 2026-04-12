package com.htr.data;

import com.htr.model.ModelConfig;

import java.util.HashMap;
import java.util.Map;

/**
 * Bidirectional mapping between characters and integer indices used by the CTC layer.
 *
 * Index 0 … (NUM_CLASSES - 2) map to CHARSET characters.
 * Index NUM_CLASSES - 1         is the CTC blank token.
 */
public class CharsetEncoder {

    private final Map<Character, Integer> charToIdx = new HashMap<>();
    private final Map<Integer, Character> idxToChar = new HashMap<>();

    public CharsetEncoder() {
        String charset = ModelConfig.CHARSET;
        for (int i = 0; i < charset.length(); i++) {
            charToIdx.put(charset.charAt(i), i);
            idxToChar.put(i, charset.charAt(i));
        }
        // Blank token is implicitly the last index — not in charToIdx
    }

    /** Encode a string as a sequence of integer label indices. */
    public int[] encode(String text) {
        int[] labels = new int[text.length()];
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            Integer idx = charToIdx.get(c);
            if (idx == null) {
                // Unknown character — map to blank (will be treated as unknown)
                labels[i] = ModelConfig.BLANK_INDEX;
            } else {
                labels[i] = idx;
            }
        }
        return labels;
    }

    /** Decode a sequence of integer indices back to a string, ignoring blanks. */
    public String decode(int[] indices) {
        StringBuilder sb = new StringBuilder();
        for (int idx : indices) {
            if (idx == ModelConfig.BLANK_INDEX) continue;
            Character c = idxToChar.get(idx);
            if (c != null) sb.append(c);
        }
        return sb.toString();
    }

    public int charsetSize() {
        return ModelConfig.NUM_CLASSES;
    }
}
