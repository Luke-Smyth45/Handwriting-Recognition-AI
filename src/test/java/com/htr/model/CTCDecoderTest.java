package com.htr.model;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class CTCDecoderTest {

    @Test
    void decoderInstantiates() {
        assertDoesNotThrow(() -> new CTCDecoder(10));
    }
}
