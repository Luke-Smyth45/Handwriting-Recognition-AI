# Handwriting Recognition AI

A Java application that recognises handwritten English words. You draw (or load) a word image, click **Recognise**, and the model predicts the text.

Built with [DeepLearning4J](https://deeplearning4j.konduit.ai/) on a CNN + LSTM architecture, trained on the [IAM Handwriting Word Database](https://www.kaggle.com/datasets/nibinv23/iam-handwriting-word-database).

---

## Requirements

| Requirement | Version |
|---|---|
| Java | 17 or later |
| Maven | 3.8 or later |
| CUDA Toolkit | 11.6 (GPU training only) |
| GPU | NVIDIA with CUDA 11.6 support (e.g. RTX 3070 Ti) |

> **CPU-only machines:** Replace `nd4j-cuda-11.6-platform` with `nd4j-native-platform` in `pom.xml` before building. Training will still work but will be significantly slower.

---

## Quick Start

### 1. Clone and build

```bash
git clone https://github.com/Luke-Smyth45/Handwriting-Recognition-AI.git
cd Handwriting-Recognition-AI
mvn package -q
```

This produces `target/handwriting-recognition-1.0-SNAPSHOT.jar`.

### 2. Get the dataset (for training)

Download the IAM Handwriting Word Database from Kaggle:
https://www.kaggle.com/datasets/nibinv23/iam-handwriting-word-database

Extract it so the folder structure looks like:

```
data/raw/archive/iam_words/
    words.txt
    words/
        a01/
            a01-000u/
                a01-000u-00-00.png
                ...
```

### 3. Validate the dataset

```bash
java -jar target/handwriting-recognition-1.0-SNAPSHOT.jar --validate
```

This checks that the dataset is in the correct location and all image files are readable. Fix any errors it reports before training.

### 4. Train the model

```bash
java -jar target/handwriting-recognition-1.0-SNAPSHOT.jar --train
```

- Validates the dataset automatically before starting
- Pre-loads all images into RAM for faster GPU throughput
- Saves the best model (by validation loss) to `models/htr_model.zip`
- If `models/htr_model.zip` already exists, training **resumes from that checkpoint**
- Logs training progress to console and to `logs/`

Training runs for 10 epochs with batch size 128. On an RTX 3070 Ti this takes roughly an hour.

### 5. Launch the UI

```bash
java -jar target/handwriting-recognition-1.0-SNAPSHOT.jar --ui
```

The UI loads `models/htr_model.zip` automatically if it exists.

---

## Modes

| Flag | Description |
|---|---|
| `--ui` | Launch the graphical interface (default) |
| `--train` | Validate dataset then train the model |
| `--validate` | Check dataset structure only and exit |
| `--test` | Run the pipeline smoke tests and exit |

---

## Using the UI

![UI Layout](docs/ui-screenshot.png)

The window has three sections:

**Toolbar (top)**
- **Load Model...** — load any `.zip` model file (opens a file picker pointing at the `models/` folder)
- **Load Image...** — load a PNG/JPEG/BMP image of a handwritten word instead of drawing
- **Clear** — wipe the canvas and any loaded image

**Drawing canvas (middle)**
- Draw a single handwritten word using the mouse
- The canvas is 480×120 pixels — a 4:1 aspect ratio that matches the training images exactly
- Keep your word centred and roughly filling the canvas height for best results

**Bottom section**
- **Recognise** button — runs inference on whatever is in the canvas (or the loaded image)
- Result display — shows the predicted word

### Tips for good predictions
- Write one word at a time
- Use the full height of the canvas
- Write clearly — the model was trained on handwritten but legible English words
- The model recognises: letters (upper and lower case), digits, and common punctuation

---

## Project Structure

```
src/main/java/com/htr/
    Main.java                       Entry point, CLI argument routing
    ModelTest.java                  Pipeline smoke tests (--test)
    data/
        IAMDataLoader.java          Parses words.txt and locates image files
        IAMSample.java              One training sample (image path + transcription)
        DatasetSplit.java           Train / val / test split container
        DatasetValidator.java       Validates dataset structure before training
        ImagePreprocessor.java      Resize → pad → normalise → invert image pipeline
        CharsetEncoder.java         Character ↔ integer index mapping for CTC
    model/
        ModelConfig.java            All hyperparameters and paths in one place
        ModelGraph.java             Builds the CNN + LSTM ComputationGraph
        HTRModel.java               Loads a saved model and runs inference
        CTCDecoder.java             Greedy and beam-search CTC decoding
    training/
        ModelTrainer.java           Training loop, image cache, checkpoint saving
    ui/
        HandwritingRecognitionUI.java   Main application window
        DrawingPanel.java               Free-hand mouse drawing canvas
        ResultPanel.java                Prediction result display
```

---

## Architecture

```
Input image [B, 1, 32, 128]  (batch × channels × height × width)
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│  5 × CNN Blocks (Conv 3×3 + ReLU + MaxPool)              │
│                                                          │
│  Block 1: 32  filters, pool (2,2) → [B, 32,  16,  64]   │
│  Block 2: 64  filters, pool (2,2) → [B, 64,   8,  32]   │
│  Block 3: 128 filters, pool (2,2) → [B, 128,  4,  16]   │
│  Block 4: 128 filters, no pool    → [B, 128,  4,  16]   │
│  Block 5: 256 filters, pool (2,1) → [B, 256,  2,  16]   │
│                        ↑ height-only pool, width kept    │
└──────────────────────────────────────────────────────────┘
         │
         ▼
  ReshapeVertex: [B, 256, 2, 16] → [B, 512, 16]
  (channels × height collapsed into features;
   width becomes 16 time steps)
         │
         ▼
┌──────────────────────────────────────────┐
│  LSTM 1: 512 → 256 units                 │
│  LSTM 2: 256 → 80 units (= NUM_CLASSES)  │
└──────────────────────────────────────────┘
         │
         ▼
  RnnOutputLayer (Identity activation + CTC loss)
  Output: [B, 80, 16]  (80 classes × 16 time steps)
         │
         ▼
  CTC Decoder → predicted word string
```

**Why 80 classes?** The model recognises 79 printable ASCII characters plus a CTC blank token used during decoding to separate repeated characters.

**Supported characters:**
```
 !"#&'()*+,-./0123456789:;?
 ABCDEFGHIJKLMNOPQRSTUVWXYZ
 abcdefghijklmnopqrstuvwxyz
```

---

## Configuration

All hyperparameters are in [src/main/java/com/htr/model/ModelConfig.java](src/main/java/com/htr/model/ModelConfig.java):

| Parameter | Value | Description |
|---|---|---|
| `IMG_HEIGHT` | 32 | Input image height (pixels) |
| `IMG_WIDTH` | 128 | Input image width (pixels) |
| `CNN_FILTERS` | [32, 64, 128, 128, 256] | Filters per CNN block |
| `RNN_UNITS` | 256 | LSTM hidden units |
| `NUM_CLASSES` | 80 | Characters + CTC blank |
| `BATCH_SIZE` | 128 | Training batch size |
| `EPOCHS` | 10 | Training epochs |
| `LEARNING_RATE` | 1e-4 | Adam learning rate |
| `BEAM_WIDTH` | 10 | CTC beam search width |
| `DATASET_ROOT` | `data/raw/archive/iam_words` | Dataset location |
| `MODEL_SAVE_DIR` | `models/htr_model.zip` | Model save path |

---

## Training Details

### Loss function
True CTC (Connectionist Temporal Classification) loss, implemented in `CTCLossFunction.java`. The network outputs raw logits at each of the 16 time steps; the CTC algorithm finds the best alignment between the output sequence and the target characters automatically — no manual label stretching is needed.

The raw loss reported during training is the negative log-probability summed over all 16 time steps. At the start of training, random-baseline loss is approximately 16 × ln(80) ≈ 70. Words up to 15 characters long are handled without truncation.

### Image pre-caching
Before the first epoch, all training and validation images are decoded and stored in RAM as flat `float[]` arrays. This eliminates per-batch disk I/O and is critical for keeping the GPU fed. On the IAM word dataset (~115,000 images at 32×128 = 4,096 floats each) this uses roughly 1.8 GB of RAM.

### Checkpoint saving
The model is saved to `models/htr_model.zip` whenever validation loss improves. Running `--train` again automatically resumes from this file, so training can be interrupted and continued at any time.

### Gradient clipping
LSTM gradient explosion is prevented with `ClipElementWiseAbsoluteValue` at threshold 1.0. If you see loss diverging to very large values (> 50), delete `models/htr_model.zip` and restart training.

---

## Running the Pipeline Tests

Before training on a new machine, run the smoke tests to confirm the CNN→LSTM pipeline is working correctly:

```bash
java -jar target/handwriting-recognition-1.0-SNAPSHOT.jar --test
```

Three tests run:
1. **Model build** — confirms the ComputationGraph compiles without error
2. **Forward pass shape** — confirms output is `[2, 80, 16]` not `[2, 80, 1]` (shape `[..., 1]` would indicate a broken CNN→RNN reshape)
3. **Real training batch** — confirms one batch from the actual dataset trains without error

All three must pass before running `--train`.

---

## Dependencies

| Library | Version | Purpose |
|---|---|---|
| DeepLearning4J | 1.0.0-M2.1 | Neural network framework |
| ND4J CUDA 11.6 | 1.0.0-M2.1 | GPU-accelerated tensor operations |
| Apache Commons Math | 3.6.1 | Numerical utilities |
| Apache Commons IO | 2.15.1 | File utilities |
| Logback | 1.2.12 | Logging |
| JUnit 5 | 5.10.2 | Testing |

---

## Known Limitations and Improvements

### 1. Unidirectional LSTM — switch to Bidirectional

**Current behaviour:** Both LSTM layers process the sequence left-to-right only.

**Why it matters:** Characters in a word depend on both the letters before and after them (e.g. distinguishing `rn` from `m` is easier with context from both directions). Bidirectional LSTMs are standard in HTR and typically improve word accuracy by several percentage points.

**How to fix:** DL4J has a `Bidirectional` wrapper, but it has a known bug in version 1.0.0-M2.1 when combined with `ReshapeVertex` in a `ComputationGraph` (it was the root cause of the "sequence length = 1" bug this project worked around). The fix either requires upgrading DL4J or switching the training to Python.

---

### 2. Inference still uses slow putScalar loops

**Current behaviour:** `HTRModel.java` builds the input tensor using nested `putScalar` calls (4,096 calls per inference). The trainer was updated to use `NDArrayIndex` slice assignment, but the inference path was not.

**How to fix:** Apply the same pattern used in `ModelTrainer.runBatch()`:

```java
// In HTRModel.predict(), replace the putScalar loops with:
INDArray input = Nd4j.create(ImagePreprocessor.flatten(imageData),
                             new int[]{1, 1, h, w});
```

This is a single native memory copy instead of 4,096 individual Java calls. It won't matter much for interactive use but is important if you ever run batch inference.

---

### 3. Beam search decoder is unused at inference time

**Current behaviour:** `CTCDecoder` implements both greedy and beam-search decoding, but `HTRModel.predict()` calls `greedyDecode()`. Beam search (`beamSearchDecode()`) is never used.

**Why it matters:** Beam search keeps multiple candidate sequences alive and picks the globally best one. For 16 time steps the improvement is measurable, especially for ambiguous characters.

**How to fix:** In `HTRModel.java`, change `decoder.greedyDecode(logitMatrix)` to `decoder.beamSearchDecode(logitMatrix)`. The beam width is already configurable via `ModelConfig.BEAM_WIDTH` (currently 10).

---

### 4. No data augmentation

**Current behaviour:** Training images are used as-is after resize, pad, and normalise. Every epoch sees identical data.

**Why it matters:** The IAM dataset contains ~115,000 word images but they come from a limited number of writers. Augmentation simulates more variety and reduces overfitting.

**Augmentations to add in `ImagePreprocessor.java`:**
- Random slight rotation (±5°)
- Random brightness/contrast jitter
- Random horizontal stretch (simulate different writing speeds)
- Gaussian noise

---

### 5. No test set evaluation

**Current behaviour:** Training reports `train_loss` and `val_loss` each epoch but never evaluates on the held-out test split. The test set (~2,915 samples) is loaded but unused.

**How to fix:** After training completes, run inference on `split.getTestSamples()` and compute Character Error Rate (CER) and Word Error Rate (WER) — the standard metrics for HTR. CER counts the Levenshtein edit distance between predicted and ground-truth characters; WER counts fully incorrect words. Add this as a `evaluateTest()` method in `ModelTrainer.java`.

---

### 6. Single-word only

**Current behaviour:** The model accepts one word at a time. The drawing canvas and image loader pass a single fixed-size image to the model.

**Why it matters:** Real-world use almost always involves full lines or paragraphs.

**How to fix:** This is a large architectural change. The standard approach is to add a text-line segmentation stage before the word recogniser — either a classical connected-components approach or a separate segmentation network that splits a line image into individual word crops before passing each to this model.

---

## Troubleshooting

**Training starts but GPU usage stays at 5%**
The image pre-cache phase runs before the first epoch and logs `Image cache ready: N loaded`. If this message does not appear, the cache is not working and data loading is bottlenecking the GPU.

**`No samples found in: data/raw/archive/iam_words`**
The dataset is not in the expected location. Run `--validate` for a detailed diagnosis. The `data/` folder must be in the same directory as the JAR.

**`SLF4J: No SLF4J providers were found`**
The logback dependency is missing or conflicting. Ensure `pom.xml` has `logback-classic` version `1.2.12` (not 1.4.x — that requires SLF4J 2.x which conflicts with DL4J's SLF4J 1.x).

**`Could not load existing model ... starting fresh`**
The saved model `.zip` is incompatible with the current architecture (e.g. after changing `ModelGraph.java`). Delete `models/htr_model.zip` and retrain from scratch.

**Loss diverges / goes to NaN**
LSTM gradient explosion. Delete `models/htr_model.zip` and retrain. The gradient clipping should prevent this on a fresh model.

**Output shape `[B, 80, 1]` instead of `[B, 80, 16]`**
A DL4J 1.0.0-M2.1 bug in `CnnToRnnPreProcessor` collapses the time dimension to 1. This project uses `ReshapeVertex` as a workaround. Run `--test` to confirm the fix is active. Do **not** add `.setInputTypes()` to `ModelGraph.java` — it silently overrides the ReshapeVertex.
