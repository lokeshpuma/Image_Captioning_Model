# Image Captioning Model

A deep learning project that generates natural language captions for images using CNN-LSTM architecture trained on the Flickr8k dataset.

## 🎯 Overview

This project implements an encoder-decoder architecture combining:
- **Encoder**: Xception CNN for visual feature extraction
- **Decoder**: LSTM-based sequence generator for caption generation

## 📋 Table of Contents

- [Environment Setup](#environment-setup)
- [Dataset Structure](#dataset-structure)
- [Workflow](#workflow)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Future Improvements](#future-improvements)

## 🛠️ Environment Setup

### Requirements
- Python 3.9+
- Virtual environment (recommended)

### Installation

```bash
cd ~/Desktop/image
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install tensorflow pillow matplotlib h5py numpy
```

## 📁 Dataset Structure

### Download Links

- **Flicker8k_Dataset (images)**: [Download from Google Drive](https://drive.google.com/file/d/1u3oqx36XApnAykFDB6EEWUIfd_CxRQQ9/view)
- **Flickr8k_text (captions)**: [Download from Google Drive](https://drive.google.com/file/d/1qcRy3WpQv4dGtu65gETtYLWxDPBrRtx1/view)

After downloading, extract them so the folder structure matches:

```
project_root/
├── Flicker8k_Dataset/          # Image files
│   └── *.jpg
├── Flickr8k_text/              # Caption data
│   ├── Flickr8k.token.txt      # Image-caption mappings
│   └── Flickr_8k.trainImages.txt  # Training image list
├── main.py                     # Training pipeline
├── test.py                     # Inference script
└── venv/                       # Virtual environment
```

### Generated Files (after training)

```
project_root/
├── descriptions.txt            # Cleaned captions
├── features.p                  # Precomputed CNN features
├── tokenizer.p                 # Fitted tokenizer
├── max_length.txt             # Maximum caption length
└── models/                    # Model checkpoints
    ├── model_0.h5
    ├── model_1.h5
    └── ...
```

## 🔄 Workflow

### Phase 1: Text Preparation

**Script**: `main.py`

1. **Load Raw Captions**
   - Reads `Flickr8k_text/Flickr8k.token.txt`
   - Groups 5 captions per image into dictionary format

2. **Caption Cleaning**
   - Converts text to lowercase
   - Removes punctuation
   - Filters non-alphabetic tokens
   - Example: `"A dog jumping."` → `"a dog jumping"`

3. **Add Special Tokens**
   - Wraps captions: `"<start> a dog jumping <end>"`
   - Enables sequence-to-sequence training

4. **Build Vocabulary**
   - Extracts unique words from all captions
   - Fits Keras `Tokenizer` on cleaned text
   - Saves to `tokenizer.p`
   - Computes and saves maximum caption length to `max_length.txt`

5. **Save Processed Data**
   - Writes cleaned captions to `descriptions.txt`

---

### Phase 2: Visual Feature Extraction

**Script**: `main.py`

1. **CNN Backbone Setup**
   - Model: Xception (from `keras.applications`)
   - Configuration:
     - `include_top=False` (remove classification layer)
     - `pooling='avg'` (global average pooling)
     - Output: 2048-dimensional feature vector

2. **Image Preprocessing**
   - Resize to 299×299 pixels
   - Convert to NumPy array
   - Normalize to range [-1, 1]

3. **Feature Extraction**
   - Process each training image through CNN
   - Generate 2048-D feature vector per image
   - Store as dictionary: `{filename: feature_vector}`
   - Save to `features.p` for reuse

---

### Phase 3: Training Data Generation

**Script**: `main.py`

1. **Sequence Creation**
   - For each (image, caption) pair:
     - Convert caption to token IDs
     - Generate partial sequences for each position

2. **Input-Output Pairs**
   - For position `i` in caption:
     - **Input 1**: Image features (2048-D)
     - **Input 2**: Token sequence [:i] (padded to max_length)
     - **Output**: One-hot encoded next word

3. **Batch Generation**
   - Yields batches via `tf.data.Dataset`:
     - `input_1`: shape `(batch_size, 2048)`
     - `input_2`: shape `(batch_size, max_length)`
     - `output`: shape `(batch_size, vocab_size)`

---

### Phase 4: Model Architecture

**Scripts**: `main.py`, `test.py`

| Component            | Input Shape        | Layers (in order)                                                                 | Output Shape           |
|----------------------|--------------------|------------------------------------------------------------------------------------|------------------------|
| **Image branch**     | `(2048,)`          | `Dropout(0.5) → Dense(256, relu)`                                                  | `(256,)`               |
| **Text branch**      | `(max_length,)`    | `Embedding(vocab_size, 256, mask_zero=True) → Dropout(0.5) → LSTM(256)`           | `(256,)`               |
| **Fusion**           | two `(256,)` vectors | `Add()` (image + text) → Dense(256, relu)                                          | `(256,)`               |
| **Output layer**     | `(256,)`           | `Dense(vocab_size, softmax)`                                                       | `(vocab_size,)`        |

This forms a classic **encoder–decoder** architecture where:
- The **encoder** maps each image to a 256‑D feature vector.
- The **decoder** consumes the partial caption and, together with the image vector, predicts the **next word** at each time step.

**Compilation**:
- **Loss**: Categorical cross-entropy
- **Optimizer**: Adam

---

### Phase 5: Training Loop

**Script**: `main.py`

1. **Compute Training Steps**
   - Calculate `steps_per_epoch` from total sequences

2. **Epoch Iteration**
   - For each epoch:
     - Build/reuse dataset generator
     - Call `model.fit()` for 1 epoch
     - Save model checkpoint: `models/model_<epoch>.h5`

3. **Progress Tracking**
   - Loss typically decreases: ~5.0 → ~3.x
   - Each epoch saves a new checkpoint

---

### Phase 6: Inference

**Script**: `test.py`

1. **Load Components**
   ```
   max_length.txt → maximum sequence length
   tokenizer.p → word-to-index mapping
   model_9.h5 → trained model weights
   ```

2. **Extract Image Features**
   - Load and preprocess input image
   - Pass through Xception CNN
   - Generate 2048-D feature vector

3. **Generate Caption Iteratively**
   ```
   Initialize: text = "<start>"
   
   Loop (max_length iterations):
     1. Convert text to token sequence
     2. Pad sequence to max_length
     3. Predict next word: model.predict([features, sequence])
     4. Get word with highest probability (argmax)
     5. Append word to text
     6. Stop if word == "<end>" or invalid
   
   Return: final caption string
   ```

4. **Display Results**
   - Print generated caption
   - Show image using matplotlib

---

## 🚀 Usage

### Training the Model

```bash
cd ~/Desktop/image
source venv/bin/activate
python main.py
```

**Outputs**:
- `descriptions.txt`
- `features.p`
- `tokenizer.p`
- `max_length.txt`
- `models/model_*.h5` checkpoints

### Generating Captions

```bash
source venv/bin/activate
python test.py -i Flicker8k_Dataset/189721896_1ffe76d89e.jpg
```

**Replace** `-i` argument with any image path.

---

## 📂 Project Structure

```
image/
├── Flicker8k_Dataset/
│   └── [image files]
├── Flickr8k_text/
│   ├── Flickr8k.token.txt
│   └── Flickr_8k.trainImages.txt
├── models/
│   ├── model_0.h5
│   ├── model_1.h5
│   └── ...
├── main.py              # Training pipeline
├── test.py              # Inference script
├── descriptions.txt     # Processed captions
├── features.p           # CNN features
├── tokenizer.p          # Vocabulary tokenizer
├── max_length.txt       # Sequence length
└── README.md            # This file
```

---

## 🎯 Future Improvements

### Model Enhancements
- **Pretrained Weights**: Enable `weights='imagenet'` for Xception to improve feature quality
- **Attention Mechanism**: Add attention layers to focus on relevant image regions
- **Beam Search**: Replace greedy decoding with beam search for better captions

### Training Optimization
- **GPU Acceleration**: Leverage GPU for faster training
- **Learning Rate Scheduling**: Implement adaptive learning rates
- **Early Stopping**: Monitor validation loss to prevent overfitting

### Evaluation Metrics
- **BLEU Score**: Measure n-gram overlap with reference captions
- **METEOR**: Consider synonyms and stemming
- **CIDEr**: Optimize for human consensus
- **SPICE**: Evaluate semantic relationships

### Infrastructure
- **Checkpoint Management**: Auto-save best model based on validation metrics
- **Validation Set**: Split data for proper model selection
- **Configuration File**: Externalize hyperparameters
- **Logging**: Add TensorBoard integration

---

## 📝 Notes

- Training loss typically ranges from ~5.0 (initial) to ~3.x (converged)
- Stop training anytime with `Ctrl+C` - completed checkpoints remain usable
- Each epoch generates one model file in `models/` directory
- For best results, use GPU and pretrained CNN weights

---

## 📄 License

This project is for educational purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

---

**Author**: [lokesh] 
**Last Updated**: February 2026
