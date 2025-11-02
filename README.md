# Generative AI: Character-Level Text Generation with an RNN

This project demonstrates how to build and train a character-level Recurrent Neural Network (RNN) for text generation. The model is trained on the complete works of Shakespeare and learns to generate new text, one character at a time, that mimics his writing style.

This notebook provides an end-to-end walkthrough of the process, from data preparation to model building, training, and generating new text.

## 🚀 Project Overview

The `Text_Generation_RNN_Shakespeare.ipynb` notebook covers the complete workflow:

1.  **Data Loading & Preparation:**
    * Downloads Shakespeare's combined works as a single text file.
    * Creates a complete vocabulary of all unique characters in the text.

2.  **Vectorization:**
    * Builds a `tf.keras.layers.StringLookup` layer to map each character to a unique integer (tokenization).
    * Creates an inverse layer to map integers back to characters for output.

3.  **Data Pipelining:**
    * Converts the entire text into a stream of token IDs.
    * Uses `tf.data.Dataset` to create a highly efficient pipeline.
    * Batches the data into sequences of a fixed length (e.g., 100 characters).
    * Generates input/target pairs, where the target sequence is the input sequence shifted by one character.

4.  **Model Architecture:**
    * Builds a `tf.keras.Sequential` model.
    * **Embedding Layer:** Converts integer tokens into dense vectors of a specific size.
    * **GRU Layer:** A Gated Recurrent Unit (a modern RNN cell) processes the sequences and learns patterns.
    * **Dense Layer:** A final, fully-connected layer that outputs logits for every character in the vocabulary.

5.  **Training & Generation:**
    * Trains the model using `SparseCategoricalCrossentropy` loss.
    * Implements a custom training loop with `tf.GradientTape` for one-step training.
    * Demonstrates how to generate new text by feeding a starting "seed" string, predicting the next character, and feeding that character back into the model in a loop.

---

## 🛠️ Setup and Installation

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/your-username/Generative-AI-Text-Generation-RNN.git](https://github.com/your-username/Generative-AI-Text-Generation-RNN.git)
    cd Generative-AI-Text-Generation-RNN
    ```

2.  **Create a Virtual Environment** (Recommended)
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Required Libraries**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Launch Jupyter**
    ```bash
    jupyter notebook
    ```
    Open the `Text_Generation_RNN_Shakespeare.ipynb` notebook to run the project. The dataset is downloaded automatically.
