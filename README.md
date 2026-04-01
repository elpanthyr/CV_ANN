# CV-ANN for Automatic Modulation Classification (AMC)

## Overview
This repository contains the implementation and evaluation pipeline for a **Complex-Valued Artificial Neural Network (CV-ANN)** designed to classify radio frequency (RF) modulations. 

Traditional neural networks treat the In-phase (I) and Quadrature (Q) components of a radio signal as two independent, real-valued spatial channels. This CV-ANN approach natively treats I/Q samples as complex numbers ($I + jQ$). By utilizing complex arithmetic within the convolutional layers, the network preserves the fundamental physical geometry of the signal, allowing it to natively extract phase shifts and magnitude variations that define dense modulation schemes.

<img width="1024" height="572" alt="image" src="https://github.com/user-attachments/assets/f520169c-a495-4b04-acf3-c04f0d3da98d" />

## The Dataset & Chunking Strategy
This project utilizes the **RadioML 2018.01A** dataset, which contains 24 distinct analog and digital modulation classes. 

Due to the massive scale of the complete dataset (over 2.5 million samples and 5.2 billion individual data points), training a single monolithic classifier on limited hardware (e.g., standard T4 GPUs) is computationally prohibitive. To solve this, the dataset is divided into **9 disjoint physical chunks**, grouping modulations by their underlying physical families:

* **Chunk 1 (Digital Amplitude):** OOK, 4ASK, 8ASK
* **Chunk 2 (Digital Phase - Low):** BPSK, QPSK, 8PSK
* **Chunk 3 (Digital Phase - High):** 16PSK, 32PSK
* **Chunk 4 (Ring APSK - Low):** 16APSK, 32APSK, 64APSK
* **Chunk 5 (Dense Mixed):** 128APSK, 16QAM, 32QAM
* **Chunk 6 (Dense Grid QAM):** 64QAM, 128QAM, 256QAM
* **Chunk 7 (Analog AM - SSB):** AM-SSB-WC, AM-SSB-SC
* **Chunk 8 (Analog AM - DSB):** AM-DSB-WC, AM-DSB-SC
* **Chunk 9 (Frequency/Phase Outliers):** FM, GMSK, OQPSK

Individual CV-ANN specialist models are trained on these specific subsets, allowing the network to focus its mathematical capacity on distinguishing highly similar constellations (e.g., 64-QAM vs. 128-QAM) without being distracted by vast topological differences across unrelated families.

## Architecture & Custom Layers
Because standard deep learning frameworks (like TensorFlow/Keras) are optimized for real numbers, this repository implements a custom suite of complex-valued operations. 

When loading the `.keras` models, the following custom objects are required to reconstruct the computational graph:

* `ComplexConv1D`: Performs convolution using complex arithmetic, cross-multiplying the real and imaginary weights against the I and Q inputs.
* `ModReLU` (Modulus ReLU): A complex-friendly activation function that applies a standard ReLU to the magnitude of the complex vector while preserving its phase angle.
* `MagnitudePooling1D`: Downsamples the signal by calculating the magnitude of the complex vectors before pooling.
* `DeterministicPhysicsExtractor`: Manually extracts physical features such as DC offset, spectral asymmetry, and variance.
* `MagnitudeSEBlock`: A Squeeze-and-Excitation block adapted to recalibrate channel weights based on complex magnitude.

## Evaluation Methodology: The 4 dB SNR Threshold
Performance is strictly evaluated using an **Honest Per-Specialist** routing approach, where each model is tested exclusively on the domain it was trained to recognize.

Furthermore, overall system capability is benchmarked at a hard cutoff of **$\ge$ 4 dB Signal-to-Noise Ratio (SNR)**. 
* **Why 4 dB?** In dense modulation schemes (such as 256-QAM or 64-APSK), the physical distance between constellation points is microscopic. Below 4 dB SNR, thermal background noise physically expands these points into overlapping clouds, permanently destroying the geometric boundaries. Evaluating high-order modulations below this threshold measures noise rather than model learning capacity.

## Installation & Setup

### Requirements
* Python 3.8+
* TensorFlow 2.x
* NumPy
* h5py
* scikit-learn
* matplotlib
* seaborn

### Loading the Models
Because the network relies on custom complex arithmetic layers, you must pass the `custom_objects` dictionary when loading the models for inference:

```python
import tensorflow as tf
from custom_layers import (
    ComplexConv1D, ModReLU, MagnitudePooling1D, 
    TemporalAttention, DeterministicPhysicsExtractor, 
    MagnitudeSEBlock, SpectralFeatureBlock, DCOffsetExtractor
)

# Define the registry
custom_objects = {
    'ComplexConv1D': ComplexConv1D, 
    'ModReLU': ModReLU, 
    'MagnitudePooling1D': MagnitudePooling1D, 
    'TemporalAttention': TemporalAttention,
    'DeterministicPhysicsExtractor': DeterministicPhysicsExtractor, 
    'MagnitudeSEBlock': MagnitudeSEBlock,
    'SpectralFeatureBlock': SpectralFeatureBlock, 
    'DCOffsetExtractor': DCOffsetExtractor
}

# Load a specialist chunk
model = tf.keras.models.load_model(
    'cv_ann_modulation_chunk_1.keras', 
    custom_objects=custom_objects, 
    compile=False
)
```

## Running the Evaluation Pipeline
The evaluation file (`CV_ANN_test_eval.ipynb`) performs the following operations:
1.  Loads the global `radioml_global_test.h5` dataset.
2.  Iterates through all 9 chunked models.
3.  Filters the test data so each model only infers on its designated domain.
4.  Stitches the local predictions back into the global 24-class index.
5.  Generates a unified Classification Report, an SNR degradation curve, and a Global Confusion Matrix.


## Results
The CV-ANN natively excels at identifying rigid geometric structures in the I/Q plane. 
* **Strengths:** Achieves near-perfect classification on discrete amplitude and phase shifts (ASK, basic PSK, AM families).
* **Limitations:** While highly accurate on low-order schemes, the pure CV-ANN architecture can struggle to maintain separation on ultra-dense grids (128-QAM, 256-QAM) compared to hybrid spatio-temporal models (like CLDNN), requiring extremely clean SNR to hold the boundaries.
