# **Recurrent Neural Network (RNN)**

## **1. Overview**

A **Recurrent Neural Network (RNN)** is a type of neural network designed for **sequential data**. Unlike feedforward networks, RNNs have **connections that loop back**, allowing information to persist across time steps.

They excel in tasks where **context** and **temporal dependencies** are important:

* Natural Language Processing (NLP)
* Time-series forecasting
* Speech recognition
* Sequential classification

---

## **2. Key Idea**

In a standard neural network, the output at a given layer depends only on the current input.
In an **RNN**, each output depends on:

1. **Current input** $x_t$
2. **Previous hidden state** $h_{t-1}$

This allows RNNs to **remember past information** and use it to influence current predictions.

---

## **3. Mathematical Formulation**

For each time step $t$:

$$
h_t = \sigma(W_{xh}x_t + W_{hh}h_{t-1} + b_h)
$$

$$
y_t = W_{hy}h_t + b_y
$$

Where:

* $x_t$ → input vector at time step $t$
* $h_t$ → hidden state at time step $t$
* $y_t$ → output at time step $t$
* $W_{xh}, W_{hh}, W_{hy}$ → weight matrices
* $b_h, b_y$ → biases
* $\sigma$ → activation function (commonly `tanh` or `ReLU`)

---

## **4. Types of RNN Architectures**

1. **Vanilla RNN**

   * Basic RNN structure.
   * Suffers from **vanishing/exploding gradients** for long sequences.

2. **LSTM (Long Short-Term Memory)**

   * Special gates (input, forget, output) control what to keep and discard.
   * Solves long-term dependency issues.

3. **GRU (Gated Recurrent Unit)**

   * Similar to LSTM but with fewer gates and parameters.
   * Faster training.

4. **Bidirectional RNNs**

   * Processes sequences forward and backward for better context.

---

## **5. Common Use Cases**

* **Text**: sentiment analysis, translation, text generation
* **Speech**: voice recognition, speech-to-text
* **Time-series**: stock prediction, sensor monitoring
* **Music**: melody generation

---

## **6. Loss Functions**

* **Cross-Entropy Loss** → classification
* **CTC Loss** → sequence-to-sequence speech/text tasks
* **MSE (Mean Squared Error)** → regression/forecasting

---

## **7. Optimizers**

* **Adam** → adaptive learning rate, popular for RNNs
* **RMSProp** → handles non-stationary sequence data well
* **SGD + Momentum** → works with careful tuning

---

## **8. Example Implementations**

### **PyTorch**

```python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)       # out: (batch, seq_len, hidden_size)
        out = self.fc(out[:, -1])  # take last time step's output
        return out

model = SimpleRNN(input_size=10, hidden_size=50, output_size=2)
```

---

### **TensorFlow/Keras**

```python
import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.SimpleRNN(50, activation='tanh', input_shape=(None, 10)),
    layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

---

## **9. Training Workflow**

1. **Prepare Sequential Data**

   * Tokenize (for text), normalize (for time-series).
   * Pad sequences to uniform length.

2. **Model Initialization**

   * Choose RNN type (Vanilla, LSTM, GRU).
   * Select number of layers & hidden size.

3. **Train**

   * Use backpropagation through time (**BPTT**).
   * Monitor validation performance.

4. **Evaluate**

   * Sequence accuracy, F1-score, or regression error.

---

## **10. Best Practices**

* Use **LSTM or GRU** instead of vanilla RNN for long sequences.
* Apply **dropout** to prevent overfitting.
* Normalize input data for stability.
* Clip gradients to handle **exploding gradients**.
* Use **bidirectional RNNs** when future context is available.

---

## **11. Advantages**

* Can model sequential dependencies.
* Good for variable-length inputs.
* LSTM/GRU handle long-term dependencies better.

---

## **12. Limitations**

* Vanilla RNNs struggle with long-term dependencies (vanishing gradient problem).
* Training can be slow for very long sequences.
* Parallelization is harder compared to CNNs/Transformers.

---

## **13. References**

* Hochreiter & Schmidhuber (1997), *Long Short-Term Memory*
* Cho et al. (2014), *Gated Recurrent Units*
* Goodfellow et al., *Deep Learning* (MIT Press)
* PyTorch & TensorFlow official docs
