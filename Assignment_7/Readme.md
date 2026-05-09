# Deep Learning Mathematical Problems - Complete Solutions

## Problem 1: Class Activation Map (CAM)

### a. Global Average Pooling (GAP) Outputs

Global Average Pooling computes the mean of each feature map:

$$F_1 = \frac{1 + 2 + 3 + 4}{4} = \frac{10}{4} = 2.5$$

$$F_2 = \frac{0 + 1 + 1 + 0}{4} = \frac{2}{4} = 0.5$$

$$F_3 = \frac{2 + 1 + 0 + 1}{4} = \frac{4}{4} = 1.0$$

### b. Class Score

$$S_c = \sum_{k} w_k^c F_k = w_1^c F_1 + w_2^c F_2 + w_3^c F_3$$

$$S_c = (2)(2.5) + (-1)(0.5) + (1)(1.0) = 5.0 - 0.5 + 1.0 = 5.5$$

### c. Class Activation Map

The CAM is computed as:
$$M_c(x,y) = \sum_{k} w_k^c f_k(x,y)$$

For each position (x,y) in the 2×2 feature maps:

$$M_c(1,1) = 2(1) + (-1)(0) + 1(2) = 2 + 0 + 2 = 4$$

$$M_c(1,2) = 2(2) + (-1)(1) + 1(1) = 4 - 1 + 1 = 4$$

$$M_c(2,1) = 2(3) + (-1)(1) + 1(0) = 6 - 1 + 0 = 5$$

$$M_c(2,2) = 2(4) + (-1)(0) + 1(1) = 8 + 0 + 1 = 9$$

**CAM Matrix:**
$$M_c = \begin{bmatrix} 4 & 4 \\ 5 & 9 \end{bmatrix}$$

### d. Normalize the CAM

$$M'_c(x,y) = \frac{M_c(x,y) - \min(M_c)}{\max(M_c) - \min(M_c)}$$

- min(M) = 4
- max(M) = 9
- Range = 9 - 4 = 5

$$M'_c = \begin{bmatrix} 0 & 0 \\ 0.2 & 1 \end{bmatrix}$$

### e. Mapping 2×2 CAM back to 5×5 Input Image

**Spatial Mapping:**
- Input image: 5×5
- Feature map: 2×2
- Upsampling factor: 5/2 = 2.5

Each cell in the 2×2 CAM is mapped to a 2.5×2.5 region in the input image:
- CAM[1,1] → Input region [0:2.5, 0:2.5] (top-left region)
- CAM[1,2] → Input region [0:2.5, 2.5:5] (top-right region)
- CAM[2,1] → Input region [2.5:5, 0:2.5] (bottom-left region)
- CAM[2,2] → Input region [2.5:5, 2.5:5] (bottom-right region)

Using bilinear interpolation or nearest neighbor:

$$\text{Upsampled CAM} = \begin{bmatrix} 0 & 0 & 0 & 0.2 & 0.2 \\ 0 & 0 & 0 & 0.2 & 0.2 \\ 0 & 0 & 0.2 & 1 & 1 \\ 0.2 & 0.2 & 1 & 1 & 1 \\ 0.2 & 0.2 & 1 & 1 & 1 \end{bmatrix}$$

---

## Problem 2: CAM with Custom Data Sets

**Set 1:**
- Input: $I_1 = \begin{bmatrix} 2 & 1 & 3 \\ 0 & 2 & 1 \\ 1 & 2 & 3 \end{bmatrix}$ (3×3, produces 2×2 feature maps)
- $f_1^{(1)} = \begin{bmatrix} 2 & 3 \\ 1 & 2 \end{bmatrix}$, $f_2^{(1)} = \begin{bmatrix} 1 & 0 \\ 2 & 1 \end{bmatrix}$, $f_3^{(1)} = \begin{bmatrix} 3 & 2 \\ 0 & 1 \end{bmatrix}$
- $w^c = [1.5, 0.5, 2]$

**Steps:**
$$F_1 = \frac{2+3+1+2}{4} = 2.0$$
$$F_2 = \frac{1+0+2+1}{4} = 1.0$$
$$F_3 = \frac{3+2+0+1}{4} = 1.5$$

$$S_c = 1.5(2.0) + 0.5(1.0) + 2(1.5) = 3.0 + 0.5 + 3.0 = 6.5$$

$$M_c(1,1) = 1.5(2) + 0.5(1) + 2(3) = 3 + 0.5 + 6 = 9.5$$
$$M_c(1,2) = 1.5(3) + 0.5(0) + 2(2) = 4.5 + 0 + 4 = 8.5$$
$$M_c(2,1) = 1.5(1) + 0.5(2) + 2(0) = 1.5 + 1 + 0 = 2.5$$
$$M_c(2,2) = 1.5(2) + 0.5(1) + 2(1) = 3 + 0.5 + 2 = 5.5$$

$$M_c = \begin{bmatrix} 9.5 & 8.5 \\ 2.5 & 5.5 \end{bmatrix}$$

**Normalize:** (min=2.5, max=9.5, range=7)

$$M'_c = \begin{bmatrix} 1.0 & 0.857 \\ 0 & 0.429 \end{bmatrix}$$

**Set 2:**
- Input: $I_2 = \begin{bmatrix} 1 & 1 & 2 \\ 3 & 2 & 1 \\ 1 & 1 & 2 \end{bmatrix}$
- $f_1^{(2)} = \begin{bmatrix} 1 & 2 \\ 3 & 2 \end{bmatrix}$, $f_2^{(2)} = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$, $f_3^{(2)} = \begin{bmatrix} 1 & 2 \\ 2 & 1 \end{bmatrix}$
- $w^c = [2, 1, -0.5]$

$$F_1 = 2.0, F_2 = 1.5, F_3 = 1.5$$
$$S_c = 2(2.0) + 1(1.5) + (-0.5)(1.5) = 4 + 1.5 - 0.75 = 4.75$$

$$M_c = \begin{bmatrix} 2(1)+1(2)-0.5(1) & 2(2)+1(1)-0.5(2) \\ 2(3)+1(1)-0.5(2) & 2(2)+1(2)-0.5(1) \end{bmatrix}$$

$$M_c = \begin{bmatrix} 3.5 & 4 \\ 6.5 & 5.5 \end{bmatrix}$$

**Normalize:** (min=3.5, max=6.5)

$$M'_c = \begin{bmatrix} 0 & 0.167 \\ 1.0 & 0.667 \end{bmatrix}$$

**Set 3:**
- Input: $I_3 = \begin{bmatrix} 2 & 2 & 1 \\ 1 & 3 & 2 \\ 2 & 1 & 3 \end{bmatrix}$
- $f_1^{(3)} = \begin{bmatrix} 2 & 1 \\ 1 & 3 \end{bmatrix}$, $f_2^{(3)} = \begin{bmatrix} 1 & 2 \\ 2 & 1 \end{bmatrix}$, $f_3^{(3)} = \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix}$
- $w^c = [0.5, 2, 1]$

$$F_1 = 1.75, F_2 = 1.5, F_3 = 2.0$$
$$S_c = 0.5(1.75) + 2(1.5) + 1(2.0) = 0.875 + 3 + 2 = 5.875$$

$$M_c = \begin{bmatrix} 0.5(2)+2(1)+1(3) & 0.5(1)+2(2)+1(1) \\ 0.5(1)+2(2)+1(1) & 0.5(3)+2(1)+1(3) \end{bmatrix}$$

$$M_c = \begin{bmatrix} 6 & 6.5 \\ 6.5 & 5.5 \end{bmatrix}$$

---

## Problem 3: Grad-CAM for Given Network

**Grad-CAM Formula:**
$$L_c^{(k)} = \frac{1}{Z} \sum_{i,j} \frac{\partial S_c}{\partial A_{ij}^{(k)}} \cdot A_{ij}^{(k)}$$

where $\frac{\partial S_c}{\partial A_{ij}^{(k)}} = w_k^c$

**Steps:**

1. **Forward Pass:** Already computed in Problem 1
   - $f_1, f_2, f_3$ (feature maps)
   - $S_c = 5.5$ (class score)

2. **Backward Pass:** Compute gradients
   $$\frac{\partial S_c}{\partial f_1} = w_1^c = 2$$
   $$\frac{\partial S_c}{\partial f_2} = w_2^c = -1$$
   $$\frac{\partial S_c}{\partial f_3} = w_3^c = 1$$

3. **Compute Gradient-weighted Activations:**
   $$L_c = \frac{\partial S_c}{\partial f_1} \cdot f_1 + \frac{\partial S_c}{\partial f_2} \cdot f_2 + \frac{\partial S_c}{\partial f_3} \cdot f_3$$

   $$L_c = 2 \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} + (-1) \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} + 1 \begin{bmatrix} 2 & 1 \\ 0 & 1 \end{bmatrix}$$

   $$L_c(1,1) = 2(1) - 1(0) + 1(2) = 2 - 0 + 2 = 4$$
   $$L_c(1,2) = 2(2) - 1(1) + 1(1) = 4 - 1 + 1 = 4$$
   $$L_c(2,1) = 2(3) - 1(1) + 1(0) = 6 - 1 + 0 = 5$$
   $$L_c(2,2) = 2(4) - 1(0) + 1(1) = 8 + 0 + 1 = 9$$

   $$L_c = \begin{bmatrix} 4 & 4 \\ 5 & 9 \end{bmatrix}$$

**Note:** For this simple case, Grad-CAM = CAM because the gradients w.r.t. the pooling operation are proportional to the weights.

4. **Apply ReLU:**
   $$L_c^{ReLU} = \max(0, L_c) = \begin{bmatrix} 4 & 4 \\ 5 & 9 \end{bmatrix}$$

5. **Normalize:**
   $$L'_c = \frac{L_c - \min(L_c)}{\max(L_c) - \min(L_c)} = \begin{bmatrix} 0 & 0 \\ 0.2 & 1 \end{bmatrix}$$

---

## Problem 4: Grad-CAM with Multiple FC Layers

**Architecture Setup:**
```
[Conv Layer] → [Feature Maps] → [GAP] → [FC1] → [FC2] → [Output]
```

**Forward Pass with Multiple FC Layers:**

Let:
- Feature map outputs: $A = [F_1, F_2, F_3]$ (from GAP)
- FC1 weights: $W^{(1)}$ (shape: 3×5), bias $b^{(1)}$
- FC2 weights: $W^{(2)}$ (shape: 5×3), bias $b^{(2)}$

$$h_1 = \sigma(W^{(1)} A + b^{(1)}) \text{ (5 hidden units)}$$
$$S_c = W^{(2)} h_1 + b^{(2)} \text{ (class score)}$$

**Backward Pass:**

1. **Gradient w.r.t. FC2 output:**
   $$\frac{\partial S_c}{\partial h_1} = W^{(2)^T}$$

2. **Gradient w.r.t. FC1 output (through activation):**
   $$\frac{\partial S_c}{\partial A} = \frac{\partial S_c}{\partial h_1} \cdot \frac{\partial h_1}{\partial A} = W^{(2)^T} \cdot \sigma'(z_1) \cdot W^{(1)}$$

3. **Compute gradient-weighted feature maps:**
   $$L_c^{(k)} = \sum_{i,j} \frac{\partial S_c}{\partial A_{ij}^{(k)}} \cdot A_{ij}^{(k)}$$

**Example with Concrete Values:**

Let:
- $F = [2.5, 0.5, 1.0]^T$
- $W^{(1)} = \begin{bmatrix} 0.5 & 1 & 0.2 \\ 1 & -0.5 & 1 \\ 0.2 & 1 & 0.5 \\ 1 & 0 & 0.3 \\ 0 & 1 & -0.2 \end{bmatrix}$, $b^{(1)} = [0, 0, 0, 0, 0]^T$

$$z_1 = W^{(1)} F = \begin{bmatrix} 1.25 + 0.5 + 0.2 \\ 2.5 - 0.25 + 1 \\ 0.5 + 0.5 + 0.5 \\ 2.5 + 0 + 0.3 \\ 0 + 0.5 - 0.2 \end{bmatrix} = \begin{bmatrix} 1.95 \\ 3.25 \\ 1.5 \\ 2.8 \\ 0.3 \end{bmatrix}$$

$$h_1 = \text{ReLU}(z_1) = \begin{bmatrix} 1.95 \\ 3.25 \\ 1.5 \\ 2.8 \\ 0.3 \end{bmatrix}$$

- $W^{(2)} = \begin{bmatrix} 1 & 0.5 & -0.2 & 1 & 0.1 \\ 0.2 & 1 & 0.5 & 0 & 0.5 \\ -0.5 & 0 & 1 & 0.5 & -0.2 \end{bmatrix}$, $b^{(2)} = [0, 0, 0]^T$

$$S = W^{(2)} h_1$$

For class c=0:
$$S_0 = 1(1.95) + 0.5(3.25) - 0.2(1.5) + 1(2.8) + 0.1(0.3) = 1.95 + 1.625 - 0.3 + 2.8 + 0.03 = 6.155$$

**Backpropagation:**

$$\frac{\partial S_0}{\partial h_1} = W^{(2)[0]} = [1, 0.5, -0.2, 1, 0.1]$$

$$\frac{\partial h_1}{\partial z_1} = \text{diag}(\text{ReLU}'(z_1)) = I \text{ (all positive)}$$

$$\frac{\partial z_1}{\partial F} = W^{(1)} = \begin{bmatrix} 0.5 & 1 & 0.2 \\ 1 & -0.5 & 1 \\ 0.2 & 1 & 0.5 \\ 1 & 0 & 0.3 \\ 0 & 1 & -0.2 \end{bmatrix}$$

**Combined gradient:**
$$\frac{\partial S_0}{\partial F} = \frac{\partial S_0}{\partial h_1} \cdot W^{(1)} = [1, 0.5, -0.2, 1, 0.1] \begin{bmatrix} 0.5 & 1 & 0.2 \\ 1 & -0.5 & 1 \\ 0.2 & 1 & 0.5 \\ 1 & 0 & 0.3 \\ 0 & 1 & -0.2 \end{bmatrix}$$

$$\frac{\partial S_0}{\partial F_1} = 0.5 + 1 - 0.04 + 1 + 0 = 2.46$$
$$\frac{\partial S_0}{\partial F_2} = 1 - 0.25 - 0.2 + 0 + 0.1 = 0.65$$
$$\frac{\partial S_0}{\partial F_3} = 0.2 + 0.5 - 0.1 + 0.3 - 0.02 = 0.88$$

**Grad-CAM computation:** Apply these gradients to the original feature maps as in Problem 3.

---

## Problem 5: Before vs After ReLU as Heatmaps

**Before ReLU:**
$$M_c^{\text{pre-ReLU}} = \begin{bmatrix} 4 & 4 \\ 5 & 9 \end{bmatrix}$$

**After ReLU:**
$$M_c^{\text{post-ReLU}} = \max(0, M_c) = \begin{bmatrix} 4 & 4 \\ 5 & 9 \end{bmatrix}$$

(Same in this case, but if we had negative values, they'd be zeroed)

**Visualization:**

```
Before ReLU (Heatmap):
┌─────────────┐
│ 4    4  │   │
│ 5    9  │   │
└─────────────┘
Color intensity: 4 (dark), 9 (bright)

After ReLU (Heatmap):
┌─────────────┐
│ 4    4  │   │
│ 5    9  │   │
└─────────────┘
Same as above (no negative values to suppress)
```

**Example with negative values:**

If $M_c = \begin{bmatrix} -2 & 3 \\ 1 & 5 \end{bmatrix}$

Before ReLU: min=-2, max=5, normalized to $\begin{bmatrix} 0 & 0.714 \\ 0.429 & 1 \end{bmatrix}$

After ReLU: $\begin{bmatrix} 0 & 3 \\ 1 & 5 \end{bmatrix}$, min=0, max=5, normalized to $\begin{bmatrix} 0 & 0.6 \\ 0.2 & 1 \end{bmatrix}$

ReLU suppresses negative contributions and enhances positive ones.

---

## Problem 6: Without ReLU (Full Attribution)

**Full Attribution Method:**
$$L_c^{\text{full}} = \sum_{k} \frac{\partial S_c}{\partial f_k} \cdot f_k$$

Without applying ReLU after computing the gradient-weighted activations:

Using our original example:
$$L_c^{\text{full}} = \begin{bmatrix} 4 & 4 \\ 5 & 9 \end{bmatrix}$$

(Same as before since no negative values)

**Example with true negative values:**

Suppose the gradients were:
$$\frac{\partial S_c}{\partial f_1} = 1, \quad \frac{\partial S_c}{\partial f_2} = -2, \quad \frac{\partial S_c}{\partial f_3} = 0.5$$

$$L_c^{\text{full}} = 1 \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} - 2 \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} + 0.5 \begin{bmatrix} 2 & 1 \\ 0 & 1 \end{bmatrix}$$

$$L_c^{\text{full}}(1,1) = 1 - 0 + 1 = 2$$
$$L_c^{\text{full}}(1,2) = 2 - 2 + 0.5 = 0.5$$
$$L_c^{\text{full}}(2,1) = 3 - 2 + 0 = 1$$
$$L_c^{\text{full}}(2,2) = 4 - 0 + 0.5 = 4.5$$

$$L_c^{\text{full}} = \begin{bmatrix} 2 & 0.5 \\ 1 & 4.5 \end{bmatrix}$$

**Comparison:**

| Aspect | With ReLU | Without ReLU |
|--------|-----------|-------------|
| Negative values | Suppressed to 0 | Preserved |
| Interpretation | Only positive contributions | Both positive & negative |
| Sparsity | Higher | Lower |
| Physical meaning | Excitation only | Full attribution |
| Visualization | Cleaner, highlights | More detailed, noise-prone |

**Heatmap Difference:**

```
WITH ReLU:
┌───────┐
│ + + │
│ + +++│
└───────┘
(Bright regions only)

WITHOUT ReLU:
┌───────┐
│ + +  │  (both + and - contributions)
│ - ++│
└───────┘
(Mixed bright and dark regions)
```

---

## Problem 7: Integrated Gradients for F(x₁, x₂) = x₁² + 2x₂

**Given:**
- Model: $F(x_1, x_2) = x_1^2 + 2x_2$
- Input: $x = (2, 3)$
- Baseline: $x' = (0, 0)$
- Steps: $m = 10$

**Integrated Gradients Formula:**
$$IG_i(x) = (x_i - x'_i) \times \int_0^1 \frac{\partial F(x' + \alpha(x - x'))}{\partial x_i} d\alpha$$

**Discrete Approximation:**
$$IG_i(x) \approx (x_i - x'_i) \times \sum_{k=1}^{m} \frac{\partial F(x' + \frac{k}{m}(x - x'))}{\partial x_i} \times \frac{1}{m}$$

**Step 1: Compute gradients analytically**

$$\frac{\partial F}{\partial x_1} = 2x_1$$
$$\frac{\partial F}{\partial x_2} = 2$$

**Step 2: Generate interpolation points**

For $k = 0, 1, 2, ..., m$:
$$\alpha_k = \frac{k}{m}$$

$$x^{(k)} = x' + \alpha_k(x - x') = (0,0) + \frac{k}{10}(2,3) = (\frac{2k}{10}, \frac{3k}{10})$$

**Step 3: Compute gradients at each interpolation point**

| k | α_k | x₁^(k) | x₂^(k) | ∂F/∂x₁ | ∂F/∂x₂ |
|---|-----|--------|--------|--------|--------|
| 0 | 0.0 | 0.0 | 0.0 | 0 | 2 |
| 1 | 0.1 | 0.2 | 0.3 | 0.4 | 2 |
| 2 | 0.2 | 0.4 | 0.6 | 0.8 | 2 |
| 3 | 0.3 | 0.6 | 0.9 | 1.2 | 2 |
| 4 | 0.4 | 0.8 | 1.2 | 1.6 | 2 |
| 5 | 0.5 | 1.0 | 1.5 | 2.0 | 2 |
| 6 | 0.6 | 1.2 | 1.8 | 2.4 | 2 |
| 7 | 0.7 | 1.4 | 2.1 | 2.8 | 2 |
| 8 | 0.8 | 1.6 | 2.4 | 3.2 | 2 |
| 9 | 0.9 | 1.8 | 2.7 | 3.6 | 2 |
| 10 | 1.0 | 2.0 | 3.0 | 4.0 | 2 |

**Step 4: Sum gradients**

$$\sum_{k=0}^{10} \frac{\partial F}{\partial x_1} = 0 + 0.4 + 0.8 + 1.2 + 1.6 + 2.0 + 2.4 + 2.8 + 3.2 + 3.6 + 4.0 = 22.0$$

$$\sum_{k=0}^{10} \frac{\partial F}{\partial x_2} = 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 = 22$$

**Step 5: Compute Integrated Gradients**

$$IG_1(x) = (2 - 0) \times \frac{1}{10} \times 22.0 = 2 \times 2.2 = 4.4$$

$$IG_2(x) = (3 - 0) \times \frac{1}{10} \times 22 = 3 \times 2.2 = 6.6$$

**Final Answer:**
$$IG(x) = [4.4, 6.6]$$

**Verification with exact integral:**

$$IG_1(x) = 2 \int_0^1 2\alpha(2) d\alpha = 4 \int_0^1 \alpha d\alpha = 4 \times \frac{1}{2} = 4$$

$$IG_2(x) = 3 \int_0^1 2 d\alpha = 3 \times 2 \times 1 = 6$$

(The discrete approximation gives 4.4 and 6.6, close to exact values 4 and 6)

---

## Problem 8: Custom IG Implementation

**Set 1:**
- Model: $F_1(x_1, x_2) = 3x_1^2 + x_1 x_2 + 2$
- Input: $x = (1, 2)$
- Baseline: $x' = (0, 0)$
- Steps: $m = 5$

**Gradients:**
$$\frac{\partial F_1}{\partial x_1} = 6x_1 + x_2$$
$$\frac{\partial F_1}{\partial x_2} = x_1$$

**Interpolation points:** $x^{(k)} = (\frac{k}{5}, \frac{2k}{5})$ for k=0 to 5

| k | x₁ | x₂ | ∂F/∂x₁ | ∂F/∂x₂ |
|---|----|----|--------|--------|
| 0 | 0 | 0 | 0 | 0 |
| 1 | 0.2 | 0.4 | 1.6 | 0.2 |
| 2 | 0.4 | 0.8 | 3.2 | 0.4 |
| 3 | 0.6 | 1.2 | 4.8 | 0.6 |
| 4 | 0.8 | 1.6 | 6.4 | 0.8 |
| 5 | 1 | 2 | 8 | 1 |

$$\sum \frac{\partial F_1}{\partial x_1} = 0 + 1.6 + 3.2 + 4.8 + 6.4 + 8 = 24.0$$
$$\sum \frac{\partial F_1}{\partial x_2} = 0 + 0.2 + 0.4 + 0.6 + 0.8 + 1 = 3.0$$

$$IG_1 = 1 \times \frac{1}{5} \times 24 = 4.8$$
$$IG_2 = 2 \times \frac{1}{5} \times 3 = 1.2$$

**Set 2:**
- Model: $F_2(x_1, x_2) = e^{x_1} + x_2^2$
- Input: $x = (0.5, 1)$
- Baseline: $x' = (0, 0)$
- Steps: $m = 4$

**Gradients:**
$$\frac{\partial F_2}{\partial x_1} = e^{x_1}$$
$$\frac{\partial F_2}{\partial x_2} = 2x_2$$

**Interpolation:** $x^{(k)} = (0.125k, 0.25k)$

| k | x₁ | x₂ | ∂F/∂x₁ | ∂F/∂x₂ |
|---|----|----|--------|--------|
| 0 | 0 | 0 | 1 | 0 |
| 1 | 0.125 | 0.25 | 1.133 | 0.5 |
| 2 | 0.25 | 0.5 | 1.284 | 1.0 |
| 3 | 0.375 | 0.75 | 1.455 | 1.5 |
| 4 | 0.5 | 1 | 1.649 | 2.0 |

$$\sum \frac{\partial F_2}{\partial x_1} = 1 + 1.133 + 1.284 + 1.455 + 1.649 = 6.521$$
$$\sum \frac{\partial F_2}{\partial x_2} = 0 + 0.5 + 1.0 + 1.5 + 2.0 = 5.0$$

$$IG_1 = 0.5 \times \frac{1}{4} \times 6.521 = 0.815$$
$$IG_2 = 1 \times \frac{1}{4} \times 5.0 = 1.25$$

**Set 3:**
- Model: $F_3(x_1, x_2) = \sin(x_1) + x_1 x_2$
- Input: $x = (\pi/4, 2)$
- Baseline: $x' = (0, 0)$
- Steps: $m = 8$

**Gradients:**
$$\frac{\partial F_3}{\partial x_1} = \cos(x_1) + x_2$$
$$\frac{\partial F_3}{\partial x_2} = x_1$$

**Interpolation:** $x^{(k)} = (\pi k/32, 2k/8) = (\pi k/32, k/4)$

(Computing numerically with $\pi/4 \approx 0.785$)

| k | x₁ | x₂ | cos(x₁) | ∂F/∂x₁ | ∂F/∂x₂ |
|---|----|----|---------|--------|--------|
| 0 | 0 | 0 | 1 | 1 | 0 |
| 1 | 0.098 | 0.25 | 0.995 | 1.245 | 0.098 |
| 2 | 0.196 | 0.5 | 0.981 | 1.481 | 0.196 |
| 3 | 0.294 | 0.75 | 0.957 | 1.707 | 0.294 |
| 4 | 0.393 | 1 | 0.923 | 1.923 | 0.393 |
| 5 | 0.491 | 1.25 | 0.879 | 2.129 | 0.491 |
| 6 | 0.589 | 1.5 | 0.826 | 2.326 | 0.589 |
| 7 | 0.687 | 1.75 | 0.764 | 2.514 | 0.687 |
| 8 | 0.785 | 2 | 0.707 | 2.707 | 0.785 |

$$\sum \frac{\partial F_3}{\partial x_1} \approx 17.032$$
$$\sum \frac{\partial F_3}{\partial x_2} \approx 3.533$$

$$IG_1 = 0.785 \times \frac{1}{8} \times 17.032 \approx 1.674$$
$$IG_2 = 2 \times \frac{1}{8} \times 3.533 \approx 0.883$$

---

## Problem 9: Grad-CAM vs Integrated Gradients Comparison

### Detailed Comparison

| Aspect | Grad-CAM | Integrated Gradients |
|--------|----------|---------------------|
| **Type** | Local attribution | Path-integrated attribution |
| **Formula** | $L_c^{(k)} = \sum_{i,j} \frac{\partial S_c}{\partial A_{ij}^{(k)}} A_{ij}^{(k)}$ | $IG_i = (x_i - x'_i) \int_0^1 \frac{\partial F}{\partial x_i}(\alpha x + (1-\alpha) x') d\alpha$ |
| **Baseline** | Not required | Required |
| **Computational cost** | Low (one backward pass) | High (m forward/backward passes) |
| **Applicable to** | CNNs (spatial maps) | Any DNN |
| **Output** | Spatial heatmap | Feature attribution scores |
| **Sensitivity to perturbations** | Lower | Higher |
| **Linearity axiom** | Not satisfied | Satisfied |
| **Completeness axiom** | Not satisfied | Satisfied |
| **ReLU requirement** | Recommended | Not required |

### Detailed Analysis using Problem 1 Example

**Given network:**
- 5×5 input image
- Feature maps: $f_1, f_2, f_3$ (2×2)
- Weights: $w^c = [2, -1, 1]$
- Class score: $S_c = 5.5$

**GRAD-CAM Analysis:**

1. **Gradient computation:**
   - $\frac{\partial S_c}{\partial f_1} = 2$
   - $\frac{\partial S_c}{\partial f_2} = -1$
   - $\frac{\partial S_c}{\partial f_3} = 1$

2. **Importance weighting by activation:**
   $$\text{Importance}(f_k) = \frac{\partial S_c}{\partial f_k} \times \text{mean}(f_k)$$
   
   - $f_1$: importance = $2 \times 2.5 = 5.0$
   - $f_2$: importance = $-1 \times 0.5 = -0.5$
   - $f_3$: importance = $1 \times 1.0 = 1.0$

3. **Localization quality:** Very good for spatial tasks

4. **Result:** 2×2 spatial map showing which regions matter

**INTEGRATED GRADIENTS Analysis:**

1. **Path definition:** From baseline $(0,0,0)$ to actual $(F_1, F_2, F_3)$

2. **Parametrized path:**
   $$F(\alpha) = \alpha F + (1-\alpha) F' = (\alpha F_1, \alpha F_2, \alpha F_3)$$

3. **Gradient along path:**
   $$\frac{\partial S_c}{\partial F_1(\alpha)} = 2 \quad \text{(constant)}$$
   $$\frac{\partial S_c}{\partial F_2(\alpha)} = -1 \quad \text{(constant)}$$
   $$\frac{\partial S_c}{\partial F_3(\alpha)} = 1 \quad \text{(constant)}$$

4. **Integration:**
   $$IG_1 = (F_1 - 0) \int_0^1 2 \, d\alpha = 2.5 \times 2 = 5.0$$
   $$IG_2 = (F_2 - 0) \int_0^1 (-1) \, d\alpha = 0.5 \times (-1) = -0.5$$
   $$IG_3 = (F_3 - 0) \int_0^1 1 \, d\alpha = 1.0 \times 1 = 1.0$$

5. **Result:** Attribution scores per feature channel

### Verification Axioms

**1. Sensitivity Axiom:**
- If $f(x) \neq f(x')$, attributions should be non-zero
- IG: ✓ Satisfied
- Grad-CAM: ✗ Not guaranteed

**2. Implementation Invariance:**
- Mathematically equivalent networks should give same attributions
- IG: ✓ Satisfied
- Grad-CAM: ✗ Depends on architecture

**3. Linearity:**
- For $F(x) = aG(x) + bH(x)$:
  - $IG_F = a \cdot IG_G + b \cdot IG_H$
- IG: ✓ Satisfied
- Grad-CAM: ✗ Not satisfied

**4. Completeness:**
- $\sum_i IG_i(x) = F(x) - F(x')$
- IG: ✓ Satisfied
  - $5.0 + (-0.5) + 1.0 = 5.5 = S_c - S_c(0,0,0)$ ✓
- Grad-CAM: ✗ Not satisfied

### Practical Comparison

```
SCENARIO: Image Classification

INPUT SPACE:
┌───────────────┐
│   Original    │
│   Image (5×5) │
└───────────────┘

GRAD-CAM OUTPUT:
┌──────────────┐
│  2×2 Spatial │
│   Heatmap    │
│  (Where to   │
│   look?)     │
└──────────────┘

IG OUTPUT:
┌──────────────┐
│  3 Feature   │
│  Attribution │
│  Scores      │
│  (How much   │
│  each path   │
│  matters?)   │
└──────────────┘
```

### Use Case Recommendations

**Use Grad-CAM when:**
- ✓ Visualizing spatial importance in images
- ✓ Quick explanations needed (computational efficiency)
- ✓ Understanding which image regions influence predictions
- ✓ Working with CNNs with spatial structure
- Example: Medical imaging, object detection

**Use Integrated Gradients when:**
- ✓ Need theoretically sound attribution
- ✓ Working with any network architecture
- ✓ Comparing attributions across different models
- ✓ Full feature-level attribution required
- ✓ Axioms compliance is important
- Example: Feature importance analysis, model auditing

### Example Scenario: Adversarial Robustness

Suppose an adversarial perturbation changes prediction from Class A to Class B.

**Grad-CAM perspective:**
- Shows which spatial regions triggered the change
- Good for understanding local patterns exploited

**IG perspective:**
- Assigns credit to each feature along the adversarial path
- Better for understanding which features are vulnerable
- Can compare different attack paths using different baselines

### Mathematical Consistency

**For linear models**, both methods agree:

$$F(x) = w \cdot x + b$$

- Grad-CAM: $\frac{\partial F}{\partial x_i} = w_i$
- IG: $IG_i = (x_i - x'_i) \cdot w_i$

**For non-linear models**, they diverge significantly due to:
- Grad-CAM: Uses only final gradients
- IG: Integrates gradients along the path

### Hybrid Approach

**Grad-CAM + Integrated Gradients:**

1. Use Grad-CAM to identify important regions spatially
2. Use IG on those regions' features for detailed attribution
3. Combine spatial and feature-level insights

**Implementation:**
```python
# Pseudocode
spatial_importance = GradCAM(model, input, class_idx)  # 2×2 map
important_regions = extract_regions(input, spatial_importance)

for each_region in important_regions:
    attribution = IntegratedGradients(model, region, baseline)
    # Provides feature-level explanation within regions
```

### Summary Table

| Task | Grad-CAM | IG |
|------|----------|-----|
| Visualizing image regions | **★★★★★** | ★★☆☆☆ |
| Feature attribution | ★★★☆☆ | **★★★★★** |
| Theoretical soundness | ★★★☆☆ | **★★★★★** |
| Computational speed | **★★★★★** | ★★☆☆☆ |
| Interpretability | **★★★★☆** | ★★★☆☆ |
| Multiple architectures | ★★★☆☆ | **★★★★★** |
| Robustness to noise | ★★★☆☆ | **★★★★☆** |

---

## Conclusion

These nine problems comprehensively cover:

1. **CAM fundamentals** - Understanding spatial importance
2. **Grad-CAM** - Gradient-based spatial attribution
3. **Multiple FC layers** - Complex architecture handling
4. **ReLU effects** - Impact on attribution
5. **Full attribution** - Without suppression
6. **Integrated Gradients** - Path-integrated attribution
7. **IG implementation** - Practical discrete computation
8. **Custom implementations** - Various models and settings
9. **Method comparison** - Theoretical and practical differences

Each method has distinct advantages: Grad-CAM excels at spatial interpretation while IG provides theoretically sound feature attribution across any architecture.