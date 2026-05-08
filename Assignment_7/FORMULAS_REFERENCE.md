# Quick Reference: Mathematical Formulas

## CAM (Class Activation Mapping)

### Basic Formula
```
CAM_c(x,y) = Σ_k w_k^c · f_k(x,y)
```

### Where:
- `c` = class index
- `k` = feature map index
- `w_k^c` = weight of feature map k for class c (from FC layer)
- `f_k(x,y)` = activation of feature map k at spatial location (x,y)
- `Σ_k` = sum over all K feature maps

### Implementation
```python
# Get FC weights for target class
fc_weights = model.fc.weight[class_idx]

# Weighted sum of feature maps
cam = torch.zeros(features.shape[-2:])
for k, weight in enumerate(fc_weights):
    cam += weight * features[k]

# Apply ReLU
cam = F.relu(cam)
```

---

## Grad-CAM (Gradient-weighted CAM)

### Basic Formula
```
Grad-CAM_c(x,y) = ReLU(Σ_k w_k^c · A_k(x,y))
```

### Weight Calculation
```
w_k^c = (1/Z) · Σ_i Σ_j (∂y^c / ∂A_ij^k)
```

### Where:
- `y^c` = class score (output before softmax)
- `A_k` = feature map k
- `∂y^c / ∂A_ij^k` = gradient of class score w.r.t. feature at position (i,j)
- `Z` = spatial dimensions (width × height)
- `ReLU()` = keeps only positive values

### Implementation
```python
# Forward pass
output = model(input)

# Backward pass for target class
target = output[0, class_idx]
target.backward()

# Get gradients and activations
gradients = activations.grad  # (C, H, W)
features = activations       # (C, H, W)

# Compute weights
weights = gradients.mean(dim=(1, 2))  # Average over spatial dimensions

# Weighted sum
grad_cam = torch.zeros(features.shape[-2:])
for k in range(len(weights)):
    grad_cam += weights[k] * features[k]

# Apply ReLU
grad_cam = F.relu(grad_cam)
```

---

## Integrated Gradients (IG)

### Basic Formula
```
IG_i(x) = (x_i - baseline_i) · ∫₀¹ (∂F(baseline + α(x - baseline)) / ∂x_i) dα
```

### Discrete Approximation
```
IG_i(x) ≈ (x_i - baseline_i) · (1/m) · Σ_{k=1}^m (∂F(x_k^α) / ∂x_i)
```

### Where:
- `x` = input image
- `baseline` = reference input (usually black/zero image)
- `α` = interpolation coefficient (0 to 1)
- `x_k^α` = baseline + (k/m) · (x - baseline)
- `F()` = model prediction function
- `m` = number of integration steps

### Key Properties
- **Sensitivity**: IG is non-zero only when F(baseline) ≠ F(x)
- **Implementability**: IG(baseline) = 0 (attribution at baseline is zero)
- **Completeness**: Σ_i IG_i(x) = F(x) - F(baseline)
- **Linearity**: IG is linear w.r.t. model changes

### Implementation
```python
# Setup
alphas = torch.linspace(0, 1, steps + 1)
accumulated_grads = torch.zeros_like(input)
baseline = torch.zeros_like(input)

# Integrate
for alpha in alphas:
    # Interpolate between baseline and input
    interpolated = baseline + alpha * (input - baseline)
    interpolated.requires_grad = True
    
    # Forward pass
    output = model(interpolated)
    target = output[0, class_idx]
    
    # Backward pass
    target.backward()
    
    # Accumulate gradients
    accumulated_grads += interpolated.grad

# Average and multiply by input difference
avg_grads = accumulated_grads / (steps + 1)
integrated_gradients = (input - baseline) * avg_grads
```

---

## Normalization & Visualization

### Min-Max Normalization
```
heatmap_normalized = (heatmap - min(heatmap)) / (max(heatmap) - min(heatmap))
```

### Overlay Formula
```
output = α · image + (1-α) · colormap(heatmap)
```
Where α ∈ [0, 1] controls transparency

### Denormalization (for visualization)
```
image_denorm = image · std + mean
```

---

## Computational Complexity

### CAM
- **Time**: O(C × H × W) where C=channels, H×W=spatial dims
- **Space**: O(C × H × W)
- **FLOPs**: ~1M for 224×224 images

### Grad-CAM
- **Time**: O(B × C × H × W) where B=batch size
- **Space**: O(B × C × H × W)
- **FLOPs**: ~2-3M (includes backward pass)

### Integrated Gradients
- **Time**: O(steps × B × N) where N=parameters
- **Space**: O(B × N)
- **FLOPs**: ~steps × FLOPs(one forward+backward)
- **Example**: 50 steps = ~50× the cost of one prediction

---

## Hyperparameter Selection Guide

### Integrated Gradients Steps
```
steps = 50       # Balanced default
steps = 30       # Quick analysis
steps = 100+     # High precision research
steps = 200+     # Very high precision

Accuracy improvement diminishes after 50 steps
Time complexity grows linearly with steps
```

### Baseline Selection
```
zeros        # Standard: black image
random       # Test robustness
blurred      # Texture-based models
mean_pixels  # Dataset statistics
```

### Layer Selection for Grad-CAM
```
Layer 1: Low-level features (edges, colors)
Layer 2: Shapes and textures  
Layer 3: Object parts
Layer 4: Semantic objects and scenes

Later layers = higher-level semantics
Early layers = lower-level details
```

---

## Common Issues & Solutions

### CAM = 0 everywhere
```
✗ Problem: Wrong layer (not conv layer), incorrect weight indexing
✓ Solution: Verify layer type, debug weight extraction
```

### Grad-CAM noisy/unclear
```
✗ Problem: Gradients not flowing, wrong target layer
✓ Solution: Check requires_grad=True, use deeper layers
```

### IG extremely slow
```
✗ Problem: Too many steps, large image, GPU not used
✓ Solution: Reduce to 30 steps, resize images, use CUDA
```

### Heatmap all same color
```
✗ Problem: Normalization issue, empty gradient
✓ Solution: Check min-max scaling, verify model output range
```

---

## Validation Metrics

### Insertion/Deletion Score
```
1. Sort pixels by importance
2. Iteratively insert/delete top pixels
3. Plot impact on model confidence
4. Steeper slope = better explanation
```

### Perturbation Analysis
```
1. Occlude important regions
2. Check if model confidence drops
3. Confidence drop should be significant
4. Inverse: Random occlusion should have less effect
```

### Correlation with Saliency
```
correlation(explanation_1, explanation_2) 
# Compare different methods
# High correlation = robust explanations
```

---

## References

**CAM**: Zhou et al., CVPR 2016
- Paper: https://arxiv.org/abs/1512.04150
- Key insight: Uses global average pooling to weight feature maps

**Grad-CAM**: Selvaraju et al., ICCV 2017  
- Paper: https://arxiv.org/abs/1610.02055
- Key insight: Gradient-based weighting makes CAM general

**Integrated Gradients**: Sundararajan et al., ICML 2017
- Paper: https://arxiv.org/abs/1703.01365
- Key insight: Axioms (Sensitivity, Implementability) guarantee sound attribution

**XAI Survey**: Montavon et al., Digital Signal Processing 2019
- Comprehensive overview of interpretation methods
- Compares CAM, Grad-CAM, IG, LRP, SHAP, and more

---

**Created**: 2026-05-05  
**Status**: Complete Reference Material ✅
