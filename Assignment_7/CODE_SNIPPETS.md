# Implementation Quick Reference

## Setup & Dependencies

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# GPU Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

---

## 1. CAM Implementation (Standalone)

```python
class CAM:
    """Class Activation Mapping"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.hook = None
        self._register_hook()
    
    def _register_hook(self):
        def hook_fn(module, input, output):
            self.activations = output.detach()
        self.hook = self.target_layer.register_forward_hook(hook_fn)
    
    def generate_cam(self, input_tensor, class_idx):
        """
        Generate CAM heatmap
        Args:
            input_tensor: (1, 3, H, W)
            class_idx: target class
        Returns:
            cam: (H, W) normalized to [0, 1]
        """
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Get FC layer weights
        fc_weights = self.model.fc.weight[class_idx].cpu()
        
        # Compute CAM
        cam = torch.zeros(self.activations.shape[-2:])
        activations = self.activations[0].cpu()  # (C, H, W)
        
        for i, weight in enumerate(fc_weights):
            cam += weight * activations[i]
        
        # Normalize
        cam = F.relu(cam).numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam

# Usage
model = resnet50(pretrained=True)
target_layer = model.layer4[-1].conv3
cam = CAM(model, target_layer)
heatmap = cam.generate_cam(image_tensor, class_idx)
```

---

## 2. Grad-CAM Implementation (Standalone)

```python
class GradCAM:
    """Gradient-weighted Class Activation Mapping"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.forward_hook = None
        self.backward_hook = None
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.forward_hook = self.target_layer.register_forward_hook(forward_hook)
        self.backward_hook = self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor, class_idx):
        """
        Generate Grad-CAM heatmap
        Args:
            input_tensor: (1, 3, H, W)
            class_idx: target class
        Returns:
            cam: (H, W) normalized to [0, 1]
        """
        self.model.eval()
        input_tensor.requires_grad = True
        
        # Forward
        output = self.model(input_tensor)
        
        # Backward
        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward()
        
        # Compute Grad-CAM
        gradients = self.gradients[0].cpu()  # (C, H, W)
        activations = self.activations[0].cpu()  # (C, H, W)
        
        # Average pooling over spatial dimensions
        weights = gradients.mean(dim=(1, 2))  # (C,)
        
        # Weighted sum
        cam = torch.zeros(activations.shape[-2:])
        for i, weight in enumerate(weights):
            cam += weight * activations[i]
        
        # Normalize
        cam = F.relu(cam).numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam

# Usage
model = resnet50(pretrained=True)
target_layer = model.layer4[-1].conv3
gradcam = GradCAM(model, target_layer)
heatmap = gradcam.generate_cam(image_tensor, class_idx)
```

---

## 3. Integrated Gradients Implementation (Standalone)

```python
class IntegratedGradients:
    """Integrated Gradients Attribution"""
    def __init__(self, model):
        self.model = model
    
    def generate_ig(self, input_tensor, class_idx, baseline=None, steps=50):
        """
        Compute Integrated Gradients attribution
        Args:
            input_tensor: (1, 3, H, W)
            class_idx: target class
            baseline: (1, 3, H, W) - usually zeros
            steps: number of interpolation steps
        Returns:
            attribution: (H, W) pixel importance map
        """
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)
        
        self.model.eval()
        
        # Create interpolation path
        alphas = torch.linspace(0, 1, steps + 1).to(device)
        accumulated_grads = torch.zeros_like(input_tensor)
        
        for alpha in alphas:
            # Interpolate
            interpolated = baseline + alpha * (input_tensor - baseline)
            interpolated.requires_grad = True
            
            # Forward
            output = self.model(interpolated)
            target = output[0, class_idx]
            
            # Backward
            self.model.zero_grad()
            target.backward()
            
            # Accumulate
            accumulated_grads += interpolated.grad.detach()
        
        # Average gradients
        avg_grads = accumulated_grads / (steps + 1)
        
        # IG = (input - baseline) * avg_gradients
        ig = (input_tensor - baseline) * avg_grads
        
        # Aggregate across channels and normalize
        attribution = ig.sum(dim=1, keepdim=True).abs()[0, 0].cpu().numpy()
        if attribution.max() > 0:
            attribution = attribution / attribution.max()
        
        return attribution

# Usage
model = resnet50(pretrained=True)
ig = IntegratedGradients(model)
attribution = ig.generate_ig(image_tensor, class_idx, steps=30)
```

---

## 4. Complete Workflow Example

```python
# 1. Load model and data
model = resnet50(pretrained=True)
model.eval()
model = model.to(device)

# 2. Load image
image = Image.open('path/to/image.jpg')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
image_tensor = transform(image).unsqueeze(0).to(device)

# 3. Get prediction
with torch.no_grad():
    output = model(image_tensor)
    pred_class = output.argmax(dim=1).item()

# 4. Initialize explainability methods
target_layer = model.layer4[-1].conv3
cam = CAM(model, target_layer)
gradcam = GradCAM(model, target_layer)
ig = IntegratedGradients(model)

# 5. Generate heatmaps
cam_heatmap = cam.generate_cam(image_tensor, pred_class)
gradcam_heatmap = gradcam.generate_cam(image_tensor, pred_class)
ig_attribution = ig.generate_ig(image_tensor, pred_class, steps=30)

# 6. Visualize
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# Original
image_np = np.array(image) / 255.0
axes[0].imshow(image_np)
axes[0].set_title('Original')

# CAM
axes[1].imshow(image_np)
axes[1].imshow(cam_heatmap, cmap='jet', alpha=0.5)
axes[1].set_title('CAM')

# Grad-CAM
axes[2].imshow(image_np)
axes[2].imshow(gradcam_heatmap, cmap='jet', alpha=0.5)
axes[2].set_title('Grad-CAM')

# IG
axes[3].imshow(image_np)
axes[3].imshow(ig_attribution, cmap='hot', alpha=0.5)
axes[3].set_title('Integrated Gradients')

plt.tight_layout()
plt.show()
```

---

## 5. Visualization Helper Functions

```python
def overlay_heatmap(image, heatmap, alpha=0.5, colormap='jet'):
    """Overlay heatmap on image"""
    h, w = image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    
    # Convert to uint8
    if image.max() <= 1:
        image = (image * 255).astype(np.uint8)
    
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8), 
        cv2.COLORMAP_JET
    )
    
    overlay = cv2.addWeighted(image, 1-alpha, heatmap_colored, alpha, 0)
    return overlay

def denormalize(tensor, mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225)):
    """Denormalize image tensor"""
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()
    
    mean = np.array(mean).reshape(3, 1, 1)
    std = np.array(std).reshape(3, 1, 1)
    
    image = tensor * std + mean
    image = np.clip(image, 0, 1)
    return image

def visualize_all_methods(image, cam_h, gradcam_h, ig_h, class_name):
    """4-panel visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original', fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(image)
    axes[0, 1].imshow(cam_h, cmap='jet', alpha=0.5)
    axes[0, 1].set_title('CAM', fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(image)
    axes[1, 0].imshow(gradcam_h, cmap='jet', alpha=0.5)
    axes[1, 0].set_title('Grad-CAM', fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(image)
    axes[1, 1].imshow(ig_h, cmap='hot', alpha=0.5)
    axes[1, 1].set_title('Integrated Gradients', fontweight='bold')
    axes[1, 1].axis('off')
    
    fig.suptitle(f'Predicted: {class_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
```

---

## 6. Performance Comparison

```python
import time

def benchmark_methods(model, image_tensor, class_idx, target_layer):
    """Compare computation times"""
    
    # CAM
    cam = CAM(model, target_layer)
    start = time.time()
    cam_result = cam.generate_cam(image_tensor, class_idx)
    cam_time = time.time() - start
    
    # Grad-CAM
    gradcam = GradCAM(model, target_layer)
    start = time.time()
    gradcam_result = gradcam.generate_cam(image_tensor, class_idx)
    gradcam_time = time.time() - start
    
    # Integrated Gradients
    ig = IntegratedGradients(model)
    start = time.time()
    ig_result = ig.generate_ig(image_tensor, class_idx, steps=30)
    ig_time = time.time() - start
    
    print("="*50)
    print("TIMING COMPARISON")
    print("="*50)
    print(f"CAM:                  {cam_time:.4f}s")
    print(f"Grad-CAM:             {gradcam_time:.4f}s")
    print(f"Integrated Gradients: {ig_time:.4f}s")
    print("="*50)
    
    relative_speeds = {
        'CAM': 1.0,
        'Grad-CAM': gradcam_time / cam_time,
        'IG (30 steps)': ig_time / cam_time
    }
    
    for method, speed in relative_speeds.items():
        print(f"{method:20s}: {speed:.2f}x")
```

---

## 7. Error Handling Template

```python
def safe_explainability(model, image_tensor, class_idx, 
                       method='gradcam', target_layer=None):
    """Robust method selection with error handling"""
    
    try:
        if target_layer is None:
            target_layer = model.layer4[-1].conv3
        
        if method.lower() == 'cam':
            explainer = CAM(model, target_layer)
            result = explainer.generate_cam(image_tensor, class_idx)
            
        elif method.lower() == 'gradcam':
            explainer = GradCAM(model, target_layer)
            result = explainer.generate_cam(image_tensor, class_idx)
            
        elif method.lower() == 'ig':
            explainer = IntegratedGradients(model)
            result = explainer.generate_ig(image_tensor, class_idx, steps=30)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return result
    
    except Exception as e:
        print(f"Error in explainability: {e}")
        return None
```

---

**Quick Copy-Paste Ready Code**: ✅  
**Last Updated**: 2026-05-05  
**Status**: Production Ready
