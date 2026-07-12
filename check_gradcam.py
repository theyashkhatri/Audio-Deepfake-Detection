import numpy as np
import tensorflow as tf
from src.models.custom_cnn import build_custom_cnn
from src.explainability import GradCAM, get_gradcam_for_model
from src.config import N_MELS, T_FRAMES

# Create dummy input
spec = np.random.rand(N_MELS, T_FRAMES, 1).astype(np.float32)

# Build untrained model
model = build_custom_cnn()

# Test GradCAM
gradcam = get_gradcam_for_model(model, "custom_cnn")
res = gradcam.explain(spec)
print(res.keys())
print("Grad-CAM shape:", res["heatmap"].shape)
print("Done.")
