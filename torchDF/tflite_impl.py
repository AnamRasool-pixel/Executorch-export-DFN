import torch
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf


onnx_model_path = "/mnt/d/puretorch/DeepFilterNet/torchDF/denoiser_model.onnx"

# Load the ONNX model
onnx_model = onnx.load(onnx_model_path)

# Convert ONNX model to TensorFlow model
tf_rep = prepare(onnx_model)
tf_model_path = "denoiser_model.pb"
tf_rep.export_graph(tf_model_path)

# Convert TensorFlow model to TFLite model
converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
tflite_model = converter.convert()

# Save the TFLite model
tflite_model_path = "/mnt/d/puretorch/DeepFilterNet/torchDF/denoiser_model.tflite"
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"Model conversion complete. TFLite model saved at {tflite_model_path}")
