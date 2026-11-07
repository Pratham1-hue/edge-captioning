"""
Quick demo: run the INT8 ONNX engine on an image and print the caption.
Usage: py demo.py [path-to-image]   (defaults to sample.jpg)
"""
import time
import sys
from PIL import Image
from infer_onnx import EdgeCaptionEngine

image_path = sys.argv[1] if len(sys.argv) > 1 else "sample.jpg"

print(f"Loading INT8 ONNX engine (CPU)...")
engine = EdgeCaptionEngine()

image = Image.open(image_path)
print(f"Image: {image_path} ({image.size[0]}x{image.size[1]})")

start = time.perf_counter()
caption = engine.generate_caption(image)
elapsed = time.perf_counter() - start

print("-" * 50)
print(f"Caption: {caption}")
print(f"Inference time: {elapsed*1000:.0f} ms  (greedy decode, CPU)")
print("-" * 50)
