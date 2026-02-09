"""
Quick test to verify the setup works correctly
Run this after download_models.py to check everything is ready
"""

import sys
from pathlib import Path

print("=" * 60)
print("SETUP VERIFICATION")
print("=" * 60)

# Check directories
INPUT_DIR = Path(__file__).parent / "input"
MODELS_DIR = Path(__file__).parent / "models"

errors = []

# Check input directories
print("\n[1] Checking directories...")
dirs_to_check = [
    INPUT_DIR / "images",
    INPUT_DIR / "labels",
    MODELS_DIR,
]
for d in dirs_to_check:
    if d.exists():
        print(f"  OK: {d}")
    else:
        print(f"  MISSING: {d}")
        errors.append(f"Directory missing: {d}")

# Check for models
print("\n[2] Checking models...")
model_files = list(MODELS_DIR.glob("*.pth"))
if model_files:
    for m in model_files:
        print(f"  FOUND: {m.name}")
else:
    print("  NO MODELS FOUND!")
    errors.append("No model files in models/ folder")

# Check for input images
print("\n[3] Checking input images...")
images_dir = INPUT_DIR / "images"
if images_dir.exists():
    images = list(images_dir.glob("*.[jJ][pP][gG]")) + \
             list(images_dir.glob("*.[pP][nN][gG]")) + \
             list(images_dir.glob("*.[jJ][pP][eE][gG]"))
    if images:
        print(f"  FOUND: {len(images)} images")
    else:
        print("  NO IMAGES FOUND!")
        print("  Please add images to input/images/")
        errors.append("No input images")
else:
    errors.append("input/images/ directory missing")

# Check PyTorch
print("\n[4] Checking PyTorch...")
try:
    import torch
    print(f"  PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"  CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("  CUDA: Not available (will use CPU - slower)")
except ImportError:
    print("  PYTORCH NOT INSTALLED!")
    errors.append("PyTorch not installed")

# Check other dependencies
print("\n[5] Checking dependencies...")
deps = ['PIL', 'numpy', 'tqdm', 'torchvision']
for dep in deps:
    try:
        __import__(dep if dep != 'PIL' else 'PIL.Image')
        print(f"  OK: {dep}")
    except ImportError:
        print(f"  MISSING: {dep}")
        errors.append(f"Missing dependency: {dep}")

# Summary
print("\n" + "=" * 60)
if errors:
    print("ISSUES FOUND:")
    for e in errors:
        print(f"  - {e}")
    print("\nFix these issues before running generate.py")
else:
    print("ALL CHECKS PASSED!")
    print("\nYou're ready to generate images:")
    print("  python generate.py --all")
print("=" * 60)
