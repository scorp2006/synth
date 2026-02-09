# Synthetic Weather Image Generator

Generate combined weather condition images using pretrained GANs for **Problem Statement 7**.

**Combined Conditions:**
- Rain + Low-light
- Fog + Night

**Severity Levels:** Light, Medium, Heavy

---

## Super Quick Start (For Team Members)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run complete setup (downloads dataset + models)
python setup.py

# 3. Run your assigned chunk
python generate.py --all --start YOUR_START --end YOUR_END
```

That's it! Everything is automated.

---

## Team Assignment

| Person | Start | End | Command |
|--------|-------|-----|---------|
| 1 | 0 | 125 | `python generate.py --all --start 0 --end 125` |
| 2 | 125 | 250 | `python generate.py --all --start 125 --end 250` |
| 3 | 250 | 375 | `python generate.py --all --start 250 --end 375` |
| 4 | 375 | 500 | `python generate.py --all --start 375 --end 500` |

Each person generates **750 images** (125 base × 6 variants).
**Total: 3,000 images**

---

## Project Structure

```
Abhinav Synth/
├── input/
│   ├── images/          # Auto-downloaded COCO images (500)
│   └── labels/          # Auto-generated YOLO labels
├── output/
│   ├── rain_lowlight/   # Generated rain+lowlight images
│   └── fog_night/       # Generated fog+night images
├── models/              # Pretrained GAN weights
├── metadata/            # Generated metadata files
├── setup.py             # ONE-CLICK SETUP (run this first!)
├── download_dataset.py  # Downloads COCO images
├── download_models.py   # Downloads GAN models
├── generate.py          # Main generation script
├── test_setup.py        # Verifies setup
├── requirements.txt
├── classes.txt
└── README.md
```

---

## Detailed Setup (If Needed)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install torch torchvision numpy opencv-python Pillow tqdm gdown pyyaml
```

### Step 2: Run Setup

**Option A: One command (recommended)**
```bash
python setup.py
```

**Option B: Step by step**
```bash
python download_dataset.py   # Downloads 500 COCO images
python download_models.py    # Downloads GAN models
python test_setup.py         # Verifies everything
```

### Step 3: Generate Images

```bash
# Generate all at once
python generate.py --all

# Or with your assigned range
python generate.py --all --start 0 --end 125
```

---

## What Gets Downloaded

### Dataset (~250MB download, 500 images)
- Source: COCO 2017
- Filtered for: person, car, traffic light
- Images with 2+ target classes prioritized

### GAN Models (~150MB)
- `summer2winter.pth` → Fog/haze effect
- `style_monet.pth` → Rain/atmospheric effect
- `style_vangogh.pth` → Color/mood shift

---

## Output Format

### Generated Images
- Location: `output/{condition}/images/`
- Naming: `{original}_{condition}_{severity}.jpg`
- Example: `000000001234_rain_lowlight_M.jpg`

### Labels (YOLO format)
- Location: `output/{condition}/labels/`
- Copied from input with matching names

### Metadata
- Location: `metadata/`
- JSON + TXT for each image

---

## Severity Levels

| Level | GAN Blend | Darkness | Code |
|-------|-----------|----------|------|
| Light | 35% | 15% | L |
| Medium | 55% | 35% | M |
| Heavy | 80% | 55% | H |

---

## Object Classes

From `classes.txt` (YOLO class IDs):
- 0 = person
- 1 = car
- 2 = traffic_light

---

## Expected Output

For **500 input images**:
- 2 conditions × 3 severities × 500 images = **3,000 images total**
- Split among 4 people = **750 images each**

---

## Troubleshooting

### "No images found"
```bash
python download_dataset.py  # Download dataset first
```

### "No models available"
```bash
python download_models.py  # Download models
```

### "CUDA out of memory"
```bash
python generate.py --all --device cpu  # Use CPU instead
```

### Slow generation
- GPU: ~5-10 sec/image
- CPU: ~30-60 sec/image

---

## Resources

- [COCO Dataset](https://cocodataset.org/)
- [Official CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- [Problem Statement PDF](./Synth%20Vision%20PS-7.pdf)

---

## Team

**Team Abhinav** - Synth Vision Hackathon 2025
