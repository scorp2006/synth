# Synth Weather Generator - TEAM SETUP 🚀

## Quick Start (UPDATED)

### 1. Pull Code
```bash
git pull origin main
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Models (~280MB)
**Run this script:**
```bash
python download_models.py
```
(Takes 2-5 mins. If fails, check `MANUAL_DOWNLOAD.md`)

### 4. Verify Setup
```bash
python test_setup.py
```
Should say: `ALL CHECKS PASSED!`

### 5. Generate Your Chunk
```bash
# Person 1 (0-125):
python generate.py --all --start 0 --end 125

# Person 2 (125-250):
python generate.py --all --start 125 --end 250

# Person 3 (250-375):
python generate.py --all --start 250 --end 375

# Person 4 (375-500):
python generate.py --all --start 375 --end 500
```

---

## What Changed?
- Fixed model download (Berkeley URLs were broken)
- Added `download_real_models.py` (uses Google Drive)
- Fixed "Tensor size mismatch" error in `generate.py`
- **Output Quality:** We use a base GAN model (Foggy) + algorithmic Rain/Night effects for distinct results.

## Troubleshooting
- **"gdown not installed"**: Run `pip install gdown`
- **Output empty?**: Make sure you ran `python download_real_models.py` successfully.
- **Slow?**: Add `--device cpu` if you don't have GPU (warning: very slow).
