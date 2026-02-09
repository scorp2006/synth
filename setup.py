"""
MASTER SETUP SCRIPT
Run this once to download everything (dataset + models)

Usage: python setup.py
"""

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent


def run_script(script_name):
    """Run a Python script"""
    script_path = BASE_DIR / script_name
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print('='*60)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(BASE_DIR)
    )
    return result.returncode == 0


def main():
    print("=" * 60)
    print("SYNTH WEATHER - COMPLETE SETUP")
    print("=" * 60)
    print("""
This will:
1. Download COCO dataset (500 images with person, car, traffic light)
2. Download pretrained GAN models
3. Verify setup

Total download: ~400-500 MB
Time: 5-15 minutes depending on internet speed
""")

    input("Press ENTER to start setup (or Ctrl+C to cancel)...")

    # Step 1: Download dataset
    print("\n" + "#" * 60)
    print("# STEP 1: DOWNLOADING DATASET")
    print("#" * 60)
    if not run_script("download_dataset.py"):
        print("WARNING: Dataset download may have had issues")

    # Step 2: Download models
    print("\n" + "#" * 60)
    print("# STEP 2: DOWNLOADING GAN MODELS")
    print("# USING UPDATED DOWNLOADER (Google Drive / Official Repo)")
    print("#" * 60)
    if not run_script("download_models.py"):
        print("WARNING: Model download may have had issues")

    # Step 3: Verify setup
    print("\n" + "#" * 60)
    print("# STEP 3: VERIFYING SETUP")
    print("#" * 60)
    run_script("test_setup.py")

    # Final summary
    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    print("""
Everything is ready! Now run your assigned chunk:

  Person 1: python generate.py --all --start 0 --end 125
  Person 2: python generate.py --all --start 125 --end 250
  Person 3: python generate.py --all --start 250 --end 375
  Person 4: python generate.py --all --start 375 --end 500

Or generate everything at once:
  python generate.py --all
""")


if __name__ == "__main__":
    main()
