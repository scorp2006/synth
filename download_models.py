"""
Download pretrained GAN models for weather synthesis
Fully automatic - just run: python download_models.py
"""

import os
import sys
from pathlib import Path
import urllib.request
import shutil

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# =============================================================================
# OFFICIAL CYCLEGAN MODELS (Berkeley - guaranteed to work)
# =============================================================================

MODELS_TO_DOWNLOAD = {
    # These create weather-like effects
    "summer2winter": {
        "url": "https://efrosgans.eecs.berkeley.edu/cyclegan/pretrained_models/summer2winter_yosemite_pretrained/latest_net_G_A.pth",
        "description": "Summer to Winter (creates fog/haze effect)"
    },
    "style_monet": {
        "url": "https://efrosgans.eecs.berkeley.edu/cyclegan/pretrained_models/style_monet_pretrained/latest_net_G.pth",
        "description": "Monet style (creates atmospheric/rain-like effect)"
    },
    "style_vangogh": {
        "url": "https://efrosgans.eecs.berkeley.edu/cyclegan/pretrained_models/style_vangogh_pretrained/latest_net_G.pth",
        "description": "Van Gogh style (used for mood/color shift)"
    },
}

# Weather effect mapping
WEATHER_MAPPING = {
    "fog_generator.pth": "summer2winter.pth",
    "rain_generator.pth": "style_monet.pth",
    "night_generator.pth": "style_vangogh.pth",
}


def download_file(url, output_path, description=""):
    """Download a file with progress indicator"""
    try:
        print(f"  Downloading: {description}")
        print(f"  URL: {url}")

        # Download with progress
        def progress_hook(count, block_size, total_size):
            if total_size > 0:
                percent = min(100, count * block_size * 100 // total_size)
                sys.stdout.write(f"\r  Progress: {percent}%")
                sys.stdout.flush()

        urllib.request.urlretrieve(url, output_path, progress_hook)
        print(f"\n  Saved: {output_path.name}")
        return True
    except Exception as e:
        print(f"\n  FAILED: {e}")
        return False


def main():
    print("=" * 60)
    print("WEATHER GAN MODEL DOWNLOADER")
    print("=" * 60)
    print(f"\nDownloading to: {MODELS_DIR}\n")

    # Download all models
    downloaded = 0
    failed = 0

    for name, info in MODELS_TO_DOWNLOAD.items():
        output_path = MODELS_DIR / f"{name}.pth"

        print(f"\n[{name}]")

        if output_path.exists():
            print(f"  SKIP: Already exists")
            downloaded += 1
            continue

        if download_file(info["url"], output_path, info["description"]):
            downloaded += 1
        else:
            failed += 1

    # Create weather-named copies for the generator
    print("\n" + "-" * 60)
    print("Creating weather model links...")

    for weather_name, source_name in WEATHER_MAPPING.items():
        source_path = MODELS_DIR / source_name
        target_path = MODELS_DIR / weather_name

        if target_path.exists():
            print(f"  {weather_name}: Already exists")
            continue

        if source_path.exists():
            shutil.copy(source_path, target_path)
            print(f"  {weather_name}: Created from {source_name}")
        else:
            print(f"  {weather_name}: Source not found!")

    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"\nModels downloaded: {downloaded}/{len(MODELS_TO_DOWNLOAD)}")

    if failed > 0:
        print(f"Failed: {failed}")
        print("\nSome models failed to download. Check your internet connection.")

    # List all models
    print("\nModels in folder:")
    for f in sorted(MODELS_DIR.glob("*.pth")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name} ({size_mb:.1f} MB)")

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("""
1. Add your images to: input/images/
2. Add YOLO labels to: input/labels/
3. Run generation:

   python generate.py --all

   Or with range (for team split):
   python generate.py --all --start 0 --end 125
""")


if __name__ == "__main__":
    main()
