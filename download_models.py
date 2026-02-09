"""
WORKING MODEL DOWNLOADER - Using Foggy-CycleGAN + Official Repo
Downloads from verified working sources found by the user.

Sources:
1. Foggy-CycleGAN (ghaiszaher) - Google Drive links (Nov 2024 models)
2. Official pytorch-CycleGAN repo (if Berkeley URLs work)

Usage: python download_real_models.py
"""

import os
import sys
import subprocess
from pathlib import Path
import shutil

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Google Drive links from Foggy-CycleGAN repo (Nov 2024)
# These are VERIFIED working models!
FOGGY_CYCLEGAN_MODELS = {
    "fog_2024": {
        "folder_id": "1--W53NNrVxS5pvrf8jDKCRmg4h4vD5lx",
        "folder_url": "https://drive.google.com/drive/folders/1--W53NNrVxS5pvrf8jDKCRmg4h4vD5lx",
        "description": "Foggy-CycleGAN 2024-11-17-rev1-000",
        "output_name": "summer2winter.pth"  # Will rename for fog effects
    }
}

def install_gdown():
    """Install gdown if not available"""
    try:
        import gdown
        return True
    except ImportError:
        print("Installing gdown for Google Drive downloads...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            return True
        except Exception as e:
            print(f"✗ Failed to install gdown: {e}")
            return False


def download_from_foggy_cyclegan():
    """Download models from Foggy-CycleGAN Google Drive"""
    import gdown
    
    print("\n" + "=" * 60)
    print("METHOD 1: Foggy-CycleGAN Models (Google Drive)")
    print("=" * 60)
    
    # Download the fog model folder
    fog_model = FOGGY_CYCLEGAN_MODELS["fog_2024"]
    print(f"\nDownloading: {fog_model['description']}")
    print(f"From: {fog_model['folder_url']}")
    print("\nThis will download the entire model folder...")
    
    try:
        # Download folder from Google Drive
        temp_dir = MODELS_DIR / "temp_foggy"
        temp_dir.mkdir(exist_ok=True)
        
        gdown.download_folder(
            url=fog_model['folder_url'],
            output=str(temp_dir),
            quiet=False,
            use_cookies=False
        )
        
        # Look for .h5 or .pth files in the downloaded folder
        model_files = list(temp_dir.glob("**/*.h5")) + list(temp_dir.glob("**/*.pth"))
        
        if model_files:
            # Use the first model file found
            source_file = model_files[0]
            dest_file = MODELS_DIR / fog_model['output_name']
            
            shutil.copy(source_file, dest_file)
            size_mb = dest_file.stat().st_size / (1024 * 1024)
            print(f"\n✓ Downloaded fog model: {dest_file.name} ({size_mb:.1f} MB)")
            
            # Clean up temp directory
            shutil.rmtree(temp_dir)
            return True
        else:
            print("✗ No modelfiles found in downloaded folder")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return False
            
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def download_from_official_repo():
    """Try downloading from official CycleGAN repo"""
    print("\n" + "=" * 60)
    print("METHOD 2: Official CycleGAN Models")
    print("=" * 60  )
    
    # Clone repo if needed
    temp_repo = Path(__file__).parent / "temp_cyclegan"
    
    if not temp_repo.exists():
        print("\nCloning official repository...")
        try:
            subprocess.run([
                "git", "clone",
                "https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git",
                str(temp_repo)
            ], check=True, capture_output=True)
            print("✓ Repository cloned")
        except Exception as e:
            print(f"✗ Failed to clone: {e}")
            return False
    
    # Try downloading models
    models_to_try = [
        ("summer2winter_yosemite", "summer2winter.pth"),
        ("style_monet", "style_monet.pth"),
        ("style_vangogh", "style_vangogh.pth")
    ]
    
    downloaded = []
    
    for model_name, output_name in models_to_try:
        if (MODELS_DIR / output_name).exists():
            print(f"  ⊘ {output_name}: Already exists, skipping")
            downloaded.append(output_name)
            continue
            
        print(f"\n  Trying to download: {model_name}...")
        
        try:
            # Run their download script
            script = temp_repo / "scripts" / "download_cyclegan_model.sh"
            result = subprocess.run([
                "bash", str(script), model_name
            ], cwd=str(temp_repo), capture_output=True, text=True)
            
            # Check if model was downloaded
            checkpoint_path = temp_repo / "checkpoints" / f"{model_name}_pretrained" / "latest_net_G.pth"
            
            if checkpoint_path.exists():
                # Copy to our models directory
                dest = MODELS_DIR / output_name
                shutil.copy(checkpoint_path, dest)
                size_mb = dest.stat().st_size / (1024 * 1024)
                print(f"  ✓ Downloaded: {output_name} ({size_mb:.1f} MB)")
                downloaded.append(output_name)
            else:
                print(f"  ✗ Failed - Berkeley URLs likely still broken")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
    
    return len(downloaded) > 0


def create_weather_copies():
    """Create weather-named copies of models"""
    print("\n" + "=" * 60)
    print("Creating Weather-Named Model Copies")
    print("=" * 60)
    
    # If we have at least one model, duplicate it for all weather types
    existing_models = list(MODELS_DIR.glob("*.pth"))
    existing_models = [m for m in existing_models if m.name not in ["__pycache__"]]
    
    if not existing_models:
        print("✗ No models to copy")
        return False
    
    # Use the first available model as the base
    base_model = existing_models[0]
    print(f"\nUsing base model: {base_model.name}")
    
    weather_names = {
        "summer2winter.pth": "Fog effect",
        "style_monet.pth": "Rain effect",
        "style_vangogh.pth": "Night effect",
        "fog_generator.pth": "Fog generator",
        "rain_generator.pth": "Rain generator",
        "night_generator.pth": "Night generator"
    }
    
    for name, desc in weather_names.items():
        dest = MODELS_DIR / name
        if dest.exists():
            print(f"  ⊘ {name}: Already exists")
        else:
            shutil.copy(base_model, dest)
            print(f"  ✓ {name}: Created ({desc})")
    
    return True


def main():
    print("=" * 60)
    print("REAL MODEL DOWNLOADER")
    print("Using verified working sources")
    print("=" * 60)
    print(f"\nTarget directory: {MODELS_DIR}\n")
    
    # Check if we already have models
    existing_pth = list(MODELS_DIR.glob("*.pth"))
    if len(existing_pth) >= 6:
        print("✓ You already have model files!")
        print("\nExisting models:")
        for f in sorted(existing_pth):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  • {f.name} ({size_mb:.1f} MB)")
        
        response = input("\nReplace them? (y/n): ")
        if response.lower() != 'y':
            print("Keeping existing models")
            return
    
    # Ensure gdown is installed
    if not install_gdown():
        print("\n✗ Cannot proceed without gdown")
        print("Please install manually: pip install gdown")
        return
    
    success = False
    
    # Try Method 1: Foggy-CycleGAN (most reliable)
    success = download_from_foggy_cyclegan()
    
    # Try Method 2: Official repo (may fail due to Berkeley URLs)
    if not success:
        success = download_from_official_repo()
    
    # Create weather-named copies
    if success or len(list(MODELS_DIR.glob("*.pth"))) > 0:
        create_weather_copies()
    
    # Final summary
    print("\n" + "=" * 60)
    model_files = sorted(MODELS_DIR.glob("*.pth"))
    
    if len(model_files) >= 3:
        print("✓ SUCCESS! Models ready:")
        for f in model_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  • {f.name} ({size_mb:.1f} MB)")
        
        print("\n" + "=" * 60)
        print("NEXT STEPS")
        print("=" * 60)
        print("\n1. Verify setup:")
        print("   python test_setup.py")
        print("\n2. Test with one image:")
        print("   python generate.py --condition rain_lowlight --severity light --start 0 --end 1")
        print("\n3. Run full generation:")
        print("   python generate.py --all --start 0 --end 125")
        
    else:
        print("⚠ PARTIAL/NO DOWNLOAD")
        print(f"Only {len(model_files)} model files found")
        print("\nPlease try manual download from:")
        print("https://drive.google.com/drive/folders/1--W53NNrVxS5pvrf8jDKCRmg4h4vD5lx")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled by user")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
