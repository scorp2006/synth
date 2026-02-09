"""
Weather Effect Generator using Pretrained GANs
Generates combined weather conditions: rain+lowlight, fog+night

Usage:
    python generate.py --condition rain_lowlight --severity medium
    python generate.py --condition fog_night --severity heavy
    python generate.py --all  # Generate all conditions and severities
"""

import os
import sys
import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime
import random

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageEnhance
from torchvision import transforms
from tqdm import tqdm

# Add models directory to path
sys.path.append(str(Path(__file__).parent))
from models.networks import Generator, load_generator


# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_DIR = Path(__file__).parent / "input"
OUTPUT_DIR = Path(__file__).parent / "output"
MODELS_DIR = Path(__file__).parent / "models"
METADATA_DIR = Path(__file__).parent / "metadata"

# Severity levels control effect intensity
SEVERITY_CONFIG = {
    "light": {
        "gan_blend": 0.35,
        "darkness": 0.15,  # For low-light effect
        "suffix": "L"
    },
    "medium": {
        "gan_blend": 0.55,
        "darkness": 0.35,
        "suffix": "M"
    },
    "heavy": {
        "gan_blend": 0.80,
        "darkness": 0.55,
        "suffix": "H"
    }
}

# Image transforms for GAN input
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# ============================================================================
# MODEL LOADING
# ============================================================================

class WeatherGenerator:
    """Handles loading and applying weather effect GANs"""

    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        if self.device == 'cpu':
            print("WARNING: Using CPU - generation will be slower")

        self.models = {}
        self._load_models()

    def _load_models(self):
        """Load all available pretrained models"""
        # Primary weather models
        model_files = {
            'rain': MODELS_DIR / 'rain_generator.pth',
            'fog': MODELS_DIR / 'fog_generator.pth',
            'night': MODELS_DIR / 'night_generator.pth'
        }

        # Fallback to official CycleGAN models if weather-specific not found
        fallback_models = {
            'rain': MODELS_DIR / 'style_monet.pth',
            'fog': MODELS_DIR / 'summer2winter.pth',
            'night': MODELS_DIR / 'style_vangogh.pth'
        }

        for name, path in model_files.items():
            # Try primary model first
            if path.exists():
                print(f"Loading {name} model (weather-specific)...")
                try:
                    self.models[name] = self._load_single_model(path)
                    print(f"  {name}: Loaded successfully")
                    continue
                except Exception as e:
                    print(f"  {name}: Failed - {e}")

            # Try fallback
            fallback_path = fallback_models.get(name)
            if fallback_path and fallback_path.exists():
                print(f"Loading {name} model (fallback: {fallback_path.name})...")
                try:
                    self.models[name] = self._load_single_model(fallback_path)
                    print(f"  {name}: Loaded fallback successfully")
                except Exception as e:
                    print(f"  {name}: Fallback also failed - {e}")
            else:
                print(f"  {name}: No model file found")

        if not self.models:
            print("\n" + "=" * 50)
            print("ERROR: No models loaded!")
            print("Run: python download_models.py")
            print("=" * 50)

    def _load_single_model(self, model_path):
        """Load a single generator model"""
        model = Generator(input_nc=3, output_nc=3, n_residual_blocks=9)

        state_dict = torch.load(model_path, map_location=self.device, weights_only=False)

        # Handle different checkpoint formats
        if isinstance(state_dict, dict):
            for key in ['model', 'state_dict', 'netG_A', 'netG', 'generator', 'G']:
                if key in state_dict:
                    state_dict = state_dict[key]
                    break

        # Remove 'module.' prefix if present (from DataParallel)
        if state_dict and list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}

        # Try to load state dict
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            # Try with strict=False if architecture differs slightly
            model.load_state_dict(state_dict, strict=False)

        model.to(self.device)
        model.eval()

        return model

    def apply_gan_effect(self, image_tensor, effect_name, blend=0.5):
        """
        Apply a GAN transformation to an image

        Args:
            image_tensor: Normalized tensor [1, 3, H, W]
            effect_name: 'rain', 'fog', or 'night'
            blend: Blend factor (0 = original, 1 = full effect)

        Returns:
            Processed image tensor
        """
        if effect_name not in self.models:
            return image_tensor

        with torch.no_grad():
            output = self.models[effect_name](image_tensor)
            result = (1 - blend) * image_tensor + blend * output
            result = torch.clamp(result, -1, 1)

        return result

    def apply_low_light(self, pil_image, darkness=0.3):
        """
        Apply low-light effect to PIL image

        Args:
            pil_image: PIL Image
            darkness: 0 = no change, 1 = very dark

        Returns:
            Darkened PIL Image
        """
        # Reduce brightness
        brightness_factor = 1.0 - (darkness * 0.7)  # Max 70% darker
        enhancer = ImageEnhance.Brightness(pil_image)
        img = enhancer.enhance(brightness_factor)

        # Reduce contrast slightly
        contrast_factor = 1.0 - (darkness * 0.2)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast_factor)

        # Add slight blue tint (night effect)
        img_array = np.array(img).astype(np.float32)
        img_array[:, :, 2] = np.clip(img_array[:, :, 2] * (1 + darkness * 0.15), 0, 255)  # Boost blue
        img_array[:, :, 0] = np.clip(img_array[:, :, 0] * (1 - darkness * 0.1), 0, 255)  # Reduce red

        return Image.fromarray(img_array.astype(np.uint8))

    def apply_combined_effect(self, pil_image, condition, severity='medium'):
        """
        Apply combined weather effects

        Args:
            pil_image: PIL Image (RGB)
            condition: 'rain_lowlight' or 'fog_night'
            severity: 'light', 'medium', or 'heavy'

        Returns:
            Processed PIL Image
        """
        config = SEVERITY_CONFIG[severity]
        gan_blend = config['gan_blend']
        darkness = config['darkness']

        # Convert to tensor
        tensor = TRANSFORM(pil_image).unsqueeze(0).to(self.device)

        if condition == 'rain_lowlight':
            # Apply rain GAN effect
            if 'rain' in self.models:
                tensor = self.apply_gan_effect(tensor, 'rain', gan_blend)
            # Apply additional night GAN if available
            if 'night' in self.models:
                tensor = self.apply_gan_effect(tensor, 'night', gan_blend * 0.3)

        elif condition == 'fog_night':
            # Apply fog GAN effect
            if 'fog' in self.models:
                tensor = self.apply_gan_effect(tensor, 'fog', gan_blend)
            # Apply night GAN if available
            if 'night' in self.models:
                tensor = self.apply_gan_effect(tensor, 'night', gan_blend * 0.3)

        # Convert back to PIL
        tensor = tensor.squeeze(0)
        tensor = (tensor + 1) / 2  # Denormalize from [-1,1] to [0,1]
        tensor = tensor.clamp(0, 1)
        img_array = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        result_pil = Image.fromarray(img_array)

        # Apply low-light/darkness effect (this adds the "lowlight" and "night" parts)
        result_pil = self.apply_low_light(result_pil, darkness)

        return result_pil


# ============================================================================
# IMAGE PROCESSING
# ============================================================================

def load_image(image_path, max_size=1280):
    """Load an image, optionally resizing if too large"""
    img = Image.open(image_path).convert('RGB')

    # Resize if too large (to fit in GPU memory)
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        new_size = (int(w * ratio), int(h * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    return img


def save_image(pil_image, output_path):
    """Save PIL image as JPEG"""
    pil_image.save(output_path, 'JPEG', quality=95)


def copy_annotation(src_label_path, dst_label_path):
    """Copy YOLO annotation file"""
    if src_label_path.exists():
        shutil.copy(src_label_path, dst_label_path)


def generate_metadata(image_name, condition, severity, params, output_path):
    """Generate metadata files for an image"""
    metadata = {
        "original_image": image_name,
        "generated_image": output_path.name,
        "weather_type": condition,
        "severity_level": severity,
        "parameters": params,
        "timestamp": datetime.now().isoformat(),
        "random_seed": params.get('seed', 'N/A')
    }

    # Save as JSON
    json_path = METADATA_DIR / f"{output_path.stem}.json"
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Save as TXT (human readable)
    txt_path = METADATA_DIR / f"{output_path.stem}.txt"
    with open(txt_path, 'w') as f:
        f.write(f"Original Image: {image_name}\n")
        f.write(f"Weather Type: {condition}\n")
        f.write(f"Severity Level: {severity}\n")
        f.write(f"GAN Blend: {params.get('gan_blend', 'N/A')}\n")
        f.write(f"Darkness: {params.get('darkness', 'N/A')}\n")
        f.write(f"Random Seed: {params.get('seed', 'N/A')}\n")
        f.write(f"Generated: {metadata['timestamp']}\n")


# ============================================================================
# MAIN GENERATION PIPELINE
# ============================================================================

def process_images(generator, condition, severity, start_idx=0, end_idx=None):
    """
    Process all images in input directory

    Args:
        generator: WeatherGenerator instance
        condition: 'rain_lowlight' or 'fog_night'
        severity: 'light', 'medium', or 'heavy'
        start_idx: Start index for processing
        end_idx: End index for processing
    """
    input_images_dir = INPUT_DIR / "images"
    input_labels_dir = INPUT_DIR / "labels"

    output_images_dir = OUTPUT_DIR / condition / "images"
    output_labels_dir = OUTPUT_DIR / condition / "labels"

    # Create output directories
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    # Get list of images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    images = sorted([
        f for f in input_images_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ])

    if not images:
        print(f"\nERROR: No images found in {input_images_dir}")
        print("Please add images to the input/images/ folder")
        return 0

    # Apply range filter
    total_images = len(images)
    if end_idx is None:
        end_idx = total_images
    images = images[start_idx:end_idx]

    print(f"\n{'=' * 50}")
    print(f"Processing: {condition} - {severity}")
    print(f"Images: {len(images)} (indices {start_idx} to {end_idx})")
    print(f"Output: {output_images_dir}")
    print(f"{'=' * 50}")

    config = SEVERITY_CONFIG[severity]
    severity_suffix = config['suffix']

    generated_count = 0

    for img_path in tqdm(images, desc=f"{condition}_{severity}"):
        try:
            # Set seed for reproducibility
            seed = random.randint(0, 999999)
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Load image
            pil_img = load_image(img_path)

            # Apply combined weather effect
            result_img = generator.apply_combined_effect(pil_img, condition, severity)

            # Generate output filename
            output_name = f"{img_path.stem}_{condition}_{severity_suffix}.jpg"
            output_path = output_images_dir / output_name

            # Save image
            save_image(result_img, output_path)

            # Copy annotation
            label_path = input_labels_dir / f"{img_path.stem}.txt"
            output_label_path = output_labels_dir / f"{img_path.stem}_{condition}_{severity_suffix}.txt"
            copy_annotation(label_path, output_label_path)

            # Generate metadata
            params = {
                'gan_blend': config['gan_blend'],
                'darkness': config['darkness'],
                'seed': seed,
                'condition': condition,
                'severity': severity
            }
            generate_metadata(img_path.name, condition, severity, params, output_path)

            generated_count += 1

        except Exception as e:
            print(f"\nError processing {img_path.name}: {e}")
            continue

    print(f"Generated {generated_count} images for {condition} - {severity}")
    return generated_count


def generate_all(generator, start_idx=0, end_idx=None):
    """Generate all conditions and severities"""
    conditions = ['rain_lowlight', 'fog_night']
    severities = ['light', 'medium', 'heavy']

    total = 0
    for condition in conditions:
        for severity in severities:
            count = process_images(generator, condition, severity, start_idx, end_idx)
            total += count

    return total


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic weather images using GANs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate.py --all                           # Generate all combinations
  python generate.py -c rain_lowlight -s heavy       # Specific condition/severity
  python generate.py --all --start 0 --end 100       # Process images 0-99 only

Team workflow (split 500 images among 4 people):
  Person 1: python generate.py --all --start 0 --end 125
  Person 2: python generate.py --all --start 125 --end 250
  Person 3: python generate.py --all --start 250 --end 375
  Person 4: python generate.py --all --start 375 --end 500
        """
    )
    parser.add_argument(
        '--condition', '-c',
        choices=['rain_lowlight', 'fog_night'],
        help='Weather condition to generate'
    )
    parser.add_argument(
        '--severity', '-s',
        choices=['light', 'medium', 'heavy'],
        default='medium',
        help='Severity level (default: medium)'
    )
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Generate all conditions and severities'
    )
    parser.add_argument(
        '--start', type=int, default=0,
        help='Start index for image processing'
    )
    parser.add_argument(
        '--end', type=int, default=None,
        help='End index for image processing'
    )
    parser.add_argument(
        '--device', '-d',
        default='cuda',
        help='Device to use (cuda/cpu)'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("SYNTHETIC WEATHER IMAGE GENERATOR")
    print("Combined Conditions: rain+lowlight, fog+night")
    print("=" * 60)

    # Check input directory
    input_images = INPUT_DIR / "images"
    if not input_images.exists() or not any(input_images.iterdir()):
        print(f"\nERROR: No images in {input_images}")
        print("Please add your base images to the input/images/ folder")
        sys.exit(1)

    # Initialize generator
    generator = WeatherGenerator(device=args.device)

    if not generator.models:
        print("\nERROR: No models available!")
        print("Please run: python download_models.py")
        sys.exit(1)

    # Generate images
    if args.all:
        total = generate_all(generator, args.start, args.end)
    elif args.condition:
        total = process_images(
            generator,
            args.condition,
            args.severity,
            start_idx=args.start,
            end_idx=args.end
        )
    else:
        print("\nPlease specify --condition or --all")
        parser.print_help()
        sys.exit(1)

    print("\n" + "=" * 60)
    print(f"GENERATION COMPLETE: {total} images generated")
    print("=" * 60)
    print(f"\nOutput locations:")
    print(f"  Images: {OUTPUT_DIR}/")
    print(f"  Metadata: {METADATA_DIR}/")


if __name__ == "__main__":
    main()
