"""
Download and prepare COCO dataset filtered for street scenes
Downloads 500 images with: person, car, traffic light

Run: python download_dataset.py
"""

import os
import sys
import json
import random
import urllib.request
import zipfile
from pathlib import Path
from collections import defaultdict

# Paths
BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "input"
IMAGES_DIR = INPUT_DIR / "images"
LABELS_DIR = INPUT_DIR / "labels"
TEMP_DIR = BASE_DIR / "temp_coco"

# Create directories
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
LABELS_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# COCO URLs
COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_IMAGES_BASE_URL = "http://images.cocodataset.org/train2017/"

# Target classes (COCO category IDs)
# person=1, car=3, traffic light=10
TARGET_CLASSES = {
    1: "person",
    3: "car",
    10: "traffic_light"
}

# How many images to download
TARGET_IMAGE_COUNT = 500


def download_file(url, output_path, description=""):
    """Download file with progress"""
    try:
        print(f"  Downloading: {description or url}")

        def progress_hook(count, block_size, total_size):
            if total_size > 0:
                percent = min(100, count * block_size * 100 // total_size)
                mb_done = count * block_size / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                sys.stdout.write(f"\r  Progress: {percent}% ({mb_done:.1f}/{mb_total:.1f} MB)")
                sys.stdout.flush()

        urllib.request.urlretrieve(url, output_path, progress_hook)
        print()
        return True
    except Exception as e:
        print(f"\n  FAILED: {e}")
        return False


def download_annotations():
    """Download COCO annotations"""
    zip_path = TEMP_DIR / "annotations.zip"
    annotations_path = TEMP_DIR / "annotations" / "instances_train2017.json"

    if annotations_path.exists():
        print("  Annotations already downloaded")
        return annotations_path

    print("\n[1/3] Downloading COCO annotations (~250MB)...")
    if not download_file(COCO_ANNOTATIONS_URL, zip_path, "COCO 2017 annotations"):
        return None

    print("  Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(TEMP_DIR)

    # Clean up zip
    zip_path.unlink()

    return annotations_path


def filter_images_with_target_classes(annotations_path):
    """Find images containing our target classes"""
    print("\n[2/3] Filtering images with target classes...")
    print(f"  Target classes: {list(TARGET_CLASSES.values())}")

    with open(annotations_path, 'r') as f:
        coco_data = json.load(f)

    # Build image_id -> annotations mapping
    image_annotations = defaultdict(list)
    for ann in coco_data['annotations']:
        if ann['category_id'] in TARGET_CLASSES:
            image_annotations[ann['image_id']].append(ann)

    # Filter images that have at least 2 of our target classes
    good_images = []
    for img in coco_data['images']:
        img_id = img['id']
        if img_id in image_annotations:
            # Check how many different target classes are in this image
            classes_in_image = set(ann['category_id'] for ann in image_annotations[img_id])
            target_classes_present = classes_in_image.intersection(TARGET_CLASSES.keys())

            # Prefer images with multiple target classes
            if len(target_classes_present) >= 2:
                good_images.append({
                    'image': img,
                    'annotations': image_annotations[img_id],
                    'class_count': len(target_classes_present)
                })

    print(f"  Found {len(good_images)} images with 2+ target classes")

    # Sort by class diversity and randomly select
    random.seed(42)  # For reproducibility
    random.shuffle(good_images)

    # Take top images
    selected = good_images[:TARGET_IMAGE_COUNT]
    print(f"  Selected {len(selected)} images")

    return selected, coco_data['categories']


def convert_to_yolo(bbox, img_width, img_height):
    """Convert COCO bbox [x, y, width, height] to YOLO format [x_center, y_center, width, height] (normalized)"""
    x, y, w, h = bbox

    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    w_norm = w / img_width
    h_norm = h / img_height

    return x_center, y_center, w_norm, h_norm


def download_images_and_create_labels(selected_images):
    """Download images and create YOLO labels"""
    print(f"\n[3/3] Downloading {len(selected_images)} images...")

    # Map COCO category IDs to our class indices (0, 1, 2)
    coco_to_yolo_class = {
        1: 0,   # person -> 0
        3: 1,   # car -> 1
        10: 2   # traffic light -> 2
    }

    downloaded = 0
    failed = 0

    for i, item in enumerate(selected_images):
        img_info = item['image']
        annotations = item['annotations']

        img_filename = img_info['file_name']
        img_id = img_info['id']
        img_width = img_info['width']
        img_height = img_info['height']

        # Progress
        sys.stdout.write(f"\r  Downloading: {i+1}/{len(selected_images)} ({img_filename})")
        sys.stdout.flush()

        # Download image
        img_url = COCO_IMAGES_BASE_URL + img_filename
        img_path = IMAGES_DIR / img_filename

        if not img_path.exists():
            try:
                urllib.request.urlretrieve(img_url, img_path)
            except Exception as e:
                failed += 1
                continue

        # Create YOLO label
        label_filename = img_filename.rsplit('.', 1)[0] + '.txt'
        label_path = LABELS_DIR / label_filename

        with open(label_path, 'w') as f:
            for ann in annotations:
                if ann['category_id'] in coco_to_yolo_class:
                    class_id = coco_to_yolo_class[ann['category_id']]
                    x_center, y_center, w, h = convert_to_yolo(
                        ann['bbox'], img_width, img_height
                    )
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

        downloaded += 1

    print(f"\n  Downloaded: {downloaded} images")
    if failed > 0:
        print(f"  Failed: {failed} images")

    return downloaded


def cleanup():
    """Remove temporary files"""
    print("\nCleaning up temporary files...")
    import shutil
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    print("  Done")


def main():
    print("=" * 60)
    print("COCO DATASET DOWNLOADER")
    print("Filtered for: person, car, traffic light")
    print("=" * 60)

    # Check if images already exist
    existing_images = list(IMAGES_DIR.glob("*.jpg"))
    if len(existing_images) >= TARGET_IMAGE_COUNT:
        print(f"\nDataset already exists ({len(existing_images)} images)")
        print("Delete input/images/ folder to re-download")
        return

    # Download annotations
    annotations_path = download_annotations()
    if not annotations_path:
        print("Failed to download annotations!")
        return

    # Filter images
    selected_images, categories = filter_images_with_target_classes(annotations_path)

    if not selected_images:
        print("No suitable images found!")
        return

    # Download images and create labels
    count = download_images_and_create_labels(selected_images)

    # Cleanup
    cleanup()

    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"""
Images downloaded: {count}
Location: {IMAGES_DIR}
Labels: {LABELS_DIR}

Classes (YOLO format):
  0 = person
  1 = car
  2 = traffic_light

Next step:
  python download_models.py   # Download GAN models
  python generate.py --all    # Generate weather images
""")


if __name__ == "__main__":
    main()
