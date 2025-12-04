#!/usr/bin/env python3
"""
Dataset Download Script for DermaScan Pro
Downloads HAM10000 dataset from Kaggle or generates expanded test dataset.

Usage:
    python download_dataset.py --source kaggle --output server/data/HAM10000
    python download_dataset.py --source expand --output server/data/expanded
"""

import os
import sys
import argparse
import zipfile
import requests
from pathlib import Path
import shutil
import pandas as pd
from tqdm import tqdm

# Fix Windows encoding issues
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

def download_file(url, destination):
    """Download file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f, tqdm(
        desc=destination.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)

def download_ham10000_from_dataverse():
    """
    Download HAM10000 dataset from Harvard Dataverse.
    This is the official source and doesn't require Kaggle API.
    """
    print("Downloading HAM10000 dataset from Harvard Dataverse...")
    
    base_url = "https://dataverse.harvard.edu/api/access/datafile"
    
    # Part 1 and Part 2 file IDs
    files = {
        "HAM10000_images_part_1.zip": "3301406",
        "HAM10000_images_part_2.zip": "3301407",
        "HAM10000_metadata.csv": "3301404"
    }
    
    output_dir = Path("server/data/HAM10000")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for filename, file_id in files.items():
        url = f"{base_url}/{file_id}"
        dest_path = output_dir / filename
        
        if dest_path.exists():
            print(f"✓ {filename} already exists, skipping...")
            continue
        
        print(f"\nDownloading {filename}...")
        try:
            download_file(url, dest_path)
            
            # Extract if zip file
            if filename.endswith('.zip'):
                print(f"Extracting {filename}...")
                with zipfile.ZipFile(dest_path, 'r') as zip_ref:
                    zip_ref.extractall(output_dir)
                print(f"✓ Extracted {filename}")
                # Remove zip file after extraction
                dest_path.unlink()
        except Exception as e:
            print(f"✗ Error downloading {filename}: {e}")
            return False
    
    # Rename metadata file
    metadata_src = output_dir / "HAM10000_metadata.csv"
    metadata_dst = output_dir / "metadata.csv"
    if metadata_src.exists() and not metadata_dst.exists():
        shutil.move(str(metadata_src), str(metadata_dst))
    
    print(f"\n✓ HAM10000 dataset downloaded successfully!")
    print(f"Location: {output_dir.absolute()}")
    
    # Count images
    image_count = len(list(output_dir.glob("*.jpg")))
    print(f"Total images: {image_count}")
    
    return True

def expand_current_dataset(output_dir="server/data/expanded"):
    """
    Expand the current small dataset using data augmentation.
    This creates more training samples from the existing 8 images.
    """
    print("Expanding current dataset with augmentation...")
    
    try:
        from PIL import Image
        import numpy as np
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
    except ImportError:
        print("Error: Required packages not found. Install with: pip install pillow tensorflow")
        return False
    
    source_dir = Path("server/data/custom")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    metadata_file = source_dir / "metadata.csv"
    if not metadata_file.exists():
        print(f"Error: {metadata_file} not found")
        return False
    
    df = pd.read_csv(metadata_file)
    
    # Create augmentation generator
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.7, 1.3],
        fill_mode='nearest'
    )
    
    # Augment each image
    augmented_rows = []
    augment_per_image = 125  # Create 125 variants of each image = 1000 total
    
    print(f"Generating {augment_per_image} variants per image...")
    
    for idx, row in df.iterrows():
        isic_id = row['isic_id']
        original_img_path = source_dir / f"{isic_id}.jpg"
        
        if not original_img_path.exists():
            print(f"Warning: {original_img_path} not found, skipping...")
            continue
        
        # Load original image
        img = Image.open(original_img_path).convert('RGB')
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, 0)
        
        # Save original
        shutil.copy(original_img_path, output_path / f"{isic_id}.jpg")
        augmented_rows.append(row.to_dict())
        
        # Generate augmented versions
        i = 0
        for batch in datagen.flow(img_array, batch_size=1):
            aug_id = f"{isic_id}_aug_{i:04d}"
            aug_path = output_path / f"{aug_id}.jpg"
            
            # Save augmented image
            aug_img = Image.fromarray(batch[0].astype('uint8'))
            aug_img.save(aug_path)
            
            # Add to metadata
            new_row = row.to_dict().copy()
            new_row['isic_id'] = aug_id
            augmented_rows.append(new_row)
            
            i += 1
            if i >= augment_per_image:
                break
        
        print(f"✓ {isic_id}: created {augment_per_image} variants")
    
    # Save augmented metadata
    aug_df = pd.DataFrame(augmented_rows)
    aug_df.to_csv(output_path / "metadata.csv", index=False)
    
    print(f"\n✓ Dataset expanded successfully!")
    print(f"Location: {output_path.absolute()}")
    print(f"Total images: {len(augmented_rows)}")
    print(f"Class distribution:")
    print(aug_df['diagnosis_1'].value_counts())
    
    return True

def download_from_kaggle(output_dir="server/data/HAM10000"):
    """
    Download dataset from Kaggle using Kaggle API.
    Requires: pip install kaggle
    and Kaggle API credentials in ~/.kaggle/kaggle.json
    """
    print("Downloading from Kaggle...")
    
    try:
        import kaggle
    except ImportError:
        print("Error: Kaggle API not installed.")
        print("Install with: pip install kaggle")
        print("Then place your kaggle.json in ~/.kaggle/")
        return False
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download HAM10000 dataset
        print("Downloading kmader/skin-cancer-mnist-ham10000...")
        kaggle.api.dataset_download_files(
            'kmader/skin-cancer-mnist-ham10000',
            path=str(output_path),
            unzip=True
        )
        
        print(f"\n✓ Dataset downloaded successfully!")
        print(f"Location: {output_path.absolute()}")
        
        # Count images
        image_count = len(list(output_path.glob("**/*.jpg")))
        print(f"Total images: {image_count}")
        
        return True
    except Exception as e:
        print(f"Error downloading from Kaggle: {e}")
        print("\nTrying Harvard Dataverse instead...")
        return download_ham10000_from_dataverse()

def main():
    parser = argparse.ArgumentParser(description='Download dermoscopy dataset')
    parser.add_argument('--source', type=str, choices=['kaggle', 'dataverse', 'expand'], 
                       default='dataverse',
                       help='Dataset source (kaggle requires API key, dataverse is free, expand augments current dataset)')
    parser.add_argument('--output', type=str, default='server/data/HAM10000',
                       help='Output directory for dataset')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DermaScan Pro - Dataset Download Tool")
    print("=" * 60)
    
    success = False
    
    if args.source == 'kaggle':
        success = download_from_kaggle(args.output)
    elif args.source == 'dataverse':
        success = download_ham10000_from_dataverse()
    elif args.source == 'expand':
        success = expand_current_dataset(args.output)
    
    if success:
        print("\n" + "=" * 60)
        print("✓ Dataset setup complete!")
        print("=" * 60)
        print(f"\nNext steps:")
        print(f"1. Verify dataset in: {args.output}")
        print(f"2. Start training with:")
        if args.source == 'expand':
            print(f"   Dataset name: 'expanded'")
        else:
            print(f"   Dataset name: 'HAM10000'")
        print(f"3. Use Admin Lab in the UI or API to train")
        return 0
    else:
        print("\n✗ Dataset download failed")
        print("\nAlternative: Download manually from:")
        print("- https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T")
        print("- https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000")
        return 1

if __name__ == '__main__':
    sys.exit(main())
