# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Download script for Alpamayo-R1 model and Physical AI AV dataset.

Auto-detects best HuggingFace endpoint based on authentication.
Use --use-mirror or --endpoint flags to override.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import physical_ai_av
from huggingface_hub import whoami
from huggingface_hub.utils import HfHubHTTPError

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1


def check_authentication():
    """Check if user is authenticated with HuggingFace Hub."""
    try:
        user_info = whoami()
        print(f"✓ Authenticated as: {user_info['name']}")
        return True
    except Exception:
        print("❌ Not authenticated with HuggingFace Hub")
        print("\n1. Request access to:")
        print("   - https://huggingface.co/nvidia/Alpamayo-R1-10B")
        print("   - https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles")
        print("\n2. Run: huggingface-cli login")
        print("   Token: https://huggingface.co/settings/tokens")
        return False


def check_directory_writable(path: Path) -> bool:
    """Check if directory is writable, create if needed."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        (path / ".write_test").touch()
        (path / ".write_test").unlink()
        return True
    except Exception as e:
        print(f"❌ Cannot write to {path}: {e}")
        return False


def download_model(model_path: Path) -> bool:
    """Download Alpamayo-R1 model to local path."""
    print(f"\n{'='*70}\nDOWNLOADING MODEL\n{'='*70}")
    print(f"Model: nvidia/Alpamayo-R1-10B\nDestination: {model_path}\nSize: ~22 GB\n{'='*70}\n")

    try:
        if (model_path / "config.json").exists():
            if input("⚠ Model exists. Re-download? [y/N]: ").strip().lower() not in ['y', 'yes']:
                print("Skipping model download.")
                return True

        print("Downloading model (this may take several minutes)...\n")
        model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16, cache_dir=None)
        model.save_pretrained(model_path)
        print(f"\n✓ Model downloaded to {model_path}")
        return True

    except HfHubHTTPError as e:
        if "403" in str(e) or "401" in str(e):
            print("\n❌ Access denied. Request access: https://huggingface.co/nvidia/Alpamayo-R1-10B")
        else:
            print(f"\n❌ Download failed: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return False


def download_dataset(dataset_path: Path, clip_ids: list[str]) -> bool:
    """Download Physical AI AV dataset clips to local path."""
    print(f"\n{'='*70}\nDOWNLOADING DATASET\n{'='*70}")
    print(f"Dataset: nvidia/PhysicalAI-Autonomous-Vehicles\nDestination: {dataset_path}")
    print(f"Clips: {len(clip_ids)}\n" + "\n".join(f"  - {cid}" for cid in clip_ids) + f"\n{'='*70}\n")

    try:
        print("Initializing dataset interface...")
        avdi = physical_ai_av.PhysicalAIAVDatasetInterface(
            local_dir=str(dataset_path),
            cache_dir=str(dataset_path),
            confirm_download_threshold_gb=float("inf")
        )

        print("\n1. Downloading metadata...")
        avdi.download_metadata()
        print("✓ Metadata downloaded")

        features = [
            avdi.features.LABELS.EGOMOTION,
            avdi.features.CAMERA.CAMERA_CROSS_LEFT_120FOV,
            avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV,
            avdi.features.CAMERA.CAMERA_CROSS_RIGHT_120FOV,
            avdi.features.CAMERA.CAMERA_FRONT_TELE_30FOV,
        ]

        print(f"\n2. Downloading {len(clip_ids)} clip(s)...")
        for i, clip_id in enumerate(clip_ids, 1):
            print(f"\nClip {i}/{len(clip_ids)}: {clip_id}")
            try:
                chunk_id = avdi.get_clip_chunk(clip_id)
                print(f"  → Chunk {chunk_id}")
                avdi.download_clip_features(clip_id=clip_id, features=features)
                print("  ✓ Downloaded")
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                return False

        print(f"\n✓ All clips downloaded to {dataset_path}")
        return True

    except HfHubHTTPError as e:
        if "403" in str(e) or "401" in str(e):
            print("\n❌ Access denied to dataset")
            print("Request access: https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles")
            if "gated" in str(e).lower():
                print("\n💡 Token issue: Enable 'Read access to public gated repos' in token settings")
                print("   https://huggingface.co/settings/tokens")
        else:
            print(f"\n❌ Download failed: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return False


def validate_model(model_path: Path) -> bool:
    """Validate that model files exist."""
    print(f"\nValidating model at {model_path}...")

    if not (model_path / "config.json").exists():
        print("  ❌ Missing config.json")
        return False
    print("  ✓ Found: config.json")

    # Check for sharded or single weight files
    sharded_index = list(model_path.glob("*.index.json"))
    sharded_weights = list(model_path.glob("model-*.safetensors")) + list(model_path.glob("pytorch_model-*.bin"))
    single_weights = list(model_path.glob("model.safetensors")) + list(model_path.glob("pytorch_model.bin"))

    if sharded_index and sharded_weights:
        print(f"  ✓ Found: {sharded_index[0].name} (sharded)")
        print(f"  ✓ Found: {len(sharded_weights)} weight shard(s)")
    elif single_weights:
        print(f"  ✓ Found: {single_weights[0].name}")
    else:
        print("  ❌ Missing model weights")
        return False

    print("✓ Model validation passed")
    return True


def validate_dataset(dataset_path: Path, clip_ids: list[str]) -> bool:
    """Validate that dataset files exist."""
    print(f"\nValidating dataset at {dataset_path}...")

    # Check required files
    required = {
        "metadata/": dataset_path / "metadata",
        "features.csv": dataset_path / "features.csv",
        "clip_index.parquet": dataset_path / "clip_index.parquet",
    }

    for name, path in required.items():
        if not path.exists():
            print(f"  ❌ Missing {name}")
            return False
        print(f"  ✓ Found: {name}")

    # Validate clip data
    try:
        avdi = physical_ai_av.PhysicalAIAVDatasetInterface(
            local_dir=str(dataset_path),
            cache_dir=str(dataset_path)
        )

        for clip_id in clip_ids:
            chunk_id = avdi.get_clip_chunk(clip_id)
            camera_chunks = list((dataset_path / "camera").glob(f"*/*.chunk_{chunk_id}.zip"))
            labels_chunks = list((dataset_path / "labels").glob(f"*/*.chunk_{chunk_id}.zip"))

            if not camera_chunks:
                print(f"  ❌ Missing camera data for clip {clip_id} (chunk {chunk_id})")
                return False
            if not labels_chunks:
                print(f"  ❌ Missing labels data for clip {clip_id} (chunk {chunk_id})")
                return False

            print(f"  ✓ Clip {clip_id}: {len(camera_chunks)} camera, {len(labels_chunks)} label file(s)")

        print("✓ Dataset validation passed")
        return True

    except Exception as e:
        print(f"  ❌ Validation failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download Alpamayo-R1 model and Physical AI AV dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python src/alpamayo_r1/download_data.py\n"
               "  python src/alpamayo_r1/download_data.py --clip-ids clip1 clip2\n"
               "  python src/alpamayo_r1/download_data.py --skip-model\n"
               "  python src/alpamayo_r1/download_data.py --use-mirror"
    )

    parser.add_argument("--model-path", type=Path, default=Path("/data/models/nvidia/Alpamayo-R1-10B"))
    parser.add_argument("--dataset-path", type=Path, default=Path("/data/datasets/nvidia/PhysicalAI-Autonomous-Vehicles"))
    parser.add_argument("--clip-ids", nargs="+", default=["030c760c-ae38-49aa-9ad8-f5650a545d26"])
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--skip-dataset", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--skip-auth-check", action="store_true")
    parser.add_argument("--use-mirror", action="store_true", help="Use hf-mirror.com endpoint")
    parser.add_argument("--endpoint", type=str, help="Custom HuggingFace endpoint URL")

    args = parser.parse_args()

    print(f"\n{'='*70}\nAlpamayo-R1 Data Download Script\n{'='*70}")

    # Determine HuggingFace endpoint (priority: --endpoint > --use-mirror > HF_ENDPOINT env > auto-detect)
    if args.endpoint:
        hf_endpoint, reason, use_mirror = args.endpoint, "from --endpoint flag", "mirror" in args.endpoint
    elif args.use_mirror:
        hf_endpoint, reason, use_mirror = 'https://hf-mirror.com', "from --use-mirror flag", True
    elif 'HF_ENDPOINT' in os.environ:
        hf_endpoint, reason = os.environ['HF_ENDPOINT'], "from HF_ENDPOINT env var"
        use_mirror = 'mirror' in hf_endpoint or hf_endpoint != 'https://huggingface.co'
    else:
        # Auto-detect based on authentication
        try:
            whoami()
            hf_endpoint, reason, use_mirror = 'https://huggingface.co', "auto-detected (authenticated)", False
            print("\n✓ Detected HuggingFace authentication")
        except:
            hf_endpoint, reason, use_mirror = 'https://hf-mirror.com', "auto-detected (not authenticated)", True
            print("\n⚠ No authentication - using mirror")

    os.environ['HF_ENDPOINT'] = hf_endpoint
    if use_mirror and not args.skip_auth_check:
        args.skip_auth_check = True

    print(f"\n🌐 Endpoint: {hf_endpoint} ({reason})")
    print(f"   💡 Override: --{'use-mirror' if not use_mirror else 'endpoint https://huggingface.co'}")

    # Check authentication
    if not args.skip_auth_check and not check_authentication():
        sys.exit(1)

    # Check write permissions
    for path in [args.model_path.parent, args.dataset_path.parent]:
        if not check_directory_writable(path):
            sys.exit(1)

    # Download
    if not args.skip_model and not download_model(args.model_path):
        sys.exit(1)
    if not args.skip_dataset and not download_dataset(args.dataset_path, args.clip_ids):
        sys.exit(1)

    # Validate
    if not args.skip_validation:
        print(f"\n{'='*70}\nVALIDATION\n{'='*70}")
        if (not args.skip_model and not validate_model(args.model_path)) or \
           (not args.skip_dataset and not validate_dataset(args.dataset_path, args.clip_ids)):
            print("\n❌ Validation failed")
            sys.exit(1)

    # Success
    print(f"\n{'='*70}\n✓ DOWNLOAD COMPLETE\n{'='*70}")
    if not args.skip_model:
        print(f"Model: {args.model_path}")
    if not args.skip_dataset:
        print(f"Dataset: {args.dataset_path} ({len(args.clip_ids)} clips)")
    print(f"{'='*70}\n\nNext: python src/alpamayo_r1/test_inference.py\n")


if __name__ == "__main__":
    main()
