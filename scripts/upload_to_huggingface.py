#!/usr/bin/env python3
"""Upload ToxiRewriteCN dataset artifacts to a Hugging Face dataset repo."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_FILES = [
    "ToxiRewriteCN.json",
    "train_1000.json",
    "test_556.json",
    "r1_train.json",
    "train_full_8148.json",
    "train_polarity_ratio121.json",
]


def copy_required_files(staging_dir: Path) -> None:
    """Create the Hugging Face dataset repo layout in a temporary folder."""
    shutil.copy2(ROOT / "LICENSE", staging_dir / "LICENSE")

    attrs = ROOT / "hf_dataset" / ".gitattributes"
    if attrs.exists():
        shutil.copy2(attrs, staging_dir / ".gitattributes")

    data_dir = staging_dir / "data"
    data_dir.mkdir()
    for file_name in DATA_FILES:
        shutil.copy2(ROOT / "data" / file_name, data_dir / file_name)


def render_dataset_card(staging_dir: Path, repo_id: str) -> None:
    card = (ROOT / "hf_dataset" / "README.md").read_text(encoding="utf-8")
    card = card.replace("{{HF_DATASET_REPO_ID}}", repo_id)
    (staging_dir / "README.md").write_text(card, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync ToxiRewriteCN data and dataset card to Hugging Face."
    )
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("HF_DATASET_REPO_ID", "shanewang/ToxiRewriteCN"),
        help="Hugging Face dataset repo id, for example shanewang/ToxiRewriteCN.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the dataset repo as private if it does not already exist.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face write token. Defaults to HF_TOKEN or local HF login.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Hub branch or revision to upload to.",
    )
    parser.add_argument(
        "--commit-message",
        default="Sync ToxiRewriteCN dataset from GitHub",
        help="Commit message used on the Hugging Face Hub.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the upload folder and print included files without uploading.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if os.environ.get("GITHUB_ACTIONS") == "true" and not args.token:
        raise SystemExit(
            "HF_TOKEN is required in GitHub Actions. Add it as a repository secret."
        )

    with tempfile.TemporaryDirectory() as tmp:
        staging_dir = Path(tmp)
        copy_required_files(staging_dir)
        render_dataset_card(staging_dir, args.repo_id)

        if args.dry_run:
            for path in sorted(staging_dir.rglob("*")):
                if path.is_file():
                    print(path.relative_to(staging_dir))
            return

        try:
            from huggingface_hub import HfApi
        except ImportError as exc:  # pragma: no cover - only hit on missing optional dep
            raise SystemExit(
                "Missing dependency: install with `pip install huggingface_hub`."
            ) from exc

        api = HfApi(token=args.token)
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=args.private,
            exist_ok=True,
        )
        api.upload_folder(
            folder_path=staging_dir,
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=args.revision,
            commit_message=args.commit_message,
        )

    print(f"Uploaded dataset artifacts to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
