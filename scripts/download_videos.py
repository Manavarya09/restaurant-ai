"""
Download free restaurant/crowd footage from YouTube and run pseudo-labeling.

Usage:
    python scripts/download_videos.py
    python scripts/download_videos.py --clips 5 --max-duration 120
"""
import sys
import os
import argparse
import subprocess
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "data" / "raw_videos"

# Creative Commons / public domain searches
SEARCH_QUERIES = [
    "ytsearch3:restaurant crowd surveillance footage",
    "ytsearch3:shopping mall indoor people walking",
    "ytsearch2:food court busy customers",
]


def download_videos(max_duration: int, max_clips: int):
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    videos = []
    for query in SEARCH_QUERIES:
        if len(videos) >= max_clips:
            break

        print(f"\n[search] {query}")
        cmd = [
            "yt-dlp",
            query,
            "--output", str(VIDEO_DIR / "%(id)s.%(ext)s"),
            "--format", "bestvideo[ext=mp4][height<=480]+bestaudio/best[height<=480]",
            "--merge-output-format", "mp4",
            "--match-filter", f"duration < {max_duration}",
            "--max-downloads", str(max(1, max_clips - len(videos))),
            "--no-playlist",
            "--quiet",
            "--progress",
            "--no-warnings",
            "--socket-timeout", "15",
            "--retries", "2",
        ]

        result = subprocess.run(cmd, capture_output=False)
        new_videos = list(VIDEO_DIR.glob("*.mp4"))
        videos = new_videos
        print(f"  {len(videos)} videos so far")

        if len(videos) >= max_clips:
            break

    return [str(v) for v in VIDEO_DIR.glob("*.mp4")]


def run_labeling(video_paths: list, interval: int, max_frames: int):
    if not video_paths:
        print("[ERROR] No videos downloaded.")
        return

    print(f"\n[label] Running pseudo-labeling on {len(video_paths)} video(s)...")
    for vp in video_paths:
        print(f"\n  → {Path(vp).name}")
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "collect_and_label.py"),
            "--source", vp,
            "--interval", str(interval),
            "--max-frames", str(max_frames),
        ]
        subprocess.run(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download crowd/restaurant videos and label them")
    parser.add_argument("--clips",        type=int, default=4,
                        help="Number of video clips to download (default 4)")
    parser.add_argument("--max-duration", dest="max_duration", type=int, default=180,
                        help="Max video duration in seconds (default 180)")
    parser.add_argument("--interval",     type=int, default=20,
                        help="Sample every N frames for labeling (default 20)")
    parser.add_argument("--max-frames",   dest="max_frames", type=int, default=300,
                        help="Max labeled frames per video (default 300)")
    parser.add_argument("--skip-download", dest="skip_download", action="store_true",
                        help="Skip download, label existing videos in data/raw_videos/")
    args = parser.parse_args()

    if args.skip_download:
        videos = [str(v) for v in VIDEO_DIR.glob("*.mp4")]
        print(f"[skip] Using {len(videos)} existing videos in {VIDEO_DIR}")
    else:
        videos = download_videos(args.max_duration, args.clips)

    if not videos:
        print(f"[ERROR] No .mp4 files found in {VIDEO_DIR}")
        sys.exit(1)

    print(f"\n[found] {len(videos)} video(s):")
    for v in videos:
        size = Path(v).stat().st_size / 1e6
        print(f"  {Path(v).name} ({size:.1f} MB)")

    run_labeling(videos, args.interval, args.max_frames)

    print("\n[done] Dataset ready.")
    print("Next: python scripts/train_restaurant.py --epochs 50 --device mps")
