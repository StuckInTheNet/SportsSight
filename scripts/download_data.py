"""Data acquisition scripts — download training data and sample footage."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import click
import httpx
from rich.console import Console
from rich.progress import Progress

console = Console()
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """Download datasets and sample footage for SportsSight."""
    pass


@cli.command()
@click.option("--output", "-o", default=str(DATA_DIR / "raw" / "trackid3x3"), help="Output directory")
def trackid3x3(output: str) -> None:
    """Download TrackID3x3 dataset (basketball tracking + pose + re-ID).

    Paper: https://arxiv.org/abs/2503.18282
    Repo:  https://github.com/open-starlab/TrackID3x3
    """
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold]TrackID3x3 Dataset[/bold]")
    console.print("Purpose: Multi-player tracking + identification + pose estimation in basketball")
    console.print(f"Output:  {out_dir}")
    console.print()

    # Clone the repository which contains download scripts and annotations
    repo_url = "https://github.com/open-starlab/TrackID3x3.git"
    repo_dir = out_dir / "TrackID3x3"

    if repo_dir.exists():
        console.print("[yellow]Repository already cloned. Pulling latest...[/yellow]")
        subprocess.run(["git", "-C", str(repo_dir), "pull"], check=True)
    else:
        console.print(f"Cloning {repo_url}...")
        subprocess.run(["git", "clone", repo_url, str(repo_dir)], check=True)

    console.print("[green]TrackID3x3 repository ready.[/green]")
    console.print("Follow the dataset README for video download instructions:")
    console.print(f"  {repo_dir}/README.md")


@cli.command()
@click.option("--output", "-o", default=str(DATA_DIR / "raw" / "nba_pbp"), help="Output directory")
@click.option("--seasons", "-s", default="2023-24", help="Comma-separated season(s)")
@click.option("--max-games", "-n", default=10, type=int, help="Max games to download")
def nba_clips(output: str, seasons: str, max_games: int) -> None:
    """Download NBA play-by-play video clips via nba_api.

    Uses: https://github.com/alijkhalil/nba_pbp_video_dataset
    """
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold]NBA Play-by-Play Clips[/bold]")
    console.print(f"Seasons: {seasons}")
    console.print(f"Max games: {max_games}")
    console.print(f"Output: {out_dir}")

    try:
        from nba_api.stats.endpoints import leaguegamefinder
        from nba_api.stats.endpoints import videoeventsasset

        console.print("nba_api available. Fetching game list...")

        # Get recent games
        finder = leaguegamefinder.LeagueGameFinder(
            season_nullable=seasons.split(",")[0],
            league_id_nullable="00",
        )
        games_df = finder.get_data_frames()[0]
        game_ids = games_df["GAME_ID"].unique()[:max_games]

        console.print(f"Found {len(game_ids)} games to process")

        for gid in game_ids:
            game_dir = out_dir / gid
            game_dir.mkdir(exist_ok=True)
            console.print(f"  Processing game {gid}...")
            # Note: actual clip download requires parsing video URLs from the API
            # This is a scaffold — full implementation would iterate play-by-play events

        console.print(f"[green]Game directories created in {out_dir}[/green]")

    except ImportError:
        console.print("[yellow]nba_api not installed. Install with: pip install nba_api[/yellow]")
        console.print("Falling back to pbp video dataset repository...")

        repo_url = "https://github.com/alijkhalil/nba_pbp_video_dataset.git"
        repo_dir = out_dir / "nba_pbp_video_dataset"

        if not repo_dir.exists():
            subprocess.run(["git", "clone", repo_url, str(repo_dir)], check=True)

        console.print(f"[green]Repository cloned to {repo_dir}[/green]")
        console.print("Follow the README for clip download instructions.")


@cli.command()
@click.option("--output", "-o", default=str(DATA_DIR / "raw" / "sportvu"), help="Output directory")
def sportvu(output: str) -> None:
    """Download SportVU 2015-16 tracking data (ground truth for validation).

    Source: https://github.com/sealneaward/nba-movement-data
    Also:   https://huggingface.co/datasets/dcayton/nba_tracking_data_15_16
    """
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold]SportVU Tracking Data (2015-16)[/bold]")
    console.print("XY coordinates for all 10 players + ball at 25fps")
    console.print(f"Output: {out_dir}")

    repo_url = "https://github.com/sealneaward/nba-movement-data.git"
    repo_dir = out_dir / "nba-movement-data"

    if repo_dir.exists():
        console.print("[yellow]Repository already exists.[/yellow]")
    else:
        console.print(f"Cloning {repo_url}...")
        subprocess.run(["git", "clone", repo_url, str(repo_dir)], check=True)

    console.print("[green]SportVU data repository ready.[/green]")
    console.print(f"JSON game files are in: {repo_dir}/data/")


@cli.command()
@click.option("--output", "-o", default=str(DATA_DIR / "raw" / "roboflow"), help="Output directory")
def roboflow_datasets(output: str) -> None:
    """Download Roboflow basketball detection datasets.

    - Player detection (654 images, YOLO format)
    - Jersey number OCR (3,615 images)
    """
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold]Roboflow Basketball Datasets[/bold]")
    console.print(f"Output: {out_dir}")
    console.print()
    console.print("To download, you need a Roboflow API key.")
    console.print("Set ROBOFLOW_API_KEY environment variable, then:")
    console.print()
    console.print("  Player detection:")
    console.print("    https://universe.roboflow.com/roboflow-jvuqo/basketball-player-detection-3-ycjdo")
    console.print()
    console.print("  Jersey number OCR:")
    console.print("    https://universe.roboflow.com/roboflow-jvuqo/basketball-jersey-numbers-ocr/dataset/5")
    console.print()

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if api_key:
        try:
            from roboflow import Roboflow
            rf = Roboflow(api_key=api_key)

            console.print("Downloading player detection dataset...")
            project = rf.workspace("roboflow-jvuqo").project("basketball-player-detection-3-ycjdo")
            project.version(1).download("yolov8", location=str(out_dir / "player_detection"))

            console.print("Downloading jersey number OCR dataset...")
            project = rf.workspace("roboflow-jvuqo").project("basketball-jersey-numbers-ocr")
            project.version(5).download("yolov8", location=str(out_dir / "jersey_ocr"))

            console.print("[green]Datasets downloaded successfully.[/green]")
        except ImportError:
            console.print("[yellow]roboflow package not installed. pip install roboflow[/yellow]")
    else:
        console.print("[yellow]ROBOFLOW_API_KEY not set. Skipping auto-download.[/yellow]")


@cli.command()
@click.argument("url")
@click.option("--output", "-o", default=str(DATA_DIR / "raw" / "videos"), help="Output directory")
@click.option("--name", "-n", default=None, help="Output filename")
def download_video(url: str, output: str, name: str | None) -> None:
    """Download a video from a URL (YouTube, archive.org, etc.)."""
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Downloading video:[/bold] {url}")
    console.print(f"Output: {out_dir}")

    try:
        cmd = ["yt-dlp", "-f", "best[height<=1080]", "-o", str(out_dir / (name or "%(title)s.%(ext)s")), url]
        subprocess.run(cmd, check=True)
        console.print("[green]Download complete.[/green]")
    except FileNotFoundError:
        console.print("[red]yt-dlp not found. Install with: pip install yt-dlp[/red]")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Download failed: {e}[/red]")


@cli.command()
def list_sources() -> None:
    """List all known data sources for basketball analytics."""
    table_data = [
        ("TrackID3x3", "Video + pose + tracking", "Academic", "trackid3x3"),
        ("NBA PBP Clips", "Play-by-play video clips", "NBA.com", "nba-clips"),
        ("SportVU 2015-16", "XY tracking data (25fps)", "Ground truth", "sportvu"),
        ("Roboflow Player Det.", "654 annotated images", "Detection training", "roboflow-datasets"),
        ("Roboflow Jersey OCR", "3,615 annotated images", "Jersey recognition", "roboflow-datasets"),
        ("Internet Archive", "Classic full games", "Full game analysis", "download-video"),
        ("basketball-video.com", "Modern game replays", "Full game analysis", "download-video"),
    ]

    from rich.table import Table
    table = Table(title="Available Data Sources")
    table.add_column("Source", style="cyan")
    table.add_column("Content")
    table.add_column("Use Case")
    table.add_column("Command", style="green")

    for name, content, use, cmd in table_data:
        table.add_row(name, content, use, cmd)

    console.print(table)


if __name__ == "__main__":
    cli()
