"""CLI interface for SportsSight."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from .config import load_config

console = Console()


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def main(verbose: bool) -> None:
    """SportsSight — Real-time sports fatigue analytics."""
    setup_logging(verbose)


@main.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option("--game-id", "-g", default="test-game", help="Game identifier")
@click.option("--output", "-o", type=click.Path(), help="Output JSON path")
@click.option("--annotate/--no-annotate", default=True, help="Output annotated video with bounding boxes")
def analyze(video_path: str, game_id: str, output: str | None, annotate: bool) -> None:
    """Analyze a recorded game video for fatigue patterns."""
    import json
    from .ingestion.sources import FileSource
    from .realtime.engine import RealtimeEngine

    config = load_config()
    engine = RealtimeEngine(config)

    source = FileSource(video_path)
    console.print(f"[bold]Analyzing:[/bold] {video_path}")
    console.print(f"[bold]Game ID:[/bold] {game_id}")
    console.print(f"[bold]Device:[/bold] {config.device}")
    console.print(f"[bold]Annotate:[/bold] {annotate}")

    # Determine annotated video output path
    annotated_path = None
    if annotate:
        base = Path(video_path)
        annotated_path = str(base.parent / f"{base.stem}_annotated.mp4")
        console.print(f"[bold]Annotated video:[/bold] {annotated_path}")

    async def run():
        await engine.initialize(require_redis=False)
        results = await engine.process_recorded(
            game_id, source, annotated_video_path=annotated_path,
        )
        return results

    results = asyncio.run(run())

    if output:
        with open(output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        console.print(f"[green]Results written to {output}[/green]")
    else:
        # Print summary
        timeline = results.get("timeline", [])
        if timeline:
            last = timeline[-1]
            table = Table(title=f"Final Fatigue Scores — {game_id}")
            table.add_column("Player ID", style="cyan")
            table.add_column("Score", style="bold")
            table.add_column("Level", style="yellow")
            table.add_column("Trend")

            for pid, score in last.get("scores", {}).items():
                level = score.get("level", "?")
                color = {"low": "green", "moderate": "yellow", "high": "red", "critical": "bold red"}.get(level, "")
                table.add_row(
                    str(pid),
                    f"{score.get('score', 0):.0f}",
                    f"[{color}]{level}[/{color}]",
                    score.get("trend", "?"),
                )
            console.print(table)


@main.command()
@click.argument("stream_url")
@click.option("--game-id", "-g", required=True, help="Game identifier")
@click.option("--source-type", "-t", type=click.Choice(["rtmp", "rtsp", "hls"]), default="rtmp")
def live(stream_url: str, game_id: str, source_type: str) -> None:
    """Process a live game stream in real-time."""
    from .ingestion.sources import RTMPSource, RTSPSource

    config = load_config()

    if source_type == "rtsp":
        source = RTSPSource(stream_url)
    else:
        source = RTMPSource(stream_url)

    engine_module = __import__("src.realtime.engine", fromlist=["RealtimeEngine"])
    engine = engine_module.RealtimeEngine(config)

    console.print(f"[bold]Live game:[/bold] {game_id}")
    console.print(f"[bold]Stream:[/bold] {stream_url}")
    console.print(f"[bold]Device:[/bold] {config.device}")

    async def run():
        await engine.initialize()
        await engine.start_game(game_id, source)

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped by user[/yellow]")


@main.command()
@click.option("--host", default="0.0.0.0")
@click.option("--port", default=8000, type=int)
def serve(host: str, port: int) -> None:
    """Start the API server."""
    import uvicorn
    console.print(f"[bold]Starting API server on {host}:{port}[/bold]")
    uvicorn.run("src.api.app:app", host=host, port=port, reload=True)


@main.command()
def download_models() -> None:
    """Download required ML models."""
    from .config import DATA_DIR

    model_dir = DATA_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold]Downloading models...[/bold]")

    # YOLOv8 detection model
    try:
        from ultralytics import YOLO
        console.print("  Downloading YOLOv8x (detection)...")
        YOLO("yolov8x.pt")
        console.print("  [green]YOLOv8x ready[/green]")
    except Exception as e:
        console.print(f"  [red]YOLOv8x failed: {e}[/red]")

    # YOLOv8 pose model
    try:
        from ultralytics import YOLO
        console.print("  Downloading YOLOv8x-Pose...")
        YOLO("yolov8x-pose.pt")
        console.print("  [green]YOLOv8x-Pose ready[/green]")
    except Exception as e:
        console.print(f"  [red]YOLOv8x-Pose failed: {e}[/red]")

    console.print("[green]Model download complete[/green]")


@main.command()
def info() -> None:
    """Show system information and configuration."""
    import torch

    config = load_config()

    table = Table(title="SportsSight System Info")
    table.add_column("Setting", style="cyan")
    table.add_column("Value")

    table.add_row("Device", config.device)
    table.add_row("PyTorch", torch.__version__)
    table.add_row("CUDA available", str(torch.cuda.is_available()))
    table.add_row("MPS available", str(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()))
    table.add_row("Database", config.database_url[:50] + "..." if len(config.database_url) > 50 else config.database_url)
    table.add_row("Redis", config.redis_url)
    table.add_row("Inference FPS", str(config.pipeline.get("inference_fps", 15)))

    console.print(table)


if __name__ == "__main__":
    main()
