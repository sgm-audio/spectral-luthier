"""
Command-line interface for Spectral Neural Luthier
"""

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import track
from pathlib import Path
from loguru import logger
import sys

from .stem_master import StemMaster
from .profiles import list_profiles, get_profile

app = typer.Typer(
    name="spectral-luthier",
    help="Autonomous Stem-Aware Spectral Mastering with Dynamic DSP",
    add_completion=False,
)
console = Console()


@app.command()
def master(
    input_file: Path = typer.Argument(
        ...,
        help="Input stereo WAV file",
        exists=True,
        dir_okay=False,
    ),
    output_file: Path = typer.Option(
        "master.wav",
        "--output", "-o",
        help="Output master WAV file",
    ),
    profile: str = typer.Option(
        "Modern_Metal",
        "--profile", "-p",
        help="Target profile name",
    ),
    multiband: bool = typer.Option(
        False,
        "--multiband", "-m",
        help="Use multiband limiter (slower, more transparent)",
    ),
    export_stems: bool = typer.Option(
        False,
        "--export-stems",
        help="Export separated stems as individual files",
    ),
    device: str = typer.Option(
        "auto",
        "--device", "-d",
        help="Processing device: 'auto', 'cuda', or 'cpu'",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose logging",
    ),
):
    """
    Master an audio file using stem-aware spectral processing

    Example:
        spectral-luthier master song.wav -p Modern_Metal -o mastered.wav
    """
    # Configure logging
    logger.remove()
    log_level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=log_level,
    )

    # Display header
    console.print("\n[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]  SPECTRAL NEURAL LUTHIER - Stem-Aware Mastering[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]\n")

    # Validate profile
    available_profiles = list_profiles()
    if profile not in available_profiles:
        console.print(f"[red]Error: Profile '{profile}' not found[/red]")
        console.print(f"[yellow]Available profiles: {', '.join(available_profiles)}[/yellow]")
        raise typer.Exit(1)

    # Show profile info
    target_profile = get_profile(profile)
    console.print(f"[bold]Profile:[/bold] {target_profile.name}")
    console.print(f"[bold]Description:[/bold] {target_profile.description}")
    console.print(f"[bold]Target LUFS:[/bold] {target_profile.final_lufs_integrated:.1f}")
    console.print()

    # Initialize
    console.print("[yellow]Initializing Spectral Neural Luthier...[/yellow]")
    master_engine = StemMaster(device=device)

    # Process
    try:
        result = master_engine.process(
            input_path=str(input_file),
            profile=profile,
            output_path=str(output_file),
            multiband_limiting=multiband,
        )

        # Display results
        console.print("\n[bold green]✓ Mastering complete![/bold green]\n")

        # Results table
        table = Table(title="Processing Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Input File", result["input_file"])
        table.add_row("Output File", result["output_file"])
        table.add_row("Profile", result["profile"])
        table.add_row("Target LUFS", f"{result['final_lufs']:.1f}")
        table.add_row("Stems Processed", ", ".join(result["stems_analyzed"]))

        console.print(table)

        # Export stems if requested
        if export_stems:
            console.print("\n[yellow]Exporting separated stems...[/yellow]")
            stem_dir = output_file.parent / f"{output_file.stem}_stems"
            # Note: This would require re-running separation
            console.print(f"[dim]Stems would be exported to: {stem_dir}[/dim]")

        console.print(f"\n[bold green]Master saved to: {output_file}[/bold green]")

    except Exception as e:
        console.print(f"\n[bold red]Error during processing: {e}[/bold red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def profiles():
    """
    List available mastering profiles
    """
    console.print("\n[bold cyan]Available Mastering Profiles[/bold cyan]\n")

    available = list_profiles()

    for profile_name in available:
        profile = get_profile(profile_name)

        console.print(f"[bold]{profile.name}[/bold]")
        console.print(f"  {profile.description}")
        console.print(f"  Target LUFS: {profile.final_lufs_integrated:.1f}")
        console.print(f"  True Peak Ceiling: {profile.final_true_peak:.1f} dBTP")
        console.print()


@app.command()
def info():
    """
    Show system information and dependencies
    """
    import torch
    from platform import platform, python_version

    console.print("\n[bold cyan]System Information[/bold cyan]\n")

    table = Table()
    table.add_column("Component", style="cyan")
    table.add_column("Version/Status", style="green")

    table.add_row("Python", python_version())
    table.add_row("Platform", platform())
    table.add_row("PyTorch", torch.__version__)
    table.add_row("CUDA Available", "Yes" if torch.cuda.is_available() else "No")

    if torch.cuda.is_available():
        table.add_row("CUDA Version", torch.version.cuda or "N/A")
        table.add_row("GPU", torch.cuda.get_device_name(0))

    console.print(table)
    console.print()


if __name__ == "__main__":
    app()
