#!/usr/bin/env python3
"""
Simple test script to verify rich library functionality.
"""

import time
from rich.console import Console
from rich.table import Table
from rich.progress import track
from rich.panel import Panel
from rich.syntax import Syntax
from rich.traceback import install
from rich.tree import Tree
from rich import print as rprint

def test_rich():
    """Test the rich library's formatting and output capabilities."""
    # Install rich traceback handler
    install()
    
    # Create a console instance
    console = Console()
    
    # Print a styled title
    console.print("[bold blue]Rich Library Test[/bold blue]", justify="center")
    console.print()
    
    # Create and display a table
    table = Table(title="Animusicator Components")
    table.add_column("Component", style="cyan")
    table.add_column("Description", style="green")
    table.add_column("Status", justify="center")
    
    table.add_row("AudioEngine", "Processes real-time audio data", "[bold green]Complete[/bold green]")
    table.add_row("FeatureExtractor", "Extracts musical features", "[bold yellow]In Progress[/bold yellow]")
    table.add_row("VisualWidget", "Renders OpenGL visualizations", "[bold red]Pending[/bold red]")
    table.add_row("MainWindow", "Application UI", "[bold yellow]In Progress[/bold yellow]")
    
    console.print(table)
    console.print()
    
    # Display a progress bar
    console.print("[bold]Simulating audio processing...[/bold]")
    for step in track(range(100), description="Processing"):
        time.sleep(0.01)  # Simulate work
    console.print("[bold green]Processing complete![/bold green]")
    console.print()
    
    # Show a panel with code syntax highlighting
    code = '''
def process_audio(buffer):
    """Process audio buffer and extract features."""
    # Convert to mono if stereo
    if buffer.shape[1] > 1:
        buffer = buffer.mean(axis=1)
    
    # Extract features
    features = feature_extractor.process(buffer)
    return features
    '''
    
    syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
    console.print(Panel(syntax, title="Sample Code", border_style="green"))
    console.print()
    
    # Create a file structure tree
    tree = Tree("📁 musicviz", guide_style="bold bright_blue")
    src = tree.add("📁 src", guide_style="bright_blue")
    src_musicviz = src.add("📁 musicviz", guide_style="bright_blue")
    src_musicviz.add("📄 __init__.py")
    src_musicviz.add("📄 main.py")
    
    audio = src_musicviz.add("📁 audio", guide_style="bright_blue")
    audio.add("📄 __init__.py")
    audio.add("📄 engine.py")
    audio.add("📄 feature_extractor.py")
    
    visual = src_musicviz.add("📁 visual", guide_style="bright_blue")
    visual.add("📄 __init__.py")
    visual.add("📄 shaders.py")
    
    gui = src_musicviz.add("📁 gui", guide_style="bright_blue")
    gui.add("📄 __init__.py")
    gui.add("📄 main_window.py")
    gui.add("📄 visual_widget.py")
    
    console.print(tree)
    console.print()
    
    # Print some rich-formatted text
    rprint("[bold]Rich[/bold] makes it easy to add [italic green]color[/italic green] and [yellow]style[/yellow] to terminal output!")
    rprint("It can also show emojis 🎵 🎧 🎹 🎸 and other unicode characters.")
    
    # Deliberately cause an error to show traceback formatting
    try:
        1 / 0
    except Exception:
        console.print_exception()

if __name__ == "__main__":
    test_rich() 