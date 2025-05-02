#!/usr/bin/env python3
"""
Launcher script for the Animusicator application.
This script serves as a convenient entry point to start the application.
"""

import sys
import os
import traceback

# Add the src directory to the Python path 
# This makes the musicviz module importable
src_path = os.path.join(os.path.dirname(__file__), 'musicviz', 'src')
sys.path.insert(0, src_path)

def run():
    """Launch the Animusicator application."""
    try:
        print("Starting Animusicator application...")
        from musicviz.main import main
        print("Main module imported successfully")
        exit_code = main()
        print(f"Application returned exit code: {exit_code}")
        return exit_code
    except ImportError as e:
        print(f"Error importing the application: {e}")
        print("Make sure all dependencies are installed by running:")
        print("  pip install -r requirements.txt")
        return 1
    except Exception as e:
        print(f"Unexpected error running the application: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(run()) 