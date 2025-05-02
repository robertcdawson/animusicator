#!/usr/bin/env python3
"""
Test runner script for Animusicator.

This script runs the test suite with proper coverage reporting.
"""
import os
import sys
import argparse
import subprocess
import time


def main():
    """Run the tests with the specified options."""
    parser = argparse.ArgumentParser(description="Run Animusicator tests")
    
    parser.add_argument(
        "--unit", action="store_true", 
        help="Run only unit tests"
    )
    parser.add_argument(
        "--integration", action="store_true", 
        help="Run only integration tests"
    )
    parser.add_argument(
        "--gpu", action="store_true", 
        help="Run GPU-dependent tests"
    )
    parser.add_argument(
        "--coverage", action="store_true", 
        help="Generate coverage report"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", 
        help="Increase verbosity"
    )
    parser.add_argument(
        "filter", nargs="?", default=None,
        help="Optional filter string to select specific tests"
    )
    
    args = parser.parse_args()
    
    # Build the pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add coverage if requested
    if args.coverage:
        cmd += ["--cov=src/musicviz", "--cov-report=term", "--cov-report=html"]
    
    # Add verbosity
    if args.verbose:
        cmd.append("-v")
    
    # Determine which tests to run
    if args.unit:
        cmd.append("-m unit")
    elif args.integration:
        cmd.append("-m integration")
    
    # Add GPU filter if specified
    if not args.gpu:
        cmd.append("-m not gpu")
    
    # Add any user filter
    if args.filter:
        cmd.append(args.filter)
    
    # Print the command
    cmd_str = " ".join(cmd)
    print(f"Running: {cmd_str}")
    
    # Run the tests
    start_time = time.time()
    result = subprocess.run(cmd)
    elapsed_time = time.time() - start_time
    
    # Report summary
    print(f"\nTest run completed in {elapsed_time:.2f} seconds")
    
    if result.returncode == 0:
        print("All tests passed!")
    else:
        print(f"Tests failed with exit code {result.returncode}")
        
    # Return the exit code from pytest
    return result.returncode


if __name__ == "__main__":
    sys.exit(main()) 