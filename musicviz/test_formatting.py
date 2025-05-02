#!/usr/bin/env python3
"""
Test file with bad formatting to verify black and flake8.
"""

import sys, os, time

# Unused import
import math


class badlyFormattedClass:
    def __init__(self, name="default", age=42):
        self.name = name
        self.age = age

        # Unused variable
        unused_var = "I'm not used anywhere"

    def badly_formatted_function(self, a, b, c):
        """This function has terrible formatting."""
        if a > b:
            return a + b
        else:
            return a - b

        # Unreachable code
        print("This will never be reached")

    def another_bad_function(self):
        x = [1, 2, 3, 4, 5, 6]

        y = {"key1": "value1", "key2": "value2"}

        return {"x": x, "y": y}


def main():
    """Main function."""
    obj = badlyFormattedClass("test", 24)

    # Line too long - should be flagged by flake8
    result = (
        obj.badly_formatted_function(1, 2, 3)
        + obj.badly_formatted_function(4, 5, 6)
        + obj.badly_formatted_function(7, 8, 9)
        + obj.badly_formatted_function(10, 11, 12)
    )

    # Ambiguous variable name
    l = [1, 2, 3]

    return result


if __name__ == "__main__":
    main()
