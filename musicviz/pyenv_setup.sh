#!/bin/bash

# Setup pyenv environment
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"

# Initialize pyenv
eval "$(pyenv init -)"

# Initialize pyenv-virtualenv if available
if pyenv commands | grep -q virtualenv-init; then
  eval "$(pyenv virtualenv-init -)"
fi

echo "pyenv environment configured successfully."
echo "Run 'source pyenv_setup.sh' to load pyenv in your current shell."
echo "To install Python 3.10.12, run: pyenv install 3.10.12"
echo "To create a virtualenv, run: pyenv virtualenv 3.10.12 animusicator"
echo "To activate, run: pyenv activate animusicator or pyenv local animusicator"
