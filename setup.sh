#!/bin/bash

DIR_TO_ADD=$PWD

if [ -f "$HOME/.zshrc" ]; then
    echo "export PYTHONPATH=\"\${PYTHONPATH}:${DIR_TO_ADD}\"" >> $HOME/.zshrc
    echo "Updated .zshrc with PYTHONPATH"
elif [ -f "$HOME/.bashrc" ]; then
    echo "export PYTHONPATH=\"\${PYTHONPATH}:${DIR_TO_ADD}\"" >> $HOME/.bashrc
    echo "Updated .bashrc with PYTHONPATH"
else
    echo "No .bashrc or .zshrc found. Please manually update your shell configuration."
fi
