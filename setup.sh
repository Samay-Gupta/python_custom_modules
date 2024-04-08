#!/bin/bash

DIR_TO_ADD=$PWD
MODELS_ZIP_URL=https://github.com/Samay-Gupta/python_custom_modules/releases/download/computer_vision_models/models.zip

if [ -f "$HOME/.zshrc" ]; then
    echo "export PYTHONPATH=\"\${PYTHONPATH}:${DIR_TO_ADD}\"" >> $HOME/.zshrc
    echo "Updated .zshrc with PYTHONPATH"
elif [ -f "$HOME/.bashrc" ]; then
    echo "export PYTHONPATH=\"\${PYTHONPATH}:${DIR_TO_ADD}\"" >> $HOME/.bashrc
    echo "Updated .bashrc with PYTHONPATH"
else
    echo "No .bashrc or .zshrc found. Please manually update your shell configuration."
fi

if [ -d "$DIR_TO_ADD" ]; then
    cd "$DIR_TO_ADD"
    cd "storage"
    wget $MODELS_ZIP_URL
    
    if [ -f "models.zip" ]; then
        unzip models.zip
        echo "Unzipped models.zip in $DIR_TO_ADD"
    else
        echo "models.zip not found in $DIR_TO_ADD"
    fi
else
    echo "$DIR_TO_ADD does not exist."
fi