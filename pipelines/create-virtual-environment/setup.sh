#!/bin/bash

source activate earthquake
if [ $? -eq 0 ]; then
    conda create --name earthquake
fi

# Install required packages.
pip install -r requirements.txt
cd ../../pysmo
pip install -e .
