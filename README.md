# CUDA Image Processor

## Overview
A CUDA-based image processing tool that performs operations such as Gaussian blur on large datasets. The tool is designed for efficiency using GPU acceleration.

## Usage
```bash
# Compile the project
mkdir build && cd build
cmake ..
make

# Run the program
./image_processor --input ../data/images --output ../data/output --mode blur --batch_size 32
