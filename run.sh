#!/bin/bash
mkdir -p build
cd build
cmake ..
make
cd ..
./build/image_processor --input ./data/images --output ./data/output --mode blur --batch_size 32
