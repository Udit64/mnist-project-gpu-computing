# Compiler and flags
CUDA_COMPILER = nvcc
CXX_COMPILER = g++
CUDA_FLAGS = -std=c++14 -O3 --use_fast_math -I/usr/local/include/opencv4
CXX_FLAGS = -std=c++14 -O3
LINK_FLAGS = -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

# Directories
SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build
BIN = $(BUILD_DIR)/image_processor

# Source files
CUDA_SOURCES = $(SRC_DIR)/main.cu $(SRC_DIR)/image_processor.cu
HEADERS = $(INCLUDE_DIR)/image_processor.h

# Targets
all: build_dir $(BIN)

build_dir:
	mkdir -p $(BUILD_DIR)

$(BIN): $(CUDA_SOURCES) $(HEADERS)
	$(CUDA_COMPILER) $(CUDA_FLAGS) $(LINK_FLAGS) -o $@ $(CUDA_SOURCES)

clean:
	rm -rf $(BUILD_DIR)

run: all
	$(BIN) --input ./data/images --output ./data/output --mode blur --batch_size 32
