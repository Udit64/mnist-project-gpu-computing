#include "image_processor.h"
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// Global log file
std::ofstream log_file;

// Function to initialize logging
void InitializeLogging(const std::string& log_path) {
    log_file.open(log_path, std::ios::out | std::ios::trunc);
    if (!log_file.is_open()) {
        std::cerr << "[ERROR] Unable to open log file: " << log_path << std::endl;
        exit(1);
    }
}

// Log function
void Log(const std::string& message) {
    if (log_file.is_open()) {
        log_file << message << std::endl;
    }
    std::cout << message << std::endl;
}

// Function to parse command-line arguments
void ParseArguments(int argc, char** argv, std::string& input_dir, std::string& output_dir, std::string& mode, int& batch_size) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) {
            input_dir = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--mode" && i + 1 < argc) {
            mode = argv[++i];
        } else if (arg == "--batch_size" && i + 1 < argc) {
            batch_size = std::stoi(argv[++i]);
        } else {
            Log("[ERROR] Unknown or incomplete argument: " + arg);
            exit(1);
        }
    }

    if (input_dir.empty() || output_dir.empty() || mode.empty() || batch_size <= 0) {
        Log("[ERROR] Missing required arguments. Usage: ");
        Log("./image_processor --input <input_dir> --output <output_dir> --mode <mode> --batch_size <batch_size>");
        exit(1);
    }
}

// Main function
int main(int argc, char** argv) {
    std::string input_dir, output_dir, mode;
    int batch_size = 0;

    // Initialize logging
    InitializeLogging("execution.log");
    Log("[INFO] Starting image processing tool.");

    // Parse arguments
    ParseArguments(argc, argv, input_dir, output_dir, mode, batch_size);

    // Log the configuration
    Log("[INFO] Input directory: " + input_dir);
    Log("[INFO] Output directory: " + output_dir);
    Log("[INFO] Processing mode: " + mode);
    Log("[INFO] Batch size: " + std::to_string(batch_size));

    // Load images
    auto images = LoadImages(input_dir);
    if (images.empty()) {
        Log("[ERROR] No images found in input directory: " + input_dir);
        return 1;
    }
    Log("[INFO] Found " + std::to_string(images.size()) + " images in the input directory.");

    // Process each image
    for (const auto& image_path : images) {
        Log("[INFO] Processing image: " + image_path);

        cv::Mat input_image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
        if (input_image.empty()) {
            Log("[ERROR] Failed to load image: " + image_path);
            continue;
        }

        int width = input_image.cols;
        int height = input_image.rows;
        size_t image_size = width * height * sizeof(unsigned char);

        Log("[INFO] Image dimensions: " + std::to_string(width) + "x" + std::to_string(height));

        // Allocate memory on the GPU
        unsigned char* d_input;
        unsigned char* d_output;
        cudaMalloc(&d_input, image_size);
        cudaMalloc(&d_output, image_size);

        // Copy image data to the GPU
        cudaMemcpy(d_input, input_image.data, image_size, cudaMemcpyHostToDevice);

        // Define grid and block dimensions
        dim3 threads(16, 16);
        dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

        // Launch the Gaussian blur kernel
        GaussianBlurKernel<<<blocks, threads>>>(d_input, d_output, width, height, 3);

        // Copy the result back to the host
        unsigned char* output_image = new unsigned char[width * height];
        cudaMemcpy(output_image, d_output, image_size, cudaMemcpyDeviceToHost);

        // Save the processed image
        std::string output_path = output_dir + "/" + fs::path(image_path).filename().string();
        SaveImage(output_path, output_image, width, height);
        Log("[INFO] Processed image saved to: " + output_path);

        // Cleanup
        delete[] output_image;
        cudaFree(d_input);
        cudaFree(d_output);
    }

    Log("[INFO] All images processed successfully.");
    Log("[INFO] Program completed.");

    // Close the log file
    if (log_file.is_open()) {
        log_file.close();
    }

    return 0;
}
