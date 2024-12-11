#include "image_processor.h"
#include <cuda_runtime.h>
#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

std::vector<std::string> LoadImages(const std::string& input_dir) {
    std::vector<std::string> image_paths;
    for (const auto& entry : fs::directory_iterator(input_dir)) {
        if (entry.is_regular_file()) {
            image_paths.push_back(entry.path().string());
        }
    }
    return image_paths;
}

void SaveImage(const std::string& output_path, const unsigned char* data, int width, int height) {
    cv::Mat img(height, width, CV_8UC1, (void*)data);
    cv::imwrite(output_path, img);
}

__global__ void GaussianBlurKernel(const unsigned char* input, unsigned char* output, int width, int height, int kernel_radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float blur = 0.0f;
        int count = 0;

        for (int ky = -kernel_radius; ky <= kernel_radius; ++ky) {
            for (int kx = -kernel_radius; kx <= kernel_radius; ++kx) {
                int nx = x + kx;
                int ny = y + ky;

                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    blur += input[ny * width + nx];
                    ++count;
                }
            }
        }

        output[y * width + x] = static_cast<unsigned char>(blur / count);
    }
}
