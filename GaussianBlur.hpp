#pragma once
#include <vector>

std::vector<std::vector<float>> generateGaussianKernel(const int kernelSize, const float sigma);
void applyGaussianBlur(const unsigned char *inputImage, unsigned char *outputImage, const int width, const int height, const int kernelSize, const float sigma);
