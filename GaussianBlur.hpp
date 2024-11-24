#pragma once
#include <vector>
#include <Eigen/Sparse>

Eigen::SparseMatrix<float> generateAMatrix(const unsigned char* inputImage, int width, int height, int kernelSize, float sigma);
std::vector<std::vector<float>> generateGaussianKernel(const int kernelSize, const float sigma);

