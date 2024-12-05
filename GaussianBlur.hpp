#pragma once
#include <vector>
#include <Eigen/Sparse>
template <typename Number>
Eigen::SparseMatrix<Number> generateAMatrix(const unsigned char* inputImage, int width, int height, int kernelSize, Number sigma);

template <typename Number>
std::vector<std::vector<Number>> generateGaussianKernel(const int kernelSize, const Number sigma);

