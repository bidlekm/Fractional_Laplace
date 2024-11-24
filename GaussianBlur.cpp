#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include "GaussianBlur.hpp"


std::vector<std::vector<float>> generateGaussianKernel(const int kernelSize, const float sigma)
{
    std::vector<std::vector<float>> kernel(kernelSize, std::vector<float>(kernelSize));
    float sum = 0.0;
    int half = kernelSize / 2;
    float pi = 3.14159265359;

    for (int i = 0; i < kernelSize; ++i)
    {
        for (int j = 0; j < kernelSize; ++j)
        {
            int x = i - half;
            int y = j - half;
            kernel[i][j] = (1.0 / (2.0 * pi * sigma * sigma)) * exp(-(x * x + y * y) / (2 * sigma * sigma));
            sum += kernel[i][j];
        }
    }
    for (int i = 0; i < kernelSize; ++i)
    {
        for (int j = 0; j < kernelSize; ++j)
        {
            kernel[i][j] /= sum;
        }
    }

    return kernel;
}

Eigen::SparseMatrix<float> generateAMatrix(const unsigned char* inputImage, int width, int height, int kernelSize, float sigma)
{
    std::vector<std::vector<float>> kernel = generateGaussianKernel(kernelSize, sigma);
    int half = kernelSize / 2;
    int n = width * height;
    
    Eigen::SparseMatrix<float> A(n, n);

    std::vector<Eigen::Triplet<float>> tripletList;
    
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int idx = y * width + x;  // Flattened pixel index
            
            // Apply the kernel to the pixel and its neighbors
            for (int ky = -half; ky <= half; ++ky)
            {
                for (int kx = -half; kx <= half; ++kx)
                {
                    int pixelX = std::min(std::max(x + kx, 0), width - 1);
                    int pixelY = std::min(std::max(y + ky, 0), height - 1);
                    int neighborIdx = pixelY * width + pixelX; 
                    

                    float kernelValue = kernel[ky + half][kx + half];
                    
                    tripletList.push_back(Eigen::Triplet<float>(idx, neighborIdx, kernelValue));
                }
            }
        }
    }
    
    A.setFromTriplets(tripletList.begin(), tripletList.end());

    return A;
}