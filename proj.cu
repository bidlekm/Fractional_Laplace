#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <algorithm>
#include "BMPImage.hpp"
#include "GaussianBlur.hpp"

#define AssertCuda(error_code)                                           \
    if (error_code != cudaSuccess)                                       \
    {                                                                    \
        std::cout << "The cuda call in " << __FILE__ << " on line "      \
                  << __LINE__ << " resulted in the error '"              \
                  << cudaGetErrorString(error_code) << "'" << std::endl; \
        std::abort();                                                    \
    }


template <typename Number>
void applyLaplacian(const unsigned char *input, std::vector<Number> &output, int width, int height)
{
    const int kernel[3][3] = {
        {0, 1, 0},
        {1, -4, 1},
        {0, 1, 0}};

    for (int y = 1; y < height - 1; ++y)
    {
        for (int x = 1; x < width - 1; ++x)
        {
            Number sum = 0.0f;
            for (int ky = -1; ky <= 1; ++ky)
            {
                for (int kx = -1; kx <= 1; ++kx)
                {
                    sum += input[(y + ky) * width + (x + kx)] * kernel[ky + 1][kx + 1];
                }
            }
            output[y * width + x] = sum;
        }
    }
}

template <typename Number>
__global__ void gradDescCuda(Number *x, const Number *b, const Number *laplacian, Number mu, Number learningRate, int width, int height, int maxIterations)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = width * height;

    if (idx < totalSize)
    {
        for (int iter = 0; iter < maxIterations; ++iter)
        {
            Number gradient = (x[idx] - b[idx]) + mu * laplacian[idx] * x[idx];
            x[idx] -= learningRate * gradient;
        }
    }
}

template <typename Number>
void solveMinCuda(std::vector<Number> &x, const std::vector<Number> &b, const std::vector<Number> &laplacian, Number mu, int width, int height)
{
    int totalSize = width * height;
    Number learningRate = 0.001f;
    int maxIterations = 100;
    int blockSize = 256;
    int numBlocks = (totalSize + blockSize - 1) / blockSize;


    AssertCuda(cudaMalloc(&d_x, totalSize * sizeof(Number)));
    AssertCuda(cudaMalloc(&d_b, totalSize * sizeof(Number)));
    AssertCuda(cudaMalloc(&d_laplacian, totalSize * sizeof(Number)));

    AssertCuda(cudaMemcpy(d_x, x.data(), totalSize * sizeof(Number), cudaMemcpyHostToDevice));
    AssertCuda(cudaMemcpy(d_b, b.data(), totalSize * sizeof(Number), cudaMemcpyHostToDevice));
    AssertCuda(cudaMemcpy(d_laplacian, laplacian.data(), totalSize * sizeof(Number), cudaMemcpyHostToDevice));

    gradDescCuda<<<numBlocks, blockSize>>>(d_x, d_b, d_laplacian, mu, learningRate, width, height, maxIterations);
    AssertCuda(cudaGetLastError());

    AssertCuda(cudaMemcpy(x.data(), d_x, totalSize * sizeof(Number), cudaMemcpyDeviceToHost));

    AssertCuda(cudaFree(d_x));
    AssertCuda(cudaFree(d_b));
    AssertCuda(cudaFree(d_laplacian));
}

int main()
{
    using Number = float;

    auto image = std::make_shared<BMPImage>("lena.bmp");
    int width = image->GetWidth();
    int height = image->GetHeight();
    int kernelSize = 7;
    Number sigma = 5.0f;
    auto blurredImage = std::unique_ptr<unsigned char[]>(new unsigned char[width * height]);
    applyGaussianBlur(image->GetData(), blurredImage.get(), width, height, kernelSize, sigma);

    BMPImage blurred(width, height, blurredImage.get());
    blurred.SaveBMP("lena_blurred.bmp");

    std::vector<Number> laplacian(width * height, 0.0f);
    applyLaplacian(blurredImage.get(), laplacian, width, height);

    std::vector<Number> b(width * height, 0.0f);
    for (int i = 0; i < width * height; ++i)
    {
        b[i] = static_cast<Number>(blurredImage[i]);
    }

    std::vector<Number> output(width * height, 0.0f);
    Number mu = 1.0f;

    solveMinCuda(output, b, laplacian, mu, width, height);

    auto outputImage = std::unique_ptr<unsigned char[]>(new unsigned char[width * height]);
    Number minVal = *std::min_element(output.begin(), output.end());
    Number maxVal = *std::max_element(output.begin(), output.end());
    std::cout << "Min value in output: " << minVal << std::endl;
    std::cout << "Max value in output: " << maxVal << std::endl;
    for (int i = 0; i < width * height; ++i)
    {
        outputImage[i] = static_cast<unsigned char>(std::min(std::max(int((output[i] - minVal) / (maxVal - minVal) * 255), 0), 255));
    }

    BMPImage result(width, height, outputImage.get());
    result.SaveBMP("lena_output_cuda.bmp");

    return 0;
}
