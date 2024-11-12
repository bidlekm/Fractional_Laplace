#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include "BMPImage.hpp"
#include "GaussianBlur.hpp" // Include your GaussianBlur implementation

void solveMinimizationProblem(std::vector<float> &x, const std::vector<float> &b, const std::vector<float> &laplacian, float mu, int width, int height)
{
    float learningRate = 0.001f;
    int maxIterations = 100;

    for (int iter = 0; iter < maxIterations; ++iter)
    {

        for (int i = 0; i < width * height; ++i)
        {
            float gradient = (x[i] - b[i]) + mu * laplacian[i] * x[i];

            float newX = x[i] - learningRate * gradient;
            x[i] = newX;
        }
    }
}

void applyLaplacian(const unsigned char *input, std::vector<float> &output, int width, int height)
{
    const int kernel[3][3] = {
        {0, 1, 0},
        {1, -4, 1},
        {0, 1, 0}};

    for (int y = 1; y < height - 1; ++y)
    {
        for (int x = 1; x < width - 1; ++x)
        {
            float sum = 0.0f;
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

int main()
{
    auto image = std::make_shared<BMPImage>("lena.bmp");
    int width = image->GetWidth();
    int height = image->GetHeight();
    int kernelSize = 7;
    float sigma = 5.0f;
    auto blurredImage = std::unique_ptr<unsigned char[]>(new unsigned char[width * height]);
    applyGaussianBlur(image->GetData(), blurredImage.get(), width, height, kernelSize, sigma);

    BMPImage blurred(width, height, blurredImage.get());
    blurred.SaveBMP("lena_blurred.bmp");

    std::vector<float> laplacian(width * height, 0.0f);
    applyLaplacian(blurredImage.get(), laplacian, width, height);

    std::vector<float> b(width * height, 0.0f);
    for (int i = 0; i < width * height; ++i)
    {
        b[i] = static_cast<float>(blurredImage[i]);
    }

    std::vector<float> output(width * height, 0.0f);
    float mu = 1.0f;
    solveMinimizationProblem(output, b, laplacian, mu, width, height);

    auto outputImage = std::unique_ptr<unsigned char[]>(new unsigned char[width * height]);

    float minVal = *std::min_element(output.begin(), output.end());
    float maxVal = *std::max_element(output.begin(), output.end());

    for (int i = 0; i < width * height; ++i)
    {
        outputImage[i] = static_cast<unsigned char>(std::min(std::max(int((output[i] - minVal) / (maxVal - minVal) * 255), 0), 255));
    }

    BMPImage result(width, height, outputImage.get());
    result.SaveBMP("lena_output.bmp");
    return 0;
}
