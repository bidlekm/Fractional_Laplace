#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>

#include "BMPImage.hpp"
#include "GaussianBlur.hpp"

int main(int argc, char **argv)
{
    auto image = std::make_shared<BMPImage>("lena.bmp");
    int width = image->GetWidth();
    int height = image->GetHeight();
    unsigned char *imageData = image->GetData();

    auto outputImage = std::shared_ptr<unsigned char>(new unsigned char[width * height], std::default_delete<unsigned char[]>());

    int kernelSize = 15;
    float sigma = 10.0;

    applyGaussianBlur(imageData, outputImage.get(), width, height, kernelSize, sigma);

    BMPImage output(width, height, outputImage.get());
    output.SaveBMP("lena_output.bmp");
    return 0;
}