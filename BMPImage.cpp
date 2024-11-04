#include "BMPImage.hpp"
#include <fstream>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>

BMPImage::BMPImage(const char *path)
{
    const char *imagepath = path;
    unsigned char header[54];
    unsigned int dataPos;
    unsigned int imageSize;

    FILE *file = fopen(imagepath, "rb");
    if (!file)
    {
        printf("BMPImage could not be opened\n");
        return;
    }

    if (fread(header, 1, 54, file) != 54)
    {
        printf("Not a correct BMP file\n");
        fclose(file);
        return;
    }
    if (header[0] != 'B' || header[1] != 'M')
    {
        printf("Not a correct BMP file\n");
        fclose(file);
        return;
    }

    dataPos = *(int *)&(header[0x0A]);
    imageSize = *(int *)&(header[0x22]);
    width = *(int *)&(header[0x12]);
    height = *(int *)&(header[0x16]);

    if (imageSize == 0)
        imageSize = width * height * 3;
    if (dataPos == 0)
        dataPos = 54;

    data = new unsigned char[imageSize];
    const size_t ret_code = fread(data, 1, imageSize, file);
    if (ret_code != imageSize)
    {
        if (feof(file))
            printf("Error reading image data: unexpected end of file\n");
        else if (ferror(file))
            perror("Error reading image data");
        delete[] data;
        fclose(file);
        return;
    }

    unsigned char *r = new unsigned char[imageSize / 3];
    unsigned char *g = new unsigned char[imageSize / 3];
    unsigned char *b = new unsigned char[imageSize / 3];
    int n = 0;
    for (int i = 0; i < imageSize / 3; ++i)
    {
        r[i] = data[n++];
        g[i] = data[n++];
        b[i] = data[n++];
    }
    delete[] data;
    data = new unsigned char[imageSize / 3];
    for (int i = 0; i < imageSize / 3; ++i)
    {
        unsigned char lum = static_cast<unsigned char>(0.299 * r[i] + 0.587 * g[i] + 0.114 * b[i]);
        data[i] = lum;
    }
    delete[] r;
    delete[] g;
    delete[] b;

    fclose(file);
}

unsigned char *BMPImage::GetData()
{
    return data;
}

int BMPImage::GetWidth()
{
    return width;
}

int BMPImage::GetHeight()
{
    return height;
}

BMPImage::~BMPImage()
{
    delete[] data;
}

void BMPImage::SaveBMP(const char *filename)
{
    std::ofstream file(filename, std::ios::out | std::ios::binary);
    if (!file)
    {
        std::cerr << "Could not open file for writing: " << filename << std::endl;
        return;
    }

    unsigned char bmpFileHeader[14] = {
        'B', 'M',
        0, 0, 0, 0,
        0, 0,
        0, 0,
        54, 0, 0, 0};

    unsigned char bmpInfoHeader[40] = {
        40, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        1, 0,
        24, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0};

    int bytesPerPixel = 3;
    int paddedRowSize = (width * bytesPerPixel + 3) & (~3);
    int imageSize = paddedRowSize * height;
    int fileSize = 54 + imageSize;

    bmpFileHeader[2] = (unsigned char)(fileSize);
    bmpFileHeader[3] = (unsigned char)(fileSize >> 8);
    bmpFileHeader[4] = (unsigned char)(fileSize >> 16);
    bmpFileHeader[5] = (unsigned char)(fileSize >> 24);

    bmpInfoHeader[4] = (unsigned char)(width);
    bmpInfoHeader[5] = (unsigned char)(width >> 8);
    bmpInfoHeader[6] = (unsigned char)(width >> 16);
    bmpInfoHeader[7] = (unsigned char)(width >> 24);

    bmpInfoHeader[8] = (unsigned char)(height);
    bmpInfoHeader[9] = (unsigned char)(height >> 8);
    bmpInfoHeader[10] = (unsigned char)(height >> 16);
    bmpInfoHeader[11] = (unsigned char)(height >> 24);

    file.write(reinterpret_cast<char *>(bmpFileHeader), sizeof(bmpFileHeader));
    file.write(reinterpret_cast<char *>(bmpInfoHeader), sizeof(bmpInfoHeader));

    unsigned char padding[3] = {0, 0, 0};
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            unsigned char grayscaleValue = data[y * width + x];
            unsigned char rgb[3] = {grayscaleValue, grayscaleValue, grayscaleValue};
            file.write(reinterpret_cast<char *>(rgb), bytesPerPixel);
        }
        file.write(reinterpret_cast<char *>(padding), (paddedRowSize - width * bytesPerPixel));
    }

    file.close();

    if (file)
    {
        std::cout << "Image saved successfully" << std::endl;
    }
    else
    {
        std::cerr << "Error while saving image" << std::endl;
    }
}