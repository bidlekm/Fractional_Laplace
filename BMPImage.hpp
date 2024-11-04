#pragma once

#include <fstream>
#include <iostream>
#include <algorithm>

class BMPImage
{
    int width, height;
    unsigned char *data;

public:
    BMPImage(const char *path);
    BMPImage(int width, int height, unsigned char *data) : width(width), height(height)
    {
        this->data = new unsigned char[width * height];
        for (int i = 0; i < width * height; ++i)
        {
            this->data[i] = data[i];
        }
    }

    unsigned char *GetData();
    int GetWidth();
    int GetHeight();

    void SaveBMP(const char *filename);

    ~BMPImage();
};