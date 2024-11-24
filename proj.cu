#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <algorithm>
#include "BMPImage.hpp"
#include "GaussianBlur.hpp"
#include <Eigen/Sparse>

#define AssertCuda(error_code)                                           \
    if (error_code != cudaSuccess)                                       \
    {                                                                    \
        std::cout << "The cuda call in " << __FILE__ << " on line "      \
                  << __LINE__ << " resulted in the error '"              \
                  << cudaGetErrorString(error_code) << "'" << std::endl; \
        std::abort();                                                    \
    }

template <typename Number>
void generateLMatrix(Eigen::SparseMatrix<Number> &L, int width, int height)
{
    int n = width * height;
    L.resize(2 * n, n);
    std::vector<Eigen::Triplet<Number>> tripletList;
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int idx = y * width + x;
            if (x < width - 1)
            {
                tripletList.emplace_back(idx, idx, -1);   
                tripletList.emplace_back(idx, idx + 1, 1);
            }
        }
    }
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int idx = y * width + x;
            int gradYRow = n + idx; 

            if (y < height - 1)
            {
                tripletList.emplace_back(gradYRow, idx, -1);        
                tripletList.emplace_back(gradYRow, idx + width, 1); 
            }
        }
    }
    L.setFromTriplets(tripletList.begin(), tripletList.end());
}

template <typename Number>
void printEigenSparseMatrix(const Eigen::SparseMatrix<Number> &L)
{
    std::cout << "L: " << L.rows() << " x " << L.cols() << ")\n";
    for (int k = 0; k < L.outerSize(); ++k)
    {
        for (typename Eigen::SparseMatrix<Number>::InnerIterator it(L, k); it; ++it)
        {
            std::cout << "(" << it.row() << ", " << it.col() << ") = " << it.value() << "\n";
        }
    }
}

template <typename Number>
__global__ void sparseMatVec(const Number *x, Number *dst,
                          const Number *l_val, const int *l_col, const int *l_row,
                          int size)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < size)
    {
        Number sum = 0;
        for (int i = l_row[row]; i < l_row[row + 1]; ++i)
        {
            sum += l_val[i] * x[l_col[i]];
        }
        dst[row] = sum;
    }
}

template <typename Number>
__global__ void gradDescCuda(Number *x, const Number *b,
                             const Number *l_val, const int *l_col, const int *l_row,
                             const Number *lx, Number mu, Number learningRate,
                             int width, int height, int maxIterations)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = width * height;

    if (idx < totalSize)
    {
        for (int iter = 0; iter < maxIterations; ++iter)
        {
            Number gradient = 2 * (x[idx] - b[idx]);

            int startIdx = l_row[idx];
            int endIdx = l_row[idx + 1];
            for (int i = startIdx; i < endIdx; ++i)
            {
                int colIdx = l_col[i];
                gradient += 2 * mu * l_val[i] * lx[colIdx];
            }

            x[idx] -= learningRate * gradient;
        }
    }
}

template <typename Number>
void solveMinCuda(std::vector<Number> &x, const std::vector<Number> &b, const Eigen::SparseMatrix<Number> &L, int width, int height)
{
    int totalSize = width * height;
    int blockSize = 256;
    int numBlocks = (totalSize + blockSize - 1) / blockSize;


    
    Number learningRate = static_cast<Number>(0.1f);
    int maxIterations = 100;
    Number mu = static_cast<Number>(5.0f);
    


    Number *d_x, *d_b;
    AssertCuda(cudaMalloc(&d_x, totalSize * sizeof(Number)));
    AssertCuda(cudaMalloc(&d_b, totalSize * sizeof(Number)));

    AssertCuda(cudaMemcpy(d_x, x.data(), totalSize * sizeof(Number), cudaMemcpyHostToDevice));
    AssertCuda(cudaMemcpy(d_b, b.data(), totalSize * sizeof(Number), cudaMemcpyHostToDevice));

    Number *d_val;
    int *d_col, *d_rowIdx;
    int numNonZeros = L.nonZeros();

    AssertCuda(cudaMalloc(&d_val, numNonZeros * sizeof(Number)));
    AssertCuda(cudaMalloc(&d_col, numNonZeros * sizeof(int)));
    AssertCuda(cudaMalloc(&d_rowIdx, (L.outerSize() + 1) * sizeof(int)));

    AssertCuda(cudaMemcpy(d_val, L.valuePtr(), numNonZeros * sizeof(Number), cudaMemcpyHostToDevice));
    AssertCuda(cudaMemcpy(d_col, L.innerIndexPtr(), numNonZeros * sizeof(int), cudaMemcpyHostToDevice));
    AssertCuda(cudaMemcpy(d_rowIdx, L.outerIndexPtr(), (L.outerSize() + 1) * sizeof(int), cudaMemcpyHostToDevice));

    Number *d_Lx;
    AssertCuda(cudaMalloc(&d_Lx, 2 * totalSize * sizeof(Number)));

    int numBlocks2 = (2 * totalSize + blockSize - 1) / blockSize;
    sparseMatVec<<<numBlocks2, blockSize>>>(d_x, d_Lx, d_val, d_col, d_rowIdx, 2 * totalSize);
    AssertCuda(cudaGetLastError());

    gradDescCuda<<<numBlocks, blockSize>>>(d_x, d_b, d_val, d_col, d_rowIdx, d_Lx, mu, learningRate, width, height, maxIterations);
    AssertCuda(cudaGetLastError());


    AssertCuda(cudaMemcpy(x.data(), d_x, totalSize * sizeof(Number), cudaMemcpyDeviceToHost));

    AssertCuda(cudaFree(d_x));
    AssertCuda(cudaFree(d_b));
    AssertCuda(cudaFree(d_Lx));
    AssertCuda(cudaFree(d_val));
    AssertCuda(cudaFree(d_col));
    AssertCuda(cudaFree(d_rowIdx));
}




template <typename Number>
void applyGaussianBlurCuda(const unsigned char* inputImage, unsigned char* outputImage, 
                            const int width, const int height, const int kernelSize, 
                            const float sigma, Eigen::SparseMatrix<Number>& A)
{
    int totalSize = width * height;
    int blockSize = 256;
    int numBlocks = (totalSize + blockSize - 1) / blockSize;


    Number* d_x;
    Number* d_y;

    cudaMalloc(&d_x, totalSize * sizeof(Number));
    cudaMalloc(&d_y, totalSize * sizeof(Number));

    Number* d_A_val;
    int* d_A_col, *d_A_row;
    int numNonZeros = A.nonZeros();

    AssertCuda(cudaMalloc(&d_A_val, numNonZeros * sizeof(Number)));
    AssertCuda(cudaMalloc(&d_A_col, numNonZeros * sizeof(int)));
    AssertCuda(cudaMalloc(&d_A_row, (A.outerSize() + 1) * sizeof(int)));

    AssertCuda(cudaMemcpy(d_A_val, A.valuePtr(), numNonZeros * sizeof(Number), cudaMemcpyHostToDevice));
    AssertCuda(cudaMemcpy(d_A_col, A.innerIndexPtr(), numNonZeros * sizeof(int), cudaMemcpyHostToDevice));
    AssertCuda(cudaMemcpy(d_A_row, A.outerIndexPtr(), (A.outerSize() + 1) * sizeof(int), cudaMemcpyHostToDevice));

    std::vector<Number> x(totalSize);
    for (int i = 0; i < totalSize; ++i)
    {
        x[i] = static_cast<Number>(inputImage[i]);
    }
    cudaMemcpy(d_x, x.data(), totalSize * sizeof(Number), cudaMemcpyHostToDevice);

    
    sparseMatVec<<<numBlocks, blockSize>>>(d_x, d_y, d_A_val, d_A_col, d_A_row, totalSize);
    cudaDeviceSynchronize();

    cudaMemcpy(x.data(), d_y, totalSize * sizeof(Number), cudaMemcpyDeviceToHost);

    Number minVal = *std::min_element(x.begin(), x.end());
    Number maxVal = *std::max_element(x.begin(), x.end());

    for (int i = 0; i < totalSize; ++i)
    {
        outputImage[i] = static_cast<unsigned char>(
            std::min(std::max(int((x[i] - minVal) / (maxVal - minVal) * 255), 0), 255)
        );
    }

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_A_val);
    cudaFree(d_A_col);
    cudaFree(d_A_row);
}


int main()
{
    using Number = float;

    auto image = std::make_shared<BMPImage>("lena.bmp");
    int width = image->GetWidth();
    int height = image->GetHeight();
    int kernelSize = 5;
    Number sigma = 3.0f;
    auto blurredImage = std::unique_ptr<unsigned char[]>(new unsigned char[width * height]);

    Eigen::SparseMatrix<float> A = generateAMatrix(image->GetData(), width, height, kernelSize, sigma);
    applyGaussianBlurCuda(image->GetData(), blurredImage.get(), width, height, kernelSize, sigma, A);

    BMPImage blurred(width, height, blurredImage.get());
    blurred.SaveBMP("lena_blurred.bmp");


    Eigen::SparseMatrix<Number> L;
    generateLMatrix(L, width,height);
    

    std::vector<Number> b(width * height, 0.0f);
    for (int i = 0; i < width * height; ++i)
    {
        b[i] = static_cast<Number>(blurredImage[i]);
    }

    std::vector<Number> output(width * height, 0.0f);

    solveMinCuda(output, b, L, width, height);

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