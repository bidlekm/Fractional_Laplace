#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <algorithm>
#include "BMPImage.hpp"
#include "GaussianBlur.hpp"
#include <Eigen/Sparse>
#include <Eigen/SVD>
#include <Eigen/Dense>
#include <Eigen/QR>

#define AssertCuda(error_code)                                           \
    if (error_code != cudaSuccess)                                       \
    {                                                                    \
        std::cout << "The cuda call in " << __FILE__ << " on line "      \
                  << __LINE__ << " resulted in the error '"              \
                  << cudaGetErrorString(error_code) << "'" << std::endl; \
        std::abort();                                                    \
    }

template <typename T>
const T& clamp(const T& v, const T& lo, const T& hi) {
    return (v < lo) ? lo : (v > hi) ? hi : v;
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


// template <typename Number>
// void solveMinCuda(std::vector<Number> &x, const std::vector<Number> &b, const Eigen::SparseMatrix<Number> &L, int width, int height)
// {
//     int totalSize = width * height;
//     int blockSize = 256;
//     int numBlocks = (totalSize + blockSize - 1) / blockSize;


    
//     Number learningRate = static_cast<Number>(0.1f);
//     int maxIterations = 100;
//     Number mu = static_cast<Number>(5.0f);
    


//     Number *d_x, *d_b;
//     AssertCuda(cudaMalloc(&d_x, totalSize * sizeof(Number)));
//     AssertCuda(cudaMalloc(&d_b, totalSize * sizeof(Number)));

//     AssertCuda(cudaMemcpy(d_x, x.data(), totalSize * sizeof(Number), cudaMemcpyHostToDevice));
//     AssertCuda(cudaMemcpy(d_b, b.data(), totalSize * sizeof(Number), cudaMemcpyHostToDevice));

//     Number *d_val;
//     int *d_col, *d_rowIdx;
//     int numNonZeros = L.nonZeros();

//     AssertCuda(cudaMalloc(&d_val, numNonZeros * sizeof(Number)));
//     AssertCuda(cudaMalloc(&d_col, numNonZeros * sizeof(int)));
//     AssertCuda(cudaMalloc(&d_rowIdx, (L.outerSize() + 1) * sizeof(int)));

//     AssertCuda(cudaMemcpy(d_val, L.valuePtr(), numNonZeros * sizeof(Number), cudaMemcpyHostToDevice));
//     AssertCuda(cudaMemcpy(d_col, L.innerIndexPtr(), numNonZeros * sizeof(int), cudaMemcpyHostToDevice));
//     AssertCuda(cudaMemcpy(d_rowIdx, L.outerIndexPtr(), (L.outerSize() + 1) * sizeof(int), cudaMemcpyHostToDevice));

//     Number *d_Lx;
//     AssertCuda(cudaMalloc(&d_Lx, 2 * totalSize * sizeof(Number)));

//     int numBlocks2 = (2 * totalSize + blockSize - 1) / blockSize;
//     sparseMatVec<<<numBlocks2, blockSize>>>(d_x, d_Lx, d_val, d_col, d_rowIdx, 2 * totalSize);
//     AssertCuda(cudaGetLastError());

//     gradDescCuda<<<numBlocks, blockSize>>>(d_x, d_b, d_val, d_col, d_rowIdx, d_Lx, mu, learningRate, width, height, maxIterations);
//     AssertCuda(cudaGetLastError());


//     AssertCuda(cudaMemcpy(x.data(), d_x, totalSize * sizeof(Number), cudaMemcpyDeviceToHost));

//     AssertCuda(cudaFree(d_x));
//     AssertCuda(cudaFree(d_b));
//     AssertCuda(cudaFree(d_Lx));
//     AssertCuda(cudaFree(d_val));
//     AssertCuda(cudaFree(d_col));
//     AssertCuda(cudaFree(d_rowIdx));
// }



template <typename Number>
void applyGaussianBlurCuda(const unsigned char* inputImage, unsigned char* outputImage, 
                            const int width, const int height, const int kernelSize, 
                            const Number sigma, Eigen::SparseMatrix<Number>& A)
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



template <typename Number>
void computeQRFactorization(const Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic>& AV, 
                             Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic>& Q, 
                             Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic>& R)
{
    Eigen::HouseholderQR<Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic>> qr(AV);
    Q = qr.householderQ();
    R = qr.matrixQR().template triangularView<Eigen::Upper>();
}

template <typename Number>
Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic> createInitialV0(int n, int k) {
    Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic> V0(n, k);
    V0.setRandom();
    Eigen::HouseholderQR<Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic>> qr(V0);
    V0 = qr.householderQ();
    return V0;
}


template <typename Number>
Eigen::VectorXd iterativeProcess(Eigen::SparseMatrix<Number>& A, Eigen::SparseMatrix<Number>& L,
                      Eigen::VectorXd& b_delta, Number delta, int q,
                      Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic>& x_k, Number epsilon, Number tau,
                      int K, int r, Number gamma) {

    //s10444-023-10020-8.pdf                        
    Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic> V_k = createInitialV0<Number>(A.cols(), 1);
    Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic> Q_A_k, R_A_k, Q_L_k, R_L_k;
    Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic> AV_k = A * V_k;
    Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic> LV_k = L * V_k;

    computeQRFactorization(AV_k, Q_A_k, R_A_k);
    computeQRFactorization(LV_k, Q_L_k, R_L_k);

    Eigen::VectorXd y_k = Eigen::VectorXd::Zero(V_k.cols());

    for (int k = 0; k < K; ++k) {
        if (k % r == 0 && k != 0) {
            V_k = x_k / x_k.norm();
            AV_k = A * V_k;
            LV_k = L * V_k;

            computeQRFactorization(AV_k, Q_A_k, R_A_k);
            computeQRFactorization(LV_k, Q_L_k, R_L_k);
        }

        Eigen::VectorXd u_k = L * x_k;
        Eigen::VectorXd omega_k = (u_k.array() *(1.0 - ((u_k.array().square() + epsilon * epsilon) / (epsilon * epsilon)).pow(q / 2.0 - 1.0))).matrix();

        Eigen::VectorXd b_At = Q_A_k.transpose() * b_delta;
        Eigen::VectorXd b_Lt = Q_L_k.transpose() * omega_k;

        Eigen::MatrixXd R_combined = R_A_k.transpose() * R_A_k + tau * (R_L_k.transpose() * R_L_k);
        Eigen::VectorXd rhs_combined = R_A_k.transpose() * b_At + tau * (R_L_k.transpose() * b_Lt);

        y_k = R_combined.ldlt().solve(rhs_combined);
        if ((y_k - y_k).norm() <= gamma * y_k.norm()) {
            break;
        }

        Eigen::VectorXd r_k_next = A.transpose() * (A * V_k * y_k - b_delta) +
                                   tau * L.transpose() * (L * V_k * y_k - omega_k);
        Eigen::VectorXd v_k_next = r_k_next / r_k_next.norm();

        Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic> V_k_next(V_k.rows(), V_k.cols() + 1);
        V_k_next << V_k, v_k_next;
        V_k = V_k_next;

        AV_k = A * V_k;
        LV_k = L * V_k;

        computeQRFactorization(AV_k, Q_A_k, R_A_k);
        computeQRFactorization(LV_k, Q_L_k, R_L_k);
    }

    Eigen::VectorXd x_star = V_k * y_k;
    return x_star;
}


int main()
{
    using Number = double;

    auto image = std::make_shared<BMPImage>("rect.bmp");
    int width = image->GetWidth();
    int height = image->GetHeight();
    int kernelSize = 5;
    Number sigma = 3.0f;

    auto blurredImage = std::unique_ptr<unsigned char[]>(new unsigned char[width * height]);

    Eigen::SparseMatrix<Number> A = generateAMatrix(image->GetData(), width, height, kernelSize, sigma);
    applyGaussianBlurCuda(image->GetData(), blurredImage.get(), width, height, kernelSize, sigma, A);

    BMPImage blurred(width, height, blurredImage.get());
    blurred.SaveBMP("rect_blurred.bmp");

    Eigen::SparseMatrix<Number> L;
    generateLMatrix(L, width, height);

    Eigen::VectorXd b_delta(width * height);
    for (int i = 0; i < width * height; ++i)
    {
        b_delta[i] = static_cast<Number>(blurredImage[i]);
    }

    // Define initial guess and parameters for the iterative process
    Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic> x_k = Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic>::Zero(width * height, 1);
    Number delta = 1e-3f;  // Threshold for error
    Number epsilon = 1e-6f;
    Number tau = 0.1f;
    int K = 100;  // Maximum iterations
    int r = 5;    // Reorthogonalization frequency
    Number gamma = 1e-4f;
    int q = 2;

    // Run the iterative process to solve for x_star
    Eigen::VectorXd x_star = iterativeProcess<Number>(A, L, b_delta, delta, q, x_k, epsilon, tau, K, r, gamma);

    // Normalize and save the output image
    auto outputImage = std::unique_ptr<unsigned char[]>(new unsigned char[width * height]);
    Number minVal = x_star.minCoeff();
    Number maxVal = x_star.maxCoeff();

    std::cout << "Min value in output: " << minVal << std::endl;
    std::cout << "Max value in output: " << maxVal << std::endl;

    for (int i = 0; i < width * height; ++i)
    {
        outputImage[i] = static_cast<unsigned char>(
            clamp(static_cast<int>((x_star[i] - minVal) / (maxVal - minVal) * 255), 0, 255));
    }

    BMPImage result(width, height, outputImage.get());
    result.SaveBMP("rect_output_cuda.bmp");

    return 0;
}
