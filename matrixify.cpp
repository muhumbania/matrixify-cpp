#include "matrixify.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <chrono>
#include <iostream>

float **matrix_alloc_host(int rows, int cols)
{

    float **matrix;

    matrix = new float *[rows];
    matrix[0] = new float[rows * cols];

    for (int i = 1; i < rows; i++)
    {
        matrix[i] = matrix[i - 1] + cols;
    }

    return matrix;
}

void matrix_free_host(float **matrix)
{
    if (matrix != nullptr)
    {
        delete[] matrix[0]; // free the big block
        delete[] matrix;    // free the table of lines pointer
    }
}

void matmul_naive(const float *A, const float *B, float *C, int M, int N, int K)
{
    // loop i for lines of C and A
    for (int i = 0; i < M; ++i)
    {
        // loop j for columns of B and C
        for (int j = 0; j < N; ++j)
        {

            float sum = 0.0f; // temporary accumulator in a register for faster R/W operations

            // loop k (Dot Product) of line i of A and column j of B
            for (int k = 0; k < K; ++k)
            {
                // [line * total_columns + columns]
                sum += A[i * K + k] * B[k * N + j];
            }

            // final result in C
            C[i * N + j] = sum;
        }
    }
}

void matmul_cache_friendly(const float *A, const float *B, float *C, int M, int N, int K)
{
    // initialize matrix C with zeros
    for (int i = 0; i < M * N; ++i)
    {
        C[i] = 0.0f;
    }

    // i -> k -> j
    for (int i = 0; i < M; ++i)
    {
        for (int k = 0; k < K; ++k)
        {

            // An element of A is extracted only once
            // And remains unchanged for the duration of j-loop
            float rA = A[i * K + k];

            for (int j = 0; j < N; ++j)
            {
                // Sequential access for B and C
                C[i * N + j] += rA * B[k * N + j];
            }
        }
    }
}

void matmul_tiled(
    const float *A,
    const float *B,
    float *C,
    int M, int N, int K,
    int block_size)
{
    // C = A(MxK) * B(KxN)

    for (int ii = 0; ii < M; ii += block_size)
    {
        for (int jj = 0; jj < N; jj += block_size)
        {
            for (int kk = 0; kk < K; kk += block_size)
            {
                // Work on one C tile at a time
                for (int i = ii; i < std::min(ii + block_size, M); ++i)
                {
                    for (int j = jj; j < std::min(jj + block_size, N); ++j)
                    {
                        float sum = C[i * N + j];

                        for (int k = kk; k < std::min(kk + block_size, K); ++k)
                        {
                            sum += A[i * K + k] * B[k * N + j];
                        }

                        C[i * N + j] = sum;
                    }
                }
            }
        }
    }
}

inline int matrix_get_index(int row, int col, int stride)
{
    return row * stride + col;
}

void matrix_verify(
    const float* C_ref,
    const float* C_test,
    int size,
    float epsilon)
{
    for (int i = 0; i < size; ++i)
    {
        float diff = std::abs(C_ref[i] - C_test[i]);

        assert(diff < epsilon);
    }
}

void matrix_benchmark(
    void (*func)(const float*, const float*, float*, int, int, int),
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    auto start = std::chrono::high_resolution_clock::now();

    func(A, B, C, M, N, K);

    auto end = std::chrono::high_resolution_clock::now();

    double seconds =
        std::chrono::duration<double>(end - start).count();

    std::cout << "Time: " << seconds << " seconds\n";
}