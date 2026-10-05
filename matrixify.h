
#ifndef MATRIXIFY_H
#define MATRIXIFY_H

// memory architecture functions
float **matrix_alloc_host(int rows, int cols); // allocate host memorry
void matrix_free_host(float **ptr);            // Frees the host memory.

// cpu execution pipeline
void matmul_naive(const float *A, const float *B, float *C, int M, int N, int K);          // naive multiplication, cache problems
void matmul_cache_friendly(const float *A, const float *B, float *C, int M, int N, int K); // Cache frienly, All the matrices are accessed sequentially
void matmul_tiled(
    const float *A,
    const float *B,
    float *C,
    int M, int N, int K,
    int block_size); // Tiled multiplication

inline int matrix_get_index(int row, int col, int stride); // To get the index of an element using the row-major formula

// This function is for verification of the accuracy of the results
// between the naive and optimize approach
void matrix_verify(
    const float *C_ref,
    const float *C_test,
    int size,
    float epsilon);

// This function is for comparison between the different functions to evaluate which one is faster
void matrix_benchmark(
    void (*func)(const float *, const float *, float *, int, int, int),
    const float *A,
    const float *B,
    float *C,
    int M,
    int N,
    int K);

#endif