#// main.cpp
#include "matrixify.h"
#include <iostream>
using namespace std;

int main()
{
    float **A = matrix_alloc_host(3, 2);
    float **B = matrix_alloc_host(2, 4);

    // Remplissage de A (3x2) avec des nombres séquentiels : 1.0, 2.0, 3.0...
    float valueA = 1.0f;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            A[i][j] = valueA++;
        }
    }

    // Remplissage de B (2x4) avec une valeur fixe ou une formule
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            B[i][j] = (float)(i + j); // Exemple de formule mathématique
        }
    }

    float **C = matrix_alloc_host(3, 4);

    matmul_naive(A[0], B[0], C[0], 3, 4, 2);

    matrix_benchmark(matmul_naive, A[0], B[0], C[0], 3, 4, 2);
    matrix_benchmark(matmul_cache_friendly, A[0], B[0], C[0], 3, 4, 2);

    return 0;
}