#include<mpi.h>
#include <iostream>
#include <algorithm>

using namespace std;

double rand_double(int from, int to) {
    return (double) (rand() % (to - from + 1) - from);
}

double *create_vector(int n) {
    auto *vector = new double[n];
    for (int i = 0; i < n; ++i) {
        vector[i] = rand_double(-10, 10);
    }
    return vector;
}

void clear_vector(const double *vector) {
    delete[] vector;
}

void print_vector(double *vector, int n) {
    for (int i = 0; i < n; ++i) {
        cout << vector[i] << "\t";
    }
    cout << endl;
}

double **create_ex_matrix(int n) {
    auto **matrix = new double *[n];
    for (int i = 0; i < n; ++i) {
        matrix[i] = new double[n + 1];
    }
    return matrix;
}

double **generate_extended_matrix(int n, const double *solution) {
    auto **matrix = new double *[n];
    for (int i = 0; i < n; ++i) {
        matrix[i] = new double[n + 1];
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            matrix[i][j] = rand_double(-100, 100);
        }
    }

    for (int i = 0; i < n; ++i) {
        double b = 0;
        for (int j = 0; j < n; ++j) {
            b += matrix[i][j] * solution[j];
        }
        matrix[i][n] = b;
    }
    return matrix;
}

void clear_matrix(double **matrix, int n) {
    for (int i = 0; i < n; ++i) {
        delete[] matrix[i];
    }
    delete[] matrix;
}

void tile(double **a, double **u, int n, int rank, int jgl, int r2, int r3, int rank_size, MPI_Status &status) {
    if (rank == 0)
        for (int igl = 0; igl < 4; ++igl)
            for (int k = 0; k < n; ++k) {
                int rank_counts_u = k / r2;
                if (rank == rank_counts_u) {
                    for (int j = max(k + 1, jgl * r3); j < ((jgl + 1) * r3); ++j) {
                        u[k][j] = a[k][j] / a[k][k];
                    }
                } else if (rank > rank_counts_u) {
                    // receive u
                    MPI_Recv(&(u[k][k]), n - k + 1, MPI_DOUBLE, rank - 1, 100 + k, MPI_COMM_WORLD, &status);
                }
                if (rank >= rank_counts_u && rank < rank_size - 1) {
                    // send u
                    MPI_Send(&(u[k][k]), n - k + 1, MPI_DOUBLE, rank + 1, 100 + k, MPI_COMM_WORLD);
                }
                for (int i = max(k + 1, igl * r2); i < ((igl + 1) * r2); ++i) {
                    for (int j = max(k + 1, jgl * r3); j < ((jgl + 1) * r3); ++j) {
                        a[i][j] -= a[i][k] * u[k][j];
                    }
                }
            }
}

int main(int argc, char **argv) {
    int n = 20;
    int q3 = 3, r2 = 5, r3 = 7;
    double *y = create_vector(n);
    double **a = generate_extended_matrix(n, y);
    auto *x = new double[n];

    MPI_Status status;
    MPI_Init(&argc, &argv);
    int currentRank, currentSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &currentRank);
    MPI_Comm_size(MPI_COMM_WORLD, &currentSize);

    if (currentRank == 0) {
        cout << "Rank size: " << currentSize << endl;
        cout << "Solution:" << endl;
        print_vector(y, n);
    }

    clear_vector(y);

    double **u = create_ex_matrix(n);
    for (int jgl = 0; jgl < q3; ++jgl) {
        tile(a, u, n, currentRank, jgl, r2, r3, currentSize, status);
    }
    clear_matrix(u, n);

    if (currentRank == 0) {
        for (int i = n - 1; i >= 0; --i) {
            x[i] = a[i][n];
            for (int j = i + 1; j < n; ++j) {
                x[i] -= a[i][j] * x[j];
            }
            x[i] /= a[i][i];
        }

        cout << "Program solution:" << endl;
        print_vector(x, n);

        clear_matrix(a, n);
        clear_vector(x);
    }

    MPI_Finalize();

    return 0;
}