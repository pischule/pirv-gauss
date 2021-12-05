#include <mpi.h>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <cmath>

using namespace std;

int N;

double *create_vector(int n) {
    auto *vector = new double[n];
    for (int i = 0; i < n; ++i) {
        vector[i] = 1.0;
    }
    return vector;
}

void clear_vector(const double *vector) {
    delete[] vector;
}

void print_vector(double *vector, int n) {
    for (int i = 0; i < n; ++i) {
        cout << vector[i] << "\n";
    }
    cout << endl;
}

double *create_matrix(int n) {
    auto *matrix = new double[n];
    return matrix;
}

int p(int y, int x) {
    return y * (N + 1) + x;
}

double *generate_extended_matrix(int n, const double *solution) {
    auto *matrix = new double[n * (n + 1)];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            matrix[p(i, j)] = 0;
        }
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                matrix[p(i, j)] = 100;
            } else {
                matrix[p(i, j)] = (2.0 * i + j) / 100000;
            }
        }
    }

    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int j = 0; j < n; ++j) {
            sum += matrix[p(i, j)] * solution[j];
        }
        matrix[p(i, n)] = sum;
    }
    return matrix;
}

void clear_matrix(const double *matrix) {
    delete[] matrix;
}

void print_matrix(double *matrix, int n, int m) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            cout << setprecision(10) << matrix[i * m + j] << "\t";
        }
        cout << endl;
    }
}

void tile(double *a, double *u, int n, int rank, int jgl, int r2, int r3, int rank_size, MPI_Status &status) {
    int iglxr2 = rank * r2;
    int jglxr3 = jgl * r3;

    int jmax = min(jglxr3 + r3, n + 1);
    int imax = min(iglxr2 + r2, n);

    // count a with previous known u
    for (int k = 0; k < iglxr2; ++k) {
        int kxr3 = k * r3;
        for (int i = iglxr2; i < imax; ++i) {
            int iloc = i - iglxr2;
            for (int j = max(k + 1, jglxr3); j < jmax; ++j) {
                int jloc = j - jglxr3;
                // a(i,j) -= a(i,k) u(k,j)
                a[p(iloc, j)] -= a[p(iloc, k)] * u[kxr3 + jloc];
            }
        }
    }

    // count u, then count a
    for (int k = rank * r2; k < (rank + 1) * r2; ++k) {
        int kxr3 = k * r3;

        int kloc = k - iglxr2;
        int klocxn1 = kloc * (n + 1);
        for (int j = max(jglxr3, k + 1); j < jmax; ++j) {
            int jloc = j - jglxr3;
            u[kxr3 + jloc] = a[klocxn1 + j] / a[klocxn1 + k];
        }
        for (int i = k + 1; i < imax; ++i) {
            int iloc = i - iglxr2;
            for (int j = max(k + 1, jglxr3); j < jmax; ++j) {
                int jloc = j - jglxr3;
                // a(i,j) -= a(i,k) u(k,j)
                a[p(iloc, j)] -= a[p(iloc, k)] * u[kxr3 + jloc];
            }
        }
    }
}


void gauss(double *a, int n, int rank, int r2, int r3, int q3, int rank_size, MPI_Status &status) {
    double *u = create_matrix((rank + 1) * r2 * r3);
    for (int i = 0; i < (rank + 1) * r2 * r3; ++i) {
        u[i] = 0.0;
    }
    for (int jgl = 0; jgl < q3; ++jgl) {
        if (rank > 0) { // if slave
            MPI_Recv(u, rank * r2 * r3, MPI_DOUBLE, rank - 1, 1001, MPI_COMM_WORLD, &status);
        }
        tile(a, u, n, rank, jgl, r2, r3, rank_size, status);
        if (rank < rank_size - 1) { // if not last
            MPI_Send(u, (rank + 1) * r2 * r3, MPI_DOUBLE, rank + 1, 1001, MPI_COMM_WORLD);
        }
    }
    clear_matrix(u);
}


void gauss_inverse(const double *a, double *x) {
    for (int i = N - 1; i >= 0; --i) {
        x[i] = a[i * (N + 1) + N];
        for (int j = i + 1; j < N; ++j) {
            x[i] -= a[i * (N + 1) + j] * x[j];
        }
        x[i] /= a[i * (N + 1) + i];
    }
}

int main(int argc, char **argv) {
    int rank, size;
    MPI_Status status;
    int n, q2, q3, r2, r3;
    double *matrix = nullptr;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        if (argc < 5 || strcmp("-size", argv[1]) != 0 || strcmp("-r3", argv[3]) != 0) {
            cerr << "Bad arguments! Input must have the following options:\n-size [n] -r3 [r3]" << endl;
            exit(1);
        }
    }
    n = atoi(argv[2]);
    r3 = atoi(argv[4]);

    N = n;
    q2 = size;

    r2 = (n % q2) == 0 ? n / q2 : n / q2 + 1;
    q3 = (n % r3) == 0 ? n / r3 : n / r3 + 1;

    if (rank == 0) {
        cout << "Rank size: " << size << endl;
        double *y = create_vector(n);
        matrix = generate_extended_matrix(n, y);
        print_matrix(matrix, n, n + 1);
        clear_vector(y);
    }

    double *a = create_matrix((n + 1) * r2);

    int *sendcounts = new int[size];
    int *displs = new int[size];

    displs[0] = 0;
    int r2xn1 = r2 * (n + 1);
    for (int p = 0; p < size - 1; ++p) {
        sendcounts[p] = r2xn1;
        displs[p + 1] = displs[p] + sendcounts[p];
    }
    sendcounts[size - 1] = (n - (size - 1) * r2) * (n + 1);


    MPI_Scatterv(matrix, sendcounts, displs, MPI_DOUBLE, a, sendcounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    gauss(a, n, rank, r2, r3, q3, size, status);
    MPI_Gatherv(a, sendcounts[rank], MPI_DOUBLE, matrix, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        auto *x = new double[n];
        gauss_inverse(matrix, x);
        print_vector(x, n);
        clear_matrix(matrix);
        delete[] x;
        delete[] displs;
        delete[] sendcounts;
    }

    clear_matrix(a);
    MPI_Finalize();
    return 0;
}