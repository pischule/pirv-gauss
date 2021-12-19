#include <mpi.h>
#include <algorithm>
#include <vector>
#include <cassert>
#include <random>


using namespace std;

int matrix_size = 400;
int current_rank;
int current_size;
int tile_size;

int yx(int y, int x);

vector<double> backwardGauss(vector<double> a);

vector<double> extendedMatrix(vector<double> &x);

bool areEqual(vector<double> &a, vector<double> &b);

int yx(int y, int x) {
    return y * (matrix_size + 1) + x;
}

vector<double> randomVector(double from, double to) {
    vector<double> result(matrix_size);
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(from, to);
    for (int i = 0; i < matrix_size; i++) {
        result[i] = dis(gen);
    }
    return result;
}

vector<double> extendedMatrix(vector<double> &x) {
    vector<double> matrix(matrix_size * (matrix_size + 1), 0);
    for (int i = 0; i < matrix_size; ++i) {
        for (int j = 0; j < matrix_size; ++j) {
            if (i == j) {
                matrix[yx(i, j)] = 100;
            } else {
                matrix[yx(i, j)] = (2.0 * i + j) / 100000;
            }
        }
    }

    for (int i = 0; i < matrix_size; ++i) {
        double sum = 0;
        for (int j = 0; j < matrix_size; ++j) {
            sum += matrix[yx(i, j)] * x[j];
        }
        matrix[yx(i, matrix_size)] = sum;
    }
    return matrix;
}

bool areEqual(vector<double> &a, vector<double> &b) {
    double max_delta = 0.5;
    for (int i = 0; i < matrix_size; ++i) {
        if (abs(a[i] - b[i]) > max_delta) {
            cerr << "Vectors are not equal, delta = " << abs(a[i] - b[i]) << endl;
            return false;
        }
    }
    return true;
}

void broadcastRow(vector<double> &row, int row_number) {
    MPI_Bcast(row.data(), matrix_size + 1, MPI_DOUBLE, row_number / tile_size, MPI_COMM_WORLD);
}

void translateRow(vector<double> &row, int row_number) {
    if (current_rank > row_number / tile_size) {
        MPI_Recv(row.data(),
                 matrix_size + 1,
                 MPI_DOUBLE,
                 current_rank - 1,
                 0,
                 MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }
    if (current_rank >= row_number / tile_size && current_rank < current_size - 1) {
        MPI_Send(row.data(),
                 matrix_size + 1,
                 MPI_DOUBLE,
                 current_rank + 1,
                 0,
                 MPI_COMM_WORLD);
    }
}

void tile(vector<double> &a, int k, vector<double> &row) {
    if (k / tile_size == current_rank) {
        int k_local = k % tile_size;
        for (int j = 0; j < matrix_size; ++j) {
            row[j] = a.at(yx(k_local, j)) / a.at(yx(k_local, k));
        }
    }

//    broadcastRow(row, k);
    translateRow(row, k);

    int i_start = max(k + 1, current_rank * tile_size);
    int i_end = min(matrix_size, (current_rank + 1) * tile_size);
    for (int i = i_start; i < i_end; ++i) {
        int i_local = i % tile_size;
        for (int j = k + 1; j < matrix_size + 1; ++j) {
            a.at(yx(i_local, j)) -= a.at(yx(i_local, k)) * row[j];
        }
    }
}

vector<double> forwardGauss(vector<double> a) {
    vector<double> row(matrix_size + 1, 0);
    for (int k = 0; k < matrix_size - 1; ++k) {
        tile(a, k, row);
    }
    return a;
}

vector<double> backwardGauss(vector<double> a) {
    vector<double> x(matrix_size);
    for (int i = matrix_size - 1; i >= 0; --i) {
        double sum = 0;
        for (int j = i + 1; j < matrix_size; ++j) {
            sum += a[yx(i, j)] * x[j];
        }
        x[i] = (a[yx(i, matrix_size)] - sum) / a[yx(i, i)];
    }
    return x;
}


int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &current_size);

    if (current_rank == 0) {
        if (argc != 2) {
            cerr << "Usage: " << argv[0] << " MATRIX_SIZE" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }
    matrix_size = atoi(argv[1]);

    tile_size = matrix_size / current_size;
    assert(matrix_size % current_size == 0);

    vector<double> x(0);               // only for master
    vector<double> extended_matrix(0); // only for master
    vector<double> local_extended_matrix(tile_size * (matrix_size + 1), 0);

    if (current_rank == 0) {
        x = vector<double>(matrix_size, 1.0);
//        x = randomVector(-100, 100);
        extended_matrix = extendedMatrix(x);

    }

    double start_time = MPI_Wtime();

    MPI_Scatter(extended_matrix.data(),
                tile_size * (matrix_size + 1),
                MPI_DOUBLE,
                local_extended_matrix.data(),
                tile_size * (matrix_size + 1),
                MPI_DOUBLE,
                0,
                MPI_COMM_WORLD);

    forwardGauss(local_extended_matrix);

    MPI_Gather(local_extended_matrix.data(),
               tile_size * (matrix_size + 1),
               MPI_DOUBLE,
               extended_matrix.data(),
               tile_size * (matrix_size + 1),
               MPI_DOUBLE,
               0,
               MPI_COMM_WORLD);

    if (current_rank == 0) {
        vector<double> solution = backwardGauss(extended_matrix);
        assert(areEqual(solution, x));

        double end_time = MPI_Wtime();
        cout << "Time: " << end_time - start_time << endl;
    }

    return MPI_Finalize();
}