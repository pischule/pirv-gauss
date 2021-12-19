#include <mpi.h>
#include <algorithm>
#include <vector>
#include <cassert>
#include <random>


using namespace std;

int matrix_size = 20;
int current_rank;
int current_size;

int yx(int y, int x);

vector<double> forward_gauss(vector<double> a);

vector<double> backward_gauss(vector<double> a);

vector<double> generate_extended_matrix(vector<double> &x);

bool check_solution(vector<double> &a, vector<double> &b);

vector<double> gauss(vector<double> extended_matrix);

int yx(int y, int x) {
    return y * (matrix_size + 1) + x;
}

vector<double> random_vector(double from, double to) {
    assert(from <= to);
    vector<double> x(matrix_size);
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(from, to);
    for (int i = 0; i < matrix_size; ++i) {
        x[i] = dis(gen);
    }
    return x;
}

vector<double> generate_extended_matrix(vector<double> &x) {
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

bool check_solution(vector<double> &a, vector<double> &b) {
    double max_delta = 0.01;

    if (a.size() != b.size()) {
        return false;
    }

    double delta = 0;
    for (int i = 0; i < a.size(); ++i) {
        delta += abs(a[i] - b[i]);
    }

    return delta < max_delta;
}

vector<double> gauss(vector<double> extended_matrix) {
    extended_matrix = forward_gauss(extended_matrix);
    return backward_gauss(extended_matrix);
}

vector<double> forward_gauss(vector<double> a) {
    for (int k = 0; k < matrix_size - 1; ++k) {
        for (int i = k + 1; i < matrix_size; ++i) {
            for (int j = k + 1; j < matrix_size + 1; ++j) {
                a[yx(i, j)] -= a[yx(i, k)] *
                               a[yx(k, j)] / a[yx(k, k)];
            }
        }
    }
    return a;
}

vector<double> backward_gauss(vector<double> a) {
    vector<double> x(matrix_size, 0);
    for (int i = matrix_size - 1; i >= 0; --i) {
        x[i] = a[yx(i, matrix_size)];
        for (int j = i + 1; j < matrix_size; ++j) {
            x[i] -= a[yx(i, j)] * x[j];
        }
        x[i] /= a[yx(i, i)];
    }
    return x;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &current_size);

    vector<double> x;
    vector<double> extended_matrix;

    if (current_rank == 0) {
//        x = vector<double>(matrix_size, 1.0);
        x = random_vector(-100, 100);
        extended_matrix = generate_extended_matrix(x);

        vector<double> my_solution = gauss(extended_matrix);
        assert(check_solution(my_solution, x));
    }
    return 0;
}