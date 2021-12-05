#include <mpi.h>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <iomanip>
using namespace std;

double* create_vector(int n) {
	double* vector = new double[n];
	for (int i = 0; i < n; ++i) {
		vector[i] = 1.0;
	}
	return vector;
}

void clear_vector(double* vector) {
	delete[] vector;
}

void print_vector(double* vector, int n) {
	for (int i = 0; i < n; ++i) {
		cout << vector[i] << "\n";
	}
	cout << endl;
}

double* create_matrix(int n) {
	double* matrix = new double[n];
	return matrix;
}

double* create_generate_ex_matrix(int n, double* solution) {
	double* matrix = new double[n * (n + 1)];
	for (int i = 0; i < n; ++i) {
		// matrix[i * (n + 1) + i] = 100;
		matrix[i * (n + 2)] = 100.0;
		for (int j = 0; j < n; ++j) {
			if (i != j) {
				matrix[i * (n + 1) + j] = (2 * i + j) / 100000.0; // TODO
			}
		}
		matrix[i * (n+1) + n] = 0;
		for (int j = 0; j < n; ++j) {
			matrix[i * (n + 1) + n] += matrix[i * (n+1) + j] * solution[j];
		}
	}
	return matrix;
}

void clear_matrix(double* matrix) {
	delete[] matrix;
}

void print_ex_matrix(double* matrix, int n) {
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n + 1; ++j) {
			cout << matrix[i * (n + 1) + j] << "\t";
		}
		cout << endl;
	}
}

void print_matrix(double* matrix, int n, int m) {
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < m; ++j) {
			cout << setprecision(10) << matrix[i * m + j] << "\t";
		}
		cout << endl;
	}
}


void tile_row(double* a, double* u, int n, int rank, int jgl, int r2, int r3, int rank_size, MPI_Status& status) {
	int iglxr2 = rank * r2;
	int jglxr3 = jgl * r3;

	int jmax = min(jglxr3 + r3, n + 1);
	int imax = min(iglxr2 + r2, n);

	for (int k = 0; k < n; ++k) {
		int rank_counts_u = k / r2;
		if (rank == rank_counts_u) {
			int kloc = k - iglxr2;
			for (int j = jglxr3; j < jmax; ++j) {
				int jloc = j - jglxr3;
				u[jloc] = a[kloc * (n + 1) + j] / a[kloc * (n + 1) + k];
			}
		}
		else if (rank > rank_counts_u) {
			MPI_Recv(&(u[0]), r3, MPI_DOUBLE, rank - 1, 100, MPI_COMM_WORLD, &status);
		}
		if (rank >= rank_counts_u && rank < rank_size - 1) {
			MPI_Send(&(u[0]), r3, MPI_DOUBLE, rank + 1, 100, MPI_COMM_WORLD);
		}
		for (int i = max(k + 1, iglxr2); i < imax; ++i) {
			int iloc = i - iglxr2;
			for (int j = max(k + 1, jglxr3); j < jmax; ++j) {
				int jloc = j - jglxr3;
				a[iloc * (n + 1) + j] -= a[iloc * (n + 1) + k] * u[jloc];
			} 
		}
	}
}

void tile_block(double* a, double* u, int n, int rank, int jgl, int r2, int r3, int rank_size, MPI_Status& status) {
	int iglxr2 = rank * r2;
	int jglxr3 = jgl * r3;

	int jmax = min(jglxr3 + r3, n + 1);
	int imax = min(iglxr2 + r2, n);

	for (int k = 0; k < n; ++k) {
		int rank_counts_u = k / r2;
		if (rank > rank_counts_u && k % r2 == 0) {
			MPI_Recv(&(u[0]), r2 * r3, MPI_DOUBLE, rank - 1, 100, MPI_COMM_WORLD, &status);
			if (rank < rank_size - 1) {
				MPI_Send(&(u[0]), r2 * r3, MPI_DOUBLE, rank + 1, 100, MPI_COMM_WORLD);
			}
		}
		if (rank == rank_counts_u) {
			int kloc = k - iglxr2;
			int klocxr3 = kloc * r3;
			int klocxn1 = kloc * (n + 1);
			int klocxn1k = klocxn1 + k;
			for (int j = jglxr3; j < jmax; ++j) {
				int jloc = j - jglxr3;
				u[klocxr3 + jloc] = a[klocxn1 + j] / a[klocxn1k];
			}
			if (rank < (k + 1) / r2 && rank < rank_size - 1) {
				MPI_Send(&(u[0]), r2 * r3, MPI_DOUBLE, rank + 1, 100, MPI_COMM_WORLD);
			}
		}
		for (int i = max(k + 1, iglxr2); i < imax; ++i) {
			int iloc = i - iglxr2;
			int ilocxr3 = iloc * r3;
			int ilocxn1 = iloc * (n + 1);
			int ilocxn1k = ilocxn1 + k;
			for (int j = max(k + 1, jglxr3); j < jmax; ++j) {
				int jloc = j - jglxr3;
				a[ilocxn1 + j] -= a[ilocxn1k] * u[ilocxr3 + jloc];
			}
		}
	}
}

void tile(double* a, double* u, int n, int rank, int jgl, int r2, int r3, int rank_size, MPI_Status& status) {
	int iglxr2 = rank * r2;
	int jglxr3 = jgl * r3;

	int jmax = min(jglxr3 + r3, n + 1);
	int imax = min(iglxr2 + r2, n);

	// count a with previous known u
	for (int k = 0; k < iglxr2; ++k) {
		int kxr3 = k * r3;
		for (int i = iglxr2; i < imax; ++i) {
			int iloc = i - iglxr2;
			int ilocxn1 = iloc * (n + 1);
			for (int j = max(k + 1, jglxr3); j < jmax; ++j) {
				int jloc = j - jglxr3;
				// a(i,j) -= a(i,k) u(k,j)
				a[ilocxn1 + j] -= a[ilocxn1 + k] * u[kxr3 + jloc];
			}
		}
	}

	// count u, then count a
	for (int k = rank * r2; k < (rank + 1) * r2; ++k) {
		int kxr3 = k * r3;
		// count u
		//int rank_counts_u = k / r2;
		//if (rank == rank_counts_u) {
			int kloc = k - iglxr2;
			int klocxn1 = kloc * (n + 1);
			for (int j = max(jglxr3, k + 1); j < jmax; ++j) {
				int jloc = j - jglxr3;
				u[kxr3 + jloc] = a[klocxn1 + j] / a[klocxn1 + k];
			}
		//}
		// count a
		for (int i = k + 1; i < imax; ++i) {
			int iloc = i - iglxr2;
			int ilocxn1 = iloc * (n + 1);
			for (int j = max(k + 1, jglxr3); j < jmax; ++j) {
				int jloc = j - jglxr3;
				// a(i,j) -= a(i,k) u(k,j)
				a[ilocxn1 + j] -= a[ilocxn1 + k] * u[kxr3 + jloc];
			}
		}
	}
}

void gauss_row(double* a, int n, int rank, int r2, int r3, int q3, int rank_size, MPI_Status& status) {
	double* u = create_matrix(r3);
	for (int jgl = 0; jgl < q3; ++jgl) {
		tile_row(a, u, n, rank, jgl, r2, r3, rank_size, status);
	}
	clear_matrix(u);
}

void gauss_block(double* a, int n, int rank, int r2, int r3, int q3, int rank_size, MPI_Status& status) {
	double* u = create_matrix(r2 * r3);
	for (int jgl = 0; jgl < q3; ++jgl) {
		tile_block(a, u, n, rank, jgl, r2, r3, rank_size, status);
	}
	clear_matrix(u);
}

void gauss(double* a, int n, int rank, int r2, int r3, int q3, int rank_size, MPI_Status& status) {
	double* u = create_matrix((rank + 1) * r2 * r3);
	for (int i = 0; i < (rank + 1) * r2 * r3; ++i) {
		u[i] = 0.0;
	}
	for (int jgl = 0; jgl < q3; ++jgl) {
		if (rank > 0) {
			MPI_Recv(u, rank * r2 * r3, MPI_DOUBLE, rank - 1, 1001, MPI_COMM_WORLD, &status);
		}
		tile(a, u, n, rank, jgl, r2, r3, rank_size, status);
		if (rank < rank_size - 1) {
			MPI_Send(u, (rank + 1) * r2 * r3, MPI_DOUBLE, rank + 1, 1001, MPI_COMM_WORLD);
		}
	}
	clear_matrix(u);
}

void gauss_inverse(double* a, double* x, int n) {
	for (int i = n - 1; i >= 0; --i) {
		x[i] = a[i * (n + 1) + n];
		for (int j = i + 1; j < n; ++j) {
			x[i] -= a[i * (n + 1) + j] * x[j];
		}
		x[i] /= a[i * (n + 1) + i];
	}
}

int main(int argc, char ** argv) {
	int rank, size;
	MPI_Status status;
	int n, q2, q3, r2, r3;

	double* matrix = nullptr;


	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (rank == 0) {
		if (argc < 5 || strcmp("-size", argv[1]) != 0 || strcmp("-r3", argv[3]) != 0) {
			cerr << "Wrong arguments! Input must have the following options:" << endl <<
				"-size [n] -r3 [r3]" << endl;
			exit(1);
		}
	}

	n = atoi(argv[2]);
	q2 = size;
	r3 = atoi(argv[4]);

	r2 = (n % q2) == 0 ? n / q2 : n / q2 + 1;
	q3 = (n % r3) == 0 ? n / r3 : n / r3 + 1;

	if (rank == 0)
	{
		cout << "Rank size: " << size << endl;

		double* y = create_vector(n);

		matrix = create_generate_ex_matrix(n, y);
		print_matrix(matrix, n, n + 1);

		clear_vector(y);
	}

	double* a = create_matrix((n + 1) * r2);

	int* sendcounts = new int[size];
	int* displs = new int[size];

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

	if (rank == 0)
	{

		auto* x = new double[n];


		gauss_inverse(matrix, x, n);

        print_vector(x, n);

		clear_matrix(matrix);
		delete[] x;
		delete[] displs;
		delete[] sendcounts;
	}

	//clear_matrix(a);

	MPI_Finalize();
	return 0;
}