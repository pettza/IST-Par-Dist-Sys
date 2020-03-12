#include <stdlib.h>
#include <vector>
#include <fstream>
#include <iostream>


/** Matrix type
*/
class matrix
{
public:
	using row_t = std::vector<double>;
	
	matrix(int n_rows, int n_columns)
	: n_rows(n_rows), n_columns(n_columns), data(n_rows, row_t(n_columns))
	{}

	row_t& operator[](int row) { return data[row]; }

	const row_t& operator[](int row) const { return data[row]; }

public:
	const int n_rows, n_columns;

private:
	std::vector<row_t> data;
};


/** Sparse matrix type
*/
class sparse_matrix
{
public:
	sparse_matrix(int n_rows, int n_columns)
	: n_rows(n_rows), n_columns(n_columns), rows(n_rows), columns(n_columns)
	{}
	
	void add_element(int row, int column, double element)
	{
		int elem_idx = elements.size();
		elements.push_back(element);
		rows[row].push_back({column, elem_idx});
		columns[column].push_back({row, elem_idx});
	}

	struct idx_pair
	{
		int idx;
		int elem_idx;
	};

	const int n_rows, n_columns;
	std::vector<double> elements;
	std::vector<std::vector<idx_pair>> rows;
	std::vector<std::vector<idx_pair>> columns;
};


// globals
int n_iter;
int nF;
int nU;
int nI;
double learning_rate;


#define RAND01 ((double) std::rand() / (double) RAND_MAX)

void random_fill_matrix(matrix& m)
{
	std::srand(0);

	for (int r = 0; r < m.n_rows; r++)
		for (int c = 0; c < m.n_columns; c++)
			m[r][c] = RAND01 / (double) nF;
}


// Creates a matrix from a file
sparse_matrix parse_file(const char* filename)
{
	std::ifstream file(filename);
	
	if (file.fail())
	{
		std::cerr << "Failed to open file: " << filename << std::endl;
		exit(1);
	}

	int n_elems;

	file >> n_iter;
	file >> learning_rate;
	file >> nF;
	file >> nU >> nI >> n_elems;

	sparse_matrix A(nU, nI);
	for (int i = 0; i < n_elems; i++)
	{
		int row, col;
		double elem;

		file >> row >> col >> elem;

		A.add_element(row, col, elem);
	}

	file.close();
	
	return A;
}


void update_LR(const sparse_matrix& A, matrix*& L, matrix*& R, matrix*& oldL, matrix*& oldR)
{
	std::swap(L, oldL);
	std::swap(R, oldR);
}


int main(int argc, char** argv)
{
	sparse_matrix A = parse_file(argv[1]);
	matrix* L    = new matrix(nU, nF);
	matrix* oldL = new matrix(nU, nF);
	matrix* R    = new matrix(nF, nI);
	matrix* oldR = new matrix(nF, nI);

	random_fill_matrix(*L);
	random_fill_matrix(*R);

	for (int i = 0; i < n_iter; i++) update_LR(A, L, R, oldL, oldR);
}
