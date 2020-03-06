#include <stdlib.h>
#include <vector>
#include <fstream>
#include <iostream>

// globals
int n_iter;
int nF;
int nU;
int nI;
double learning_rate;
double** L, R;

#define RAND01 ((double) std::rand() / (double) RAND_MAX)

void random_fill_LR(int nU, int nI, int nF)
{
	std::srand(0);

	for (int i = 0; i < nU; i++)
		for (int j = 0; j < nF; j++)
			L[i][j] = RAND01 / (double) nF;

	for (int i = 0; i < nF; i++)
		for (int j = 0; j < nI; j++)
			L[i][j] = RAND01 / (double) nF;
}


// type used by sparse matrix represantation
struct entry_t
{
	int idx;
	double elem;
};

// type for sparse matrix (List of list represantation)
using sparse_mat = std::vector<std::vector<entry_t>>;


// Creates a matrix from a file
sparse_mat parse_file(const char* filename)
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

	sparse_mat A(nU);
	for (int i = 0; i < n_elems; i++)
	{
		int row, col;
		double elem;

		file >> row >> col >> elem;

		A[row].push_back({col, elem});
	}

	file.close();
	
	return A;
}



int main(int argc, char** argv)
{
	sparse_mat A = parse_file(argv[1]);

	for (int row = 0; row < nU; row++)
		for (auto&& entry : A[row])
			std::cout << entry.idx << ' ' << entry.elem << '\n';

	std::cout.flush();
}
