#include <stdlib.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <omp.h>


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
    
    matrix& swap(matrix& rhs)
    {
        std::swap(n_rows,rhs.n_rows);
        std::swap(n_columns,rhs.n_columns);
        std::swap(data, rhs.data);
        return *this;
    }
    
public:
    int n_rows, n_columns;
    
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
        idx_pair temp = {column, elem_idx};
        rows[row].push_back(temp);
        idx_pair temp2 = {row, elem_idx};
        columns[column].push_back(temp2);
    }
    
    struct idx_pair
    {
        int idx;
        int elem_idx;
    };
    
    int n_rows, n_columns;
    std::vector<double> elements;
    std::vector< std::vector<idx_pair> > rows;
    std::vector< std::vector<idx_pair> > columns;
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
    std::cout << omp_get_thread_num();
    
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

// Calculate B_ij
double B(int i, int j, const matrix& L, const matrix& R)
{
    double elem = .0;
    for(int k = 0; k < nF; k++) {
        elem += L[i][k]*R[k][j];
    }
    return elem;
}

void update_LR(const sparse_matrix& A, matrix& L, matrix& R, matrix& oldL, matrix& oldR)
{
    L.swap(oldL);
    R.swap(oldR);
    // (*L)[] - to access an element in L
    
    for (int i = .0; i < nU; i++)
        for (int k = .0; k < nF; k++)
        {
            double deltaL = .0;
            for (auto&& entry : A.rows[i])
                deltaL += -2*(A.elements[entry.elem_idx] -B(i,entry.idx, L, R))*oldR[k][entry.idx];
            L[i][k] = oldL[i][k] - learning_rate*deltaL;
        }
    
    for (int k = .0; k < nF; k++)
        for (int j = .0; j < nI; j++)
        {
            double deltaR = .0;
            for (auto&& entry : A.columns[j])
                deltaR += -2*(A.elements[entry.elem_idx] -B(entry.idx, j, L, R))*oldL[entry.idx][k];
            R[k][j] = oldR[k][j] - learning_rate*deltaR;
        }
}

void result(const sparse_matrix& A, const matrix& L, const matrix& R)
{
    double max,col;
    std::cout << "\n";
    for(int i = 0; i < nU; i++) {
        max = 0;
        col = 0;
        auto it = A.rows[i].begin();
        
        for(int j = 0; j < nI; j++) {
            if (j == it->idx)
            {
                if (it != A.rows[i].end()) it++;
                continue;
            }
            
            double b = B(i,j, L, R);
            if (b > max) {
                max = b;
                col = j;
            }
        }
        std::cout << col << "\n";
    }
}

int main(int argc, char** argv)
{
    std::srand(0);
    sparse_matrix A = parse_file(argv[1]);
    matrix L(nU, nF);
    matrix oldL(nU, nF);
    matrix R(nF, nI);
    matrix oldR(nF, nI);
    
    #pragma omp parallel sections
    {
        #pragma omp section
        random_fill_matrix(L);
        
        #pragma omp section
        random_fill_matrix(R);
    }
    
    for (int i = 0; i < n_iter; i++) update_LR(A, L, R, oldL, oldR);
    
    result(A, L, R);
}


