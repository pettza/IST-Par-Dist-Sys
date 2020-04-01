#include <stdlib.h>
#include <vector>
#include <fstream>
#include <iostream>

constexpr int cache_line = 64 / sizeof(double);


/** Sparse matrix type
 */
class sparse_matrix
{
public:
    sparse_matrix(int n_rows, int n_columns)
    : n_rows(n_rows), n_columns(n_columns)
    {}

    void add_element(int row, int column, double rating)
    {
        elements.push_back({row, column, rating});
    }

    struct entry_t
    {
        int row;
        int col;
        double rating;
    };
    
    int n_rows, n_columns;
    std::vector<entry_t> elements;
};


/** Matrix type
 */
class matrix
{
public:
    matrix(int n_rows, int n_columns)
    : n_rows(n_rows), n_columns(n_columns),
      row_size((n_columns / cache_line + 1) * cache_line),
      data(n_rows * row_size)
    {}
    
    matrix(const sparse_matrix& m)
    : n_rows(m.n_rows), n_columns(m.n_columns),
      row_size((n_columns / cache_line + 1) * cache_line),
      data(n_rows * row_size, 0.f)
    {
        for (auto& entry : m.elements) (*this)(entry.row, entry.col) = entry.rating;
    }

    double& operator()(int row, int col) { return data[row * row_size + col]; }
    
    double operator()(int row, int col) const { return data[row * row_size + col]; }

    matrix& operator=(const matrix& m) 
    {
        for (int i = 0; i < data.size(); i++)
            this->data[i] = m.data[i];
    }

    friend std::ostream& operator<<(std::ostream& s, const matrix& m)
    {
        for (int r = 0; r < m.n_rows; r++)
        {
            for (int c = 0; c < m.n_columns; c++)
                s << m(r, c) << '\t';
            s << '\n';
        }
        
        return s;
    }

public:
    int n_rows, n_columns;
    int row_size;
    
private:
    std::vector<double> data;
};

// globals
int n_iter;
int nF;
int nU;
int nI;
double learning_rate;


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
        double rating;
        
        file >> row >> col >> rating;
        
        A.add_element(row, col, rating);
    }
    
    file.close();
    
    return A;
}

// Calculate B_ij
double B(int i, int j, const matrix& L, const matrix& Rt)
{
    double elem = .0;
    for(int k = 0; k < nF; k++) {
        elem += L(i, k)*Rt(j, k);
    }
    return elem;
}

void update_LR(const sparse_matrix& A, matrix& L, matrix& Rt, matrix& oldL, matrix& oldRt)
{
    oldL = L;
    oldRt = Rt;

    float deltaL, deltaR, temp;
    for (auto& entry : A.elements)
    {
        temp = entry.rating - B(entry.row, entry.col, oldL, oldRt);
        for (int k = 0; k < nF; k++)
        {
            deltaL = -2 * temp * oldRt(entry.col, k);
            L(entry.row, k) -= learning_rate * deltaL;

            deltaR = -2 * temp * oldL(entry.row, k);
            Rt(entry.col, k) -= learning_rate * deltaR;
        }
    }
}

void result(const sparse_matrix& A, const matrix& L, const matrix& Rt)
{
    double max;
    int col;
    auto it = A.elements.begin();
    auto it_end = A.elements.end();
    
    for(int i = 0; i < nU; i++) {
        max = 0;
        col = 0;
        bool finished_row = (it == it_end || it->row > i);

        for(int j = 0; j < nI; j++) {
            if (!finished_row && j == it->col)
            {
                it++;
                finished_row = (it == it_end || it->row > i);
                continue;
            }
            
            double b = B(i,j, L, Rt);
            if (b > max) {
                max = b;
                col = j;
            }
        }
        std::cout << col << "\n";
    }
}

#define RAND01 ((double) std::rand() / (double) RAND_MAX)

int main(int argc, char** argv)
{
    std::srand(0);
    sparse_matrix A = parse_file(argv[1]);
    matrix L(nU, nF);
    matrix oldL(nU, nF);
    matrix Rt(nI, nF);
    matrix oldRt(nI, nF);
    
    for (int r = 0; r < nU; r++)
        for (int c = 0; c < nF; c++)
            L(r, c) = RAND01 / (double) nF;

    for (int r = 0; r < nF; r++)
        for (int c = 0; c < nI; c++)
            Rt(c, r) = RAND01 / (double) nF;
    
    //std::cout << "L:\n" << L << "\nR:\n" << Rt;

    //std::cout << matrix(A) << std::endl;

    for (int i = 0; i < n_iter; i++)
    {
        update_LR(A, L, Rt, oldL, oldRt);
    }
    
    //std::cout << "L:\n" << L << "\nR:\n" << Rt;

    result(A, L, Rt);
}
