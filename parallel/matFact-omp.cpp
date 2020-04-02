#include <stdlib.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <omp.h>

constexpr int cache_line = 64 / sizeof(double);

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
    
    double& operator()(int row, int col) { return data[row * row_size + col]; }
    
    double operator()(int row, int col) const { return data[row * row_size + col]; }
    
    matrix& operator=(const matrix& m)
    {
        std::copy(m.data.begin(), m.data.end(), this->data.begin());
        return *this;
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
    float deltaL, deltaR, temp;
    
    #pragma omp parallel private(deltaL, deltaR, temp)
    {
        #pragma omp sections
        {
            #pragma omp section
            oldL = L;

            #pragma omp section
            oldRt = Rt;
        }

        #pragma omp barrier

        #pragma omp for nowait
        for (int i = 0; i < nU; i++)
        {
            int j;
            auto& row = A.rows[i];
            for (auto& c : row)
            {
                double rating = A.elements[c.elem_idx];
                j = c.idx;
                temp = rating - B(i, j, oldL, oldRt);
                for (int k = 0; k < nF; k++)
                {
                    deltaL = -2 * temp * oldRt(j, k);
                    L(i, k) -= learning_rate * deltaL;
                }
            }
        }

        #pragma omp for schedule(dynamic, 10)
        for (int j = 0; j < nI; j++)
        {
            int i;
            auto& col = A.columns[j];
            for (auto& r : col)
            {
                double rating = A.elements[r.elem_idx];
                i = r.idx;
                temp = rating - B(i, j, oldL, oldRt);
                for (int k = 0; k < nF; k++)
                {
                    deltaR = -2 * temp * oldL(i, k);
                    Rt(j, k) -= learning_rate * deltaR;
                }
            }
        }
    } // pragma omp parallel
}

void result(const sparse_matrix& A, const matrix& L, const matrix& Rt)
{
    double max;
    int col;
    for(int i = 0; i < nU; i++) {
        max = 0;
        col = 0;
        auto it = A.rows[i].begin();
        auto it_end = A.rows[i].end();
        bool finished_row = it == it_end;
        
        for(int j = 0; j < nI; j++) {
            if (!finished_row && j == it->idx)
            {
                ++it;
                finished_row = it == it_end;
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

    for (int i = 0; i < n_iter; i++){
        update_LR(A, L, Rt, oldL, oldRt);
    }
        
    //std::cout << "L:\n" << L << "\nR:\n" << Rt;

    result(A, L, Rt);
}