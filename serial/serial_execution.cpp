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
double** L;
double** R;
double** oldL;
double** oldR;
double** B;


#define RAND01 ((double) std::rand() / (double) RAND_MAX)

void random_fill_LR(int nU, int nI, int nF)
{
    std::srand(0);
    
    for (int i = 0; i < nU; i++)
        for (int j = 0; j < nF; j++)
            L[i][j] = RAND01 / (double) nF;
    
    for (int i = 0; i < nF; i++)
        for (int j = 0; j < nI; j++)
            R[i][j] = RAND01 / (double) nF;
}


// type used by sparse matrix represantation
struct entry_t
{
    int idx;    //column
    double elem;
};

// type for sparse matrix (List of list represantation)
using sparse_mat = std::vector< std::vector<entry_t> >;


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
        int row, col;           //Can this be done outside the for-loop?
        double elem;
        
        file >> row >> col >> elem;
        
        entry_t temp = {col,elem};
        A[row].push_back(temp);//{col, elem});
    }
    
    file.close();
    
    return A;
}

// Initializes the lists
void initialize(double** mat, int a, int b) {
    mat = (double**) calloc(a, sizeof(double*));
    for (int i = 0; i < a; i++) {
        mat[i] = (double*) calloc(b, sizeof(double));
    }
}

// Matrix multiplication, B = L*R
void makeB() {
    for(int i = 0; i < nU; i++) {
        for(int j = 0; j < nI; j++) {
            B[i][j] = 0;
            for(int k = 0; k < nF; k++) {
                B[i][j] += L[i][k]*R[k][j];
            }
        }
    }
}

// Calculate a part of update of R
double find_sum(int j, int k, sparse_mat A) {
    double temp;
    for(int i = 0; i < nU; i++) {
        for (auto&& entry : A[i]) {
            if (j == entry.idx) {
                temp += -2*(entry.elem - B[i][j])*oldL[i][k];
            }
        }
    }
    return temp;
}

// Update matrices L and R
void update_LR(sparse_mat A) {
    for(int i = 0; i < n_iter; i++)  { //n_iter
        // Make new copies of L and R
        for(int i = 0; i < nU; i++) {
            for (int j = 0; j < nF; j++) {
                oldL[i][j] = L[i][j];
            }
        }
        for(int i = 0; i < nF; i++) {
            for (int j = 0; j < nI; j++) {
                oldR[i][j] =  R[i][j];
            }
        }
        
        // Update L
        double temp;
        for (int i = 0; i < nU; i++) { //row
            for(int k = 0; k < nF; k++) {
                temp = 0;
                for (auto&& entry : A[i]) { //column
                    temp += -2*(entry.elem - B[i][entry.idx])*oldR[k][entry.idx];
                }
                L[i][k] = oldL[i][k] - learning_rate*temp;
            }
        }
        
        /*
        // Print L matrix
        std::cout << "Print L: \n";
        for (int row = 0; row < nU; row++) {
            if(row != 0) {
                std::cout << "\n";
            }
            for (int col = 0; col < nF; col++) {
                std::cout << L[row][col] << ' ';
            }
        }
        std::cout << "\n\n";
        */
        
        // Update R
        for (int i = 0; i < nU; i++) { //row
            for (auto&& entry : A[i]) { //column
                temp = 0;
                for(int k = 0; k < nF; k++) {
                    temp = find_sum(entry.idx,k,A);
                    R[k][entry.idx] = oldR[k][entry.idx] - learning_rate*temp;
                }
            }
        }
        
        /*
        // Print R matrix
        std::cout << "Print R: \n";
        for (int row = 0; row < nF; row++) {
            if(row != 0) {
                std::cout << "\n";
            }
            for (int col = 0; col < nI; col++) {
                std::cout << R[row][col] << ' ';
            }
        }
        std::cout << "\n\n";
        */
        
        // Make B
        makeB();
        
        if(i == (n_iter-1)) {
            // Print B matrix
            std::cout << "Print B: \n";
            for (int i = 0; i < nU; i++) {
                if(i != 0) {
                    std::cout << "\n";
                }
                for (int j = 0; j < nI; j++) {
                    std::cout << B[i][j] << ' ';
                }
            }
            std::cout << "\n\n";
        }
    }
}

// Prints result
void result() {
    double max,col;
    std::cout << "\n";
    for(int i = 0; i < nU; i++) {
        max = 0;
        col = 0;
        for(int j = 0; j < nI; j++) {
            if (B[i][j] > max) {
                max = B[i][j];
                col = j;
            }
        }
        std::cout << col << "\n";
    }
}



int main(int argc, char** argv)
{
    sparse_mat A = parse_file(argv[1]);
    
    /*
// Print A matrix
    for (int row = 0; row < nU; row++)
        for (auto&& entry : A[row])
            std::cout << row << ' ' << entry.idx << ' ' << entry.elem << '\n';
    */
    
    /*
    //Initialize, but does not work yet, why???
    initialize(L,nU,nF);
    initialize(R, nF, nI);
    initialize(oldL,nU,nF);
    initialize(oldR, nF, nI);
    initialize(B,nU,nI);
    */
    
    
    //Initialize L
    L = (double**) calloc(nU, sizeof(double*));
    for (int i = 0; i < nU; i++) {
        L[i] = (double*) calloc(nF, sizeof(double));
    }
    //Initialize R
    R = (double**) calloc(nF, sizeof(double*));
    for (int i = 0; i < nF; i++) {
        R[i] = (double*) calloc(nI, sizeof(double));
    }
    //Initialize oldL
    oldL = (double**) calloc(nU, sizeof(double*));
    for (int i = 0; i < nU; i++) {
        oldL[i] = (double*) calloc(nF, sizeof(double));
    }
    //Initialize oldR
    oldR = (double**) calloc(nF, sizeof(double*));
    for (int i = 0; i < nF; i++) {
        oldR[i] = (double*) calloc(nI, sizeof(double));
    }
    //Initialize B
    B = (double**) calloc(nU, sizeof(double*));
    for (int i = 0; i < nU; i++) {
        B[i] = (double*) calloc(nI, sizeof(double));
    }
    
    //Fill L and R with random numbers
    random_fill_LR(nU, nI, nF);
    
    // Make B matrix
    makeB();
    
    /*
    // Print L matrix
    std::cout << "Print L: \n";
    for (int i = 0; i < nU; i++)
        for (int k = 0; k < nF; k++) {
            std::cout << i << ' ' << k << ' ' << L[i][k] << '\n';
        }
     */
    /*
    // Print B matrix
    std::cout << "Print B: \n";
    for (int i = 0; i < nU; i++) {
        if(i != 0) {
            std::cout << "\n";
        }
        for (int j = 0; j < nI; j++) {
            std::cout << B[i][j] << ' ';
            //if ((j % nI) == 0 && j > 0) {
            //    std::cout << '\n';
            
        }
    }
    std::cout << "\n";
 */
    
    // Update matrices L and R
    update_LR(A);
    
    // Print result
    result();
    
    std::cout.flush();
}

