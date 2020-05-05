#include <stdlib.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <mpi.h>


constexpr int GlobalsTag = 0;
constexpr int SizeTag = 0;
constexpr int DataTag = 0;

constexpr int cache_line = 64 / sizeof(double);

// globals
struct {
int n_iter;
int nF;
int nU;
int nI;
double learning_rate;
} globals;

int id, p;
int* proc_rows = nullptr;
int rows_per_proc;
int extra_rows;

/** Sparse matrix type
 */
class sparse_matrix
{
public:
    sparse_matrix() = default;

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
	
	friend void swap(matrix& a, matrix& b)
	{
		std::swap(a.n_rows, b.n_rows);
		std::swap(a.n_columns, b.n_columns);
		std::swap(a.row_size, b.row_size);
		std::swap(a.data, b.data);
	}

public:
    int n_rows, n_columns;
    int row_size;
    
    std::vector<double> data;
};

// Creates a matrix from a file
int parse_file_distribute_A(const char* filename, sparse_matrix& A)
{
    std::ifstream file(filename);
    
    if (file.fail())
    {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(1);
    }
    
    int n_elems;
    
    file >> globals.n_iter;
    file >> globals.learning_rate;
    file >> globals.nF;
    file >> globals.nU >> globals.nI >> n_elems;
	
	rows_per_proc = globals.nU / p;
	extra_rows = globals.nU % p;
	
	int row = 0, col = 0;
    double elem;
    std::vector<sparse_matrix::entry_t> sendingData;
	int n_rows = 0;
	int rows_to_send = rows_per_proc + (extra_rows == 0 ? 0 : 1);
	int id = 0;
    proc_rows[0] = rows_to_send;
	for (int i = 0; i < n_elems; i++)
    {
		int prev_row = row;
		
        file >> row >> col >> elem;

		if (prev_row != row)
		{
			n_rows += row - prev_row;
			
			if (n_rows >= rows_to_send)
			{	
				if (id != 0)
				{
					int size[] = {(int) sendingData.size(), rows_to_send};
					
					MPI_Send(size, 2, MPI_INT, id, SizeTag, MPI_COMM_WORLD);
					if (size[0] != 0) MPI_Send(sendingData.data(), size[0] * sizeof(sparse_matrix::entry_t), MPI_BYTE, id, DataTag, MPI_COMM_WORLD);
				}
				
				sendingData.clear();
				n_rows -= rows_to_send ;
				id++;
				rows_to_send = rows_per_proc + (id >= extra_rows ? 0 : 1);
				proc_rows[id] = rows_to_send;
			}
        }
		
        sendingData.push_back({row, col, elem});
		A.add_element(row, col, elem);
    }
    
	//Send to last processors
	int size[] = {(int) sendingData.size(), rows_to_send};
	
	MPI_Send(size, 2, MPI_INT, id, SizeTag, MPI_COMM_WORLD);
	if (size[0] != 0) MPI_Send(sendingData.data(), size[0] * sizeof(sparse_matrix::entry_t), MPI_BYTE, id, DataTag, MPI_COMM_WORLD);
	
	id++;
	size[0] = 0;
	size[1] = 0;
	for (; id < p; id++)
	{
		proc_rows[id] = 0;
		MPI_Send(size, 2, MPI_INT, id, SizeTag, MPI_COMM_WORLD);
	}

    file.close();
    
    return proc_rows[0];
}

void initialize_distribute_LR(matrix& L, matrix& Rt, matrix& bufferL)
{
	#define RAND01 ((double) std::rand() / (double) RAND_MAX)

	if (id == 0)
	{
		for (int i = 0; i < p; ++i)
		{
			for (int r = 0; r < proc_rows[i]; r++)
				for (int c = 0; c < globals.nF; c++)
					bufferL(r, c) = RAND01 / (double) globals.nF;
				
			if (i == 0) swap(L, bufferL);
			else if (proc_rows[i] != 0) MPI_Send(bufferL.data.data(), proc_rows[i] * bufferL.row_size, MPI_DOUBLE, i, DataTag, MPI_COMM_WORLD);
		}

		for (int r = 0; r < globals.nF; r++)
			for (int c = 0; c < globals.nI; c++)
				Rt(c, r) = RAND01 / (double) globals.nF;
	}
	else if (L.data.size() != 0)
	{
		MPI_Status status;
		MPI_Recv(L.data.data(), L.data.size(), MPI_DOUBLE, 0, DataTag, MPI_COMM_WORLD, &status);
	}
	
	// Broadcast Rt
	MPI_Bcast(Rt.data.data(), Rt.data.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void exchangeRt(matrix& Rt, double* buffer)
{
	MPI_Request request;
	MPI_Status status;
	int size = Rt.n_rows * Rt.row_size;
	
	// Exchange messages in hypercube topology
	for (int neighbor_mask = 0b1; neighbor_mask < p; neighbor_mask <<= 1)
	{
		int neighbor_id = (id & ~neighbor_mask) | (~id & neighbor_mask);

		MPI_Isend(Rt.data.data(), size, MPI_DOUBLE, neighbor_id, DataTag, MPI_COMM_WORLD, &request);
		MPI_Recv(buffer, size, MPI_DOUBLE, neighbor_id, DataTag, MPI_COMM_WORLD, &status);
		MPI_Wait(&request, &status);
		
		// Update Rt before sending to next naighbor
		for (int i = 0; i < size; ++i) Rt.data[i] += buffer[i];
	}
}

void gatherL(matrix& FullL, matrix& L)
{	
	int* proc_sizes;
	int* proc_offsets;

	if (id == 0)
	{
		proc_sizes = new int[p];
		proc_offsets = new int[p];
		
		proc_sizes[0] = L.n_rows * L.row_size;
		proc_offsets[0] = 0;
		for (int i = 1; i < p; ++i)
		{
			proc_sizes[i] = proc_rows[i] * L.row_size;
			proc_offsets[i] = proc_sizes[i-1] + proc_offsets[i-1];
		}
	}
	
	int size = L.n_rows * L.row_size;
	MPI_Gatherv(L.data.data(), size, MPI_DOUBLE, FullL.data.data(), proc_sizes, proc_offsets, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}


// Calculate B_ij
double B(int i, int j, const matrix& L, const matrix& Rt)
{
    double elem = .0;
    for(int k = 0; k < globals.nF; k++) {
        elem += L(i, k)*Rt(j, k);
    }
    return elem;
}

void update_LR(const sparse_matrix& A, matrix& L, matrix& Rt, matrix& oldL, matrix& oldRt)
{
    swap(L, oldL);
    swap(Rt, oldRt);
	
	for (int i = 0; i < L.n_rows; ++i)
		for (int j = 0; j < L.n_columns; ++j)
			L(i, j) = 0;

	for (int i = 0; i < Rt.n_rows; ++i)
		for (int j = 0; j < Rt.n_columns; ++j)
			Rt(i, j) = 0;
    
	float deltaL, deltaR, temp;
    for (auto& entry : A.elements)
    {
		if (entry.row >= L.n_rows) break;
		
        temp = entry.rating - B(entry.row, entry.col, oldL, oldRt);
        for (int k = 0; k < globals.nF; k++)
        {
            deltaL = -2 * temp * oldRt(entry.col, k);
            L(entry.row, k) -= globals.learning_rate * deltaL;

            deltaR = -2 * temp * oldL(entry.row, k);
            Rt(entry.col, k) -= globals.learning_rate * deltaR;
        }
    }
}

void result(const sparse_matrix& A, const matrix& L, const matrix& Rt)
{
    double max;
    int col;
    auto it = A.elements.begin();
    auto it_end = A.elements.end();
    
    for(int i = 0; i < globals.nU; i++) {
        max = 0;
        col = 0;
        bool finished_row = (it == it_end || it->row > i);

        for(int j = 0; j < globals.nI; j++) {
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

int main(int argc, char* argv[])
{
	std::srand(0);
	int n_rows;
	sparse_matrix A;
	double elapsed_time;
	
	MPI_Init(&argc, &argv);
	
	elapsed_time = -MPI_Wtime();
	
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
	
	proc_rows = new int[p];
	
	if (id == 0) n_rows = parse_file_distribute_A(argv[1], A);
	else
	{
		MPI_Status status;
		int size[2];
		MPI_Recv(size, 2, MPI_INT, 0, SizeTag, MPI_COMM_WORLD, &status);
		
		n_rows = size[1];
		
		sparse_matrix::entry_t* buffer;
		
		if (size[0] != 0)
		{
			buffer = new sparse_matrix::entry_t[size[0]];
			
			MPI_Recv(buffer, size[0] * sizeof(sparse_matrix::entry_t), MPI_BYTE, 0, DataTag, MPI_COMM_WORLD, &status);
			
			int offset = buffer[0].row;
			
			for (int i = 0; i < size[0]; ++i)
			{
				A.add_element(buffer[i].row - offset, buffer[i].col, buffer[i].rating);
			}
			
			delete[] buffer;
		}
	}
	
	
	//Send globals
	MPI_Bcast(&globals, sizeof(globals), MPI_BYTE, 0, MPI_COMM_WORLD);
    
	matrix L(n_rows, globals.nF);
    matrix oldL(n_rows, globals.nF);
    matrix Rt(globals.nI, globals.nF);
    matrix oldRt(globals.nI, globals.nF);
	double* bufferRt = new double[Rt.n_rows * Rt.row_size];
	
	initialize_distribute_LR(L, Rt, oldL);
	
	for (int i = 0; i < globals.n_iter; i++)
	{
		update_LR(A, L, Rt, oldL, oldRt);
		
		exchangeRt(Rt, bufferRt);
		
		for (int i = 0; i < Rt.n_rows; ++i)
			for (int j = 0; j < Rt.n_columns; ++j)
				Rt(i, j) += oldRt(i, j);
			
		for (int i = 0; i < L.n_rows; ++i)
			for (int j = 0; j < L.n_columns; ++j)
				L(i, j) += oldL(i, j);
	}
	
	matrix FullL(globals.nU, globals.nF);
	
	gatherL(FullL, L);

	if (id == 0)
	{
		//std::cout << FullL << "\n\n" << Rt << std::endl;
		result(A, FullL, Rt);
	}
	
	delete[] proc_rows;
	delete[] bufferRt;
	
	elapsed_time += MPI_Wtime();
	if (id == 0) std::cout << "Elapsed time: " << elapsed_time << std::endl;
	
	MPI_Finalize();
	
	return 0;
}