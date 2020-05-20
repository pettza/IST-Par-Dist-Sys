#include <fstream>
#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>
#include <mpi.h>

constexpr int GlobalsTag = 0;
constexpr int SizeTag = 1;
constexpr int DataTag = 2;

constexpr int cache_line = 64 / sizeof(double);

// globals
int n_iter;
int nF;
int nU;
int nI;
double learning_rate;

// The id of the of this task, the id of the root task and the number of tasks
int id, root_id, p;

// The dimensions of the task grid
int dimensions[2];

// The coordinates on the grid of this task
int cart_coords[2];

// Arrays with the number of rows/columns and offsets to the full matrices
// for each grid row/column number
std::unique_ptr<int[]> proc_rows(nullptr);
std::unique_ptr<int[]> proc_row_offsets(nullptr);
std::unique_ptr<int[]> proc_columns(nullptr);
std::unique_ptr<int[]> proc_column_offsets(nullptr);

// Cartesian communicator
MPI_Comm Cart_Comm;

// Communicators for tasks on the same grid row/column
MPI_Comm Rows_Comm, Columns_Comm;

/** Sparse matrix type
 */
class sparse_matrix
{
public:
	sparse_matrix(int n_rows = 0, int n_columns = 0)
	: n_rows(n_rows), n_columns(n_columns)
	{}
	
	void add_element(int row, int column, double rating)
	{
		elements.push_back({row, column, rating});
	}
	
	void reset(int n_rows, int n_columns)
	{
		this->n_rows = n_rows;
		this->n_columns = n_columns;
		elements.clear();
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
	data(new double[n_rows * row_size])
	{}
	
	matrix(const sparse_matrix& m)
	: n_rows(m.n_rows), n_columns(m.n_columns),
	row_size((n_columns / cache_line + 1) * cache_line),
	data(new double[n_rows * row_size])
	{
		for (int r = 0; r < n_rows; r++)
			for (int c = 0; c < n_columns; c++)
				(*this)(r, c) = 0;

		for (auto& e : m.elements)
			(*this)(e.row, e.col) = e.rating;
	}
	
	double& operator()(int row, int col) { return data[row * row_size + col]; }
	
	double operator()(int row, int col) const { return data[row * row_size + col]; }
	
	double* getRow(int row) { return data.get() + row * row_size; }
	
	const double* getRow(int row) const { return data.get() + row * row_size; }

	double* raw() { return data.get(); }
	
	matrix& operator=(const matrix& m) 
	{
		std::copy(m.data.get(), m.data.get() + m.n_rows * m.row_size, this->data.get());
		return *this;
	}
	
	friend std::ostream& operator<<(std::ostream& s, const matrix& m)
	{
		for (int r = 0; r < m.n_rows; r++) {
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
		a.data.swap(b.data);
	}
	
	int n_rows, n_columns;
	int row_size;

private:
	std::unique_ptr<double[]> data;
};

// Calculate dimensions so that the number of rows per taks is as
// close as possible to the number of columns per task
void calculate_dimesions()
{
	int min_diff = std::abs(nU / p - nI);
	dimensions[0] = p;
	dimensions[1] = 1;

	for (int i = 2; i <= p; ++i) {
		if (p % i == 0) {
			int j = p / i;
			int diff = std::abs(nU / j - nI / i);
			if (diff < min_diff) {
				diff = min_diff;
				dimensions[0] = j;
				dimensions[1] = i;
			}
		}
	}
}

// Calculate topology and partitioning info
void initialize_topology_info()
{
	// Create 2D cartesian grid
    MPI_Comm_size(MPI_COMM_WORLD, &p);
	calculate_dimesions();
	int periodics[] = {0, 0};
	MPI_Cart_create(MPI_COMM_WORLD, 2, dimensions, periodics, 1, &Cart_Comm);
	MPI_Comm_rank(Cart_Comm, &id);
	MPI_Cart_coords(Cart_Comm, id, 2, cart_coords);
	int root_coords[] = {0, 0};
	MPI_Cart_rank(Cart_Comm, root_coords, &root_id);
	MPI_Comm_split(Cart_Comm, cart_coords[0], cart_coords[1], &Rows_Comm);
	MPI_Comm_split(Cart_Comm, cart_coords[1], cart_coords[0], &Columns_Comm);
}

void initialize_partitioning_info()
{
	// Calculate number of rows a node manupulates
	proc_rows.reset(new int[dimensions[0]]);
	proc_row_offsets.reset(new int[dimensions[0]]);
	int rows_per_proc = nU / dimensions[0];
	int extra_rows = nU % dimensions[0];
	proc_rows[0] = rows_per_proc + (extra_rows == 0 ? 0 : 1);
	proc_row_offsets[0] = 0;
	for(int i = 1; i < dimensions[0]; ++i) {
		proc_rows[i] = rows_per_proc + (i >= extra_rows ? 0 : 1);
		proc_row_offsets[i] = proc_row_offsets[i-1] + proc_rows[i-1];
	}

	// Calculate number of column a node manupulates
	proc_columns.reset(new int[dimensions[1]]);
	proc_column_offsets.reset(new int[dimensions[1]]);
	int columns_per_proc = nI / dimensions[1];
	int extra_columns = nI % dimensions[1];
	proc_columns[0] = columns_per_proc + (extra_columns == 0 ? 0 : 1);
	proc_column_offsets[0] = 0;
	for(int j = 1; j < dimensions[1]; ++j) {
		proc_columns[j] = columns_per_proc + (j >= extra_columns ? 0 : 1);
		proc_column_offsets[j] = proc_column_offsets[j-1] + proc_columns[j-1]; 
	}
}

// Creates a matrix from a file
void parse_file_initialize_info(const char* filename, sparse_matrix& A)
{
	std::ifstream file(filename);
	int n_elements;

	if (file.fail()) {
		std::cerr << "Failed to open file: " << filename << std::endl;
		exit(1);
	}
		
	file >> n_iter;
	file >> learning_rate;
	file >> nF;
	file >> nU >> nI >> n_elements;
	
	initialize_topology_info();
	initialize_partitioning_info();
	
	A.reset(proc_rows[cart_coords[0]], proc_columns[cart_coords[1]]);

	int lower_row_limit = proc_row_offsets[cart_coords[0]];
	int upper_row_limit = lower_row_limit + proc_rows[cart_coords[0]];
	
	int lower_column_limit = proc_column_offsets[cart_coords[1]];
	int upper_column_limit = lower_column_limit + proc_columns[cart_coords[1]];

	int row, column;
	double rating;
	for (int i = 0; i < n_elements; i++) {
		// Read next element
		file >> row >> column >> rating;

		// Keep element only if its row and column are in the range this task is responsible for
		if (lower_row_limit <= row && row < upper_row_limit && 
			lower_column_limit <= column && column < upper_column_limit)
			A.add_element(row - lower_row_limit, column - lower_column_limit, rating);
	}

	file.close();
}


// L and R are initialized by the root task so that they are the same as in the serial case
void initialize_LR(matrix& L, matrix& bufferL, matrix& Rt, double* buffer)
{	
	#define RAND01 ((double) std::rand() / (double) RAND_MAX)
	
	if (id == root_id)
	{
		// The first rows of row don't need to be sent to other task,
		// since they are the ones the root task manages
		double* Lr;
		for (int r = 0; r < L.n_rows; ++r) {
			Lr = L.getRow(r);
			for (int c = 0; c < nF; c++)
				Lr[c] = RAND01 / (double) nF;
		}
		
		// Send the rows of L to the task with the corresponding row coordinate and
		// column coordinate equal to 0
		// Cannot send all rows at once because row_size in the matrix class may be
		// different due to padding for cache alignment
		double* bufferLr;
		for (int coords[] = {1, 0}; coords[0] < dimensions[0]; ++coords[0]) {
			for (int r = 0; r < proc_rows[coords[0]]; ++r) {
				bufferLr = bufferL.getRow(r);
				for (int c = 0; c < nF; ++c)
					bufferLr[c] = RAND01 / (double) nF;
				
				MPI_Send(bufferLr, nF, MPI_DOUBLE, coords[0], DataTag, Columns_Comm);
			}
		}
		
		// Do the same for R (not Rt)
		for (int r = 0; r < nF; ++r) {
			for (int coords[] = {0, 0}; coords[1] < dimensions[1]; ++coords[1]) {
				if (coords[1] == 0) 
					for (int c = 0; c < proc_columns[0]; ++c)
						Rt(c, r) = RAND01 / (double) nF;
				else{
					for (int c = 0; c < proc_columns[coords[1]]; ++c)
						buffer[c] = RAND01 / (double) nF;
					
					MPI_Send(buffer, proc_columns[coords[1]], MPI_DOUBLE, coords[1], DataTag, Rows_Comm);
				}
			}
		}
	}
	// If row coordinate is zero then receive Rt
	else if (cart_coords[0] == 0) {
		MPI_Status status;
		for (int r = 0; r < Rt.n_columns; ++r) {
			MPI_Recv(buffer, Rt.n_rows, MPI_DOUBLE, 0, DataTag, Rows_Comm, &status);
        	
			for (int c = 0; c < Rt.n_rows; c++)
            	Rt(c, r) = buffer[c];
		}
	}
	// If column coordinate is zero then receive L
	else if (cart_coords[1] == 0) {
		MPI_Status status;
		for (int r = 0; r < L.n_rows; ++r)
			MPI_Recv(L.getRow(r), nF, MPI_DOUBLE, 0, DataTag, Columns_Comm, &status);
	}

	// Broadcast to the rest of the grid
	MPI_Bcast(L.raw(), L.n_rows * L.row_size, MPI_DOUBLE, 0, Rows_Comm);
	MPI_Bcast(Rt.raw(), Rt.n_rows * Rt.row_size, MPI_DOUBLE, 0, Columns_Comm);

	#undef RAND01
}

// Perform all-reduce operations across grid row and column to accumulate the updates
void reduce_LR(matrix& L, matrix& bufferL, matrix& Rt, matrix& bufferRt)
{
	MPI_Request requests[2];
	MPI_Status statuses[2];
	int sizeL = L.n_rows * L.row_size;
	int sizeRt = Rt.n_rows * Rt.row_size;
	
	MPI_Iallreduce(L.raw(), bufferL.raw(), sizeL, MPI_DOUBLE, MPI_SUM, Rows_Comm, requests);
	MPI_Iallreduce(Rt.raw(), bufferRt.raw(), sizeRt, MPI_DOUBLE, MPI_SUM, Columns_Comm, requests + 1);
	
	MPI_Waitall(2, requests, statuses);
}

// Calculate B_ij
double B(int i, int j, const matrix& L, const matrix& Rt)
{
	double elem = .0;
	const double* Li = L.getRow(i);
	const double* Rtj = Rt.getRow(j);
	for(int k = 0; k < nF; k++)
	{
		elem += Li[k] * Rtj[k];
	}

	return elem;
}

// Update L and R using the elements of A this task is responsible for
void update_LR(const sparse_matrix& A, matrix& L, matrix& Rt, matrix& oldL, matrix& oldRt)
{
	swap(L, oldL);
	swap(Rt, oldRt);
	

	// In contrast to the serial case the matrices need to be zeroed because
	// otherwise the all-reduce operations would accumulate their elements
	// multiple times
	for (int i = 0; i < L.n_rows; ++i) {
		double* Li = L.getRow(i);
		for (int j = 0; j < L.n_columns; ++j)
			Li[j] = 0;
	}
	
	for (int j = 0; j < Rt.n_rows; ++j) {
		double* Rtj = Rt.getRow(j);
		for (int i = 0; i < Rt.n_columns; ++i)
			Rtj[i] = 0;
	}
	
	float deltaL, deltaR, temp;
	for (auto& entry : A.elements)
	{
		temp = entry.rating - B(entry.row, entry.col, oldL, oldRt);
		double* Li = L.getRow(entry.row);
		double* Rtj = Rt.getRow(entry.col);
		double* oldLi = oldL.getRow(entry.row);
		double* oldRtj = oldRt.getRow(entry.col);
		for (int k = 0; k < nF; k++)
		{
			deltaL = -2 * temp * oldRtj[k];
			Li[k] -= learning_rate * deltaL;
			
			deltaR = -2 * temp * oldLi[k];
			Rtj[k] -= learning_rate * deltaR;
		}
	}
}

// Compute and print result
void result(const sparse_matrix& A, const matrix& L, const matrix& Rt)
{
	// Struct to hold the mac score and the index of the column it was achieved by
	struct result_pair
	{
		double value;
		int position;
	};
	// Need 2 arrays to store results because MPI send and receive buffers need to be different
	std::unique_ptr<result_pair[]> local_results(new result_pair[A.n_rows]);
	std::unique_ptr<result_pair[]> global_results(new result_pair[A.n_rows]);
	
	auto it = A.elements.begin();
	auto it_end = A.elements.end();
	
	for(int i = 0; i < A.n_rows; ++i) {
		auto& r = local_results[i];
		r.value = -1;
		r.position = -1;
		
		bool finished_row = (it == it_end || it->row > i);
		
		for(int j = 0; j < A.n_columns; ++j) {
			if (!finished_row && j == it->col) {
				it++;
				finished_row = (it == it_end || it->row > i);
				continue;
			}
			
			double b = B(i, j, L, Rt);
			if (b > r.value) {
				r.value = b;
				r.position = proc_column_offsets[cart_coords[1]] + j;
			}
		}
	}

	// Reduce accross grid row
	MPI_Reduce(local_results.get(), global_results.get(), A.n_rows, MPI_DOUBLE_INT, MPI_MAXLOC, 0, Rows_Comm);

	// The root receives results and prints them
	if (id == root_id) {
		for (int i = 0; i < A.n_rows; ++i) std::cout << global_results[i].position << '\n';

		int sender_id;
		MPI_Status status;
		for (int coords[] = {1, 0}; coords[0] < dimensions[0]; ++coords[0]) {
			MPI_Cart_rank(Cart_Comm, coords, &sender_id);
			MPI_Recv(global_results.get(), proc_rows[coords[0]], MPI_DOUBLE_INT, sender_id, DataTag, Cart_Comm, &status);

			for (int i = 0; i < proc_rows[coords[0]]; ++i) std::cout << global_results[i].position << '\n';
		}
	}
	// The tasks with grid column coordinate equal to 0 send their results to root
	else if (cart_coords[1] == 0) {
		MPI_Send(global_results.get(), A.n_rows, MPI_DOUBLE_INT, root_id, DataTag, Cart_Comm);
	}

}

int main(int argc, char* argv[])
{
	std::srand(0);
	sparse_matrix A;
	double elapsed_time;
	double comm_time = 0.;
	double timer_tmp;
	
	MPI_Init(&argc, &argv);
	elapsed_time = -MPI_Wtime();
	
	parse_file_initialize_info(argv[1], A);

	matrix L(proc_rows[cart_coords[0]], nF);
	matrix oldL(proc_rows[cart_coords[0]], nF);
	matrix bufferL(proc_rows[cart_coords[0]], nF);
	
	matrix Rt(proc_columns[cart_coords[1]], nF);
	matrix oldRt(proc_columns[cart_coords[1]], nF);
	matrix bufferRt(proc_columns[cart_coords[1]], nF);

	std::unique_ptr<double[]> buffer(new double[proc_columns[0]]);
	
	initialize_LR(L, bufferL, Rt, buffer.get());

	for (int i = 0; i < n_iter; i++)
	{
		update_LR(A, L, Rt, oldL, oldRt);

		timer_tmp = -MPI_Wtime();
		reduce_LR(L, bufferL, Rt, bufferRt);
		timer_tmp += MPI_Wtime();
		comm_time += timer_tmp;
		
		// After accumulating updates, compute L and Rt for next iteration
		for (int i = 0; i < L.n_rows; ++i) {
			double* Li = L.getRow(i);
			double* oldLi = oldL.getRow(i);
			double* bufferLi = bufferL.getRow(i);
			for (int j = 0; j < L.n_columns; ++j) Li[j] = oldLi[j] + bufferLi[j];
		}
	
		for (int j = 0; j < Rt.n_rows; ++j) {
			double* Rtj = Rt.getRow(j);
			double* oldRtj = oldRt.getRow(j);
			double* bufferRtj = bufferRt.getRow(j);
			for (int i = 0; i < Rt.n_columns; ++i) Rtj[i] = oldRtj[i] + bufferRtj[i];
		}
	}

	result(A, L, Rt);
	elapsed_time += MPI_Wtime();
	
	MPI_Barrier(Cart_Comm);

// Root prints info about runtime
/*	
	if (id == root_id) {
		std::cout << "\n-----------------------------\n";
		std::cout << "Dimensions: " << dimensions[0] << ' ' << dimensions[1] << '\n';
		std::cout << "Elapsed time: " << elapsed_time << '\n';
		std::cout << "Communication: " << comm_time << std::endl;
	}
*/

	MPI_Finalize();

	return 0;
}