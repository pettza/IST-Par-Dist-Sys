#include <stdlib.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <mpi.h>


constexpr int GlobalsTag = 0;
constexpr int SizeTag = 0;
constexpr int DataTag = 0;


// globals
struct {
int n_iter;
int nF;
int nU;
int nI;
double learning_rate;
} globals;

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


struct sparse_data
{
	int row, col;
	double elem;
};

// Creates a matrix from a file
sparse_matrix parse_file_send_data(const char* filename, int nP)
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
	
	int rows_per_proc = globals.nU / nP;
	int extra_rows = globals.nU % nP;
	
	int row = 0, col = 0;
    double elem;
    std::vector<sparse_data> sendingData;
	int n_rows = 0;
	int rows_to_send = rows_per_proc + (extra_rows == 0 ? 0 : 1);
	int id = 0;
    sparse_matrix A;
    for (int i = 0; i < n_elems; i++)
    {
		int prev_row = row;
		
        file >> row >> col >> elem;
		
		std::cout << "Row: " << row << ", Column: " << col << ", Element: " << elem << std::endl;
		
		if (prev_row != row)
		{
			std::cout << "New Row!" << std::endl;
			n_rows += row - prev_row;
			
			if (n_rows >= rows_to_send)
			{	
				if (id != 0)
				{
					int size[] = {(int) sendingData.size(), rows_to_send};
					
					std::cout << "Sending data to processor " << id << std::endl;
					
					MPI_Send(size, 2, MPI_INT, id, SizeTag, MPI_COMM_WORLD);
					if (size[0] != 0) MPI_Send(sendingData.data(), size[0] * sizeof(sparse_data), MPI_BYTE, id, DataTag, MPI_COMM_WORLD);
				}
				
				sendingData.clear();
				n_rows -= rows_to_send ;
				id++;
				rows_to_send = rows_per_proc + (id > extra_rows ? 0 : 1);
			}
        }
		
        sendingData.push_back({row, col, elem});
		A.add_element(row, col, elem);
    }
    
	//Send to last processors
	int size[] = {(int) sendingData.size(), rows_to_send};
					
	std::cout << "Sending data to processor " << id << std::endl;
	
	MPI_Send(size, 2, MPI_INT, id, SizeTag, MPI_COMM_WORLD);
	if (size[0] != 0) MPI_Send(sendingData.data(), size[0] * sizeof(sparse_data), MPI_BYTE, id, DataTag, MPI_COMM_WORLD);
	
	id++;
	size[0] = 0;
	size[1] = 0;
	for (; id < nP; id++) MPI_Send(size, 2, MPI_INT, id, SizeTag, MPI_COMM_WORLD);

    file.close();
    
    return A;
}


int main(int argc, char* argv[])
{
	int id, p;
	sparse_matrix A;
	
	MPI_Init(&argc, &argv);
	
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
	
	if (id == 0) A = parse_file_send_data(argv[1], p);
	else
	{
		MPI_Status status;
		int size[2];
		MPI_Recv(size, 2, MPI_INT, 0, SizeTag, MPI_COMM_WORLD, &status);
		
		sparse_data* buffer;
		
		if (size[0] != 0)
		{
			buffer = new sparse_data[size[0]];
			
			MPI_Recv(buffer, size[0] * sizeof(sparse_data), MPI_BYTE, 0, DataTag, MPI_COMM_WORLD, &status);
			
			int offset = buffer[0].row;
			
			for (int i = 0; i < size[0]; ++i)
			{
				A.add_element(buffer[i].row - offset, buffer[i].col, buffer[i].elem);
			}
			
			std::cout << "Processor " << id << " received data" << std::endl;
			
			delete[] buffer;
		}
	}
	
	//Send globals
	MPI_Bcast(&globals, sizeof(globals), MPI_BYTE, 0, MPI_COMM_WORLD);
	
	MPI_Finalize();
	
	return 0;
}