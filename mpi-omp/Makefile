.PHONY: distclean default omp

CXX=mpic++
CXXFLAGS=-Wall -std=c++11 -g -fopenmp

default: matFact-mpi-omp

matFact-mpi-omp: matFact-mpi-omp.cpp
	$(CXX) $(CXXFLAGS) -o matFact-mpi-omp matFact-mpi-omp.cpp

distclean:
	$(RM) matFact-mpi-omp