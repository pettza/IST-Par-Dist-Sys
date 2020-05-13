.PHONY: distclean default omp

CXX=mpic++
CXXFLAGS=-Wall -std=c++11 -g

default: matFact-mpi

matFact-mpi: matFact-mpi.cpp
	$(CXX) $(CXXFLAGS) -o matFact-mpi matFact-mpi.cpp

distclean:
	$(RM) matFact-mpi