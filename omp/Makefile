.PHONY: distclean default kinst opt

CXX=g++
CXXFLAGS=-Wall -std=c++11 -fopenmp

default: matFact-omp

opt: CXXFLAGS += -O2
opt: matFact-omp

kinst: CXX := kinst-ompp $(CXX)
kinst: matFact-omp

matFact-omp: matFact-omp.cpp
	$(CXX) $(CXXFLAGS) -o matFact-omp matFact-omp.cpp

distclean:
	$(RM) matFact-omp