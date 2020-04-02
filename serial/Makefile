.PHONY: distclean default opt

CXX=g++
CXXFLAGS=-Wall -std=c++11

default: matFact

opt: CXXFLAGS += -O2
opt: matFact

matFact: matFact.cpp
	$(CXX) $(CXXFLAGS) -o matFact matFact.cpp

distclean:
	$(RM) matFact