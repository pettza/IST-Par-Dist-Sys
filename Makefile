.PHONY: distclean serial omp mpi mpi-omp default

default: serial omp mpi mpi-omp

serial:
	$(MAKE) -C serial
	cp serial/matFact ./

omp:
	$(MAKE) -C omp
	cp omp/matFact-omp ./

mpi:
	$(MAKE) -C mpi
	cp mpi/matFact-mpi ./

mpi-omp:
	$(MAKE) -C mpi-omp
	cp mpi-omp/matFact-mpi-omp ./

distclean:
	$(RM) matFact*