program: main.o pcm_utils.o seq_cfd_code.o finite_diff_functions.o
	mpiicpc -qopenmp -mt_mpi main.o pcm_utils.o seq_cfd_code.o finite_diff_functions.o ./pcm/cpucounters.o ./pcm/msr.o ./pcm/pci.o ./pcm/client_bw.o -o main_prog
main.o: main.cpp
	mpiicpc -O3 -mt_mpi -c main.cpp -qopenmp -o main.o
seq_cfd_code.o: seq_cfd_code.cpp
	icpc -O1 -c seq_cfd_code.cpp
pcm_utils.o: pcm_utils.cpp
	icpc -O3 -c pcm_utils.cpp
finite_diff_functions.o: finite_diff_functions.cpp
	icpc -O3 -c finite_diff_functions.cpp -qopenmp -o finite_diff_functions.o
