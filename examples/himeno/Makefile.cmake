PHYSISC = @CMAKE_INSTALL_PREFIX@/bin/physisc -DENABLE_DUMP
PHYSIS_INCLUDE = -I@CMAKE_INSTALL_PREFIX@/include
CFLAGS = -Wall -g $(PHYSIS_INCLUDE) -O2 -DENABLE_DUMP
NVCC_CFLAGS = -g -Xcompiler -Wall $(PHYSIS_INCLUDE) -arch sm_20 -DENABLE_DUMP
MPI_INCLUDE = $(shell for mpiinc in $(shell echo "@MPI_INCLUDE_PATH@" | sed 's/;/ /g'); do echo -n "-I$$mpiinc "; done)
NVCC_LDFLAGS= -lcudart -L@CUDA_RT_DIR@

all:  ref cuda mpi mpi-cuda original

# Original
original: himenobmtxpa_original.exe

himenobmtxpa_original.exe: himenobmtxpa_original.o
	cc $^ -o $@

# Reference target
ref: himenobmtxpa_physis.ref.exe

himenobmtxpa_physis.ref.c: himenobmtxpa_physis.c
	$(PHYSISC) --ref $^ $(PHYSIS_INCLUDE)

himenobmtxpa_physis.ref.exe: himenobmtxpa_physis.ref.o
	c++ $^ -o $@ $(LDFLAGS) @CMAKE_INSTALL_PREFIX@/lib/libphysis_rt_ref.a

# CUDA target
cuda: himenobmtxpa_physis.cuda.exe

himenobmtxpa_physis.cuda.cu: himenobmtxpa_physis.c
	$(PHYSISC) --cuda $^ $(PHYSIS_INCLUDE)

himenobmtxpa_physis.cuda.o: himenobmtxpa_physis.cuda.cu
	nvcc -c $^ $(NVCC_CFLAGS)

himenobmtxpa_physis.cuda.exe: himenobmtxpa_physis.cuda.o
	nvcc $^ -o $@ $(LDFLAGS) @CMAKE_INSTALL_PREFIX@/lib/libphysis_rt_cuda.a \
		@CUDA_CUT_LIBRARIES@

# MPI target
mpi: himenobmtxpa_physis.mpi.exe

himenobmtxpa_physis.mpi.c: himenobmtxpa_physis.c
	$(PHYSISC) --mpi $^ $(PHYSIS_INCLUDE)

himenobmtxpa_physis.mpi.o: himenobmtxpa_physis.mpi.c
	mpicc -c $^ $(CFLAGS)

himenobmtxpa_physis.mpi.exe: himenobmtxpa_physis.mpi.o
	mpicxx $^ -o $@ $(LDFLAGS) @CMAKE_INSTALL_PREFIX@/lib/libphysis_rt_mpi.a

# MPI-CUDA target
mpi-cuda: himenobmtxpa_physis.mpi-cuda.exe

himenobmtxpa_physis.mpi-cuda.cu: himenobmtxpa_physis.c
	$(PHYSISC) --mpi-cuda $^ $(PHYSIS_INCLUDE)

himenobmtxpa_physis.mpi-cuda.o: himenobmtxpa_physis.mpi-cuda.cu
	nvcc -c $^ $(NVCC_CFLAGS) $(MPI_INCLUDE) 

himenobmtxpa_physis.mpi-cuda.exe: himenobmtxpa_physis.mpi-cuda.o
	mpicxx $^ -o $@ $(LDFLAGS) @CMAKE_INSTALL_PREFIX@/lib/libphysis_rt_mpi_cuda.a \
		@CUDA_CUT_LIBRARIES@ $(NVCC_LDFLAGS)

clean:
	-rm -f *.exe *.o *~ himenobmtxpa_physis.*.*