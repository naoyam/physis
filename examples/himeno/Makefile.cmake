PHYSISC_CONFIG ?= opt.conf
PHYSISC = @CMAKE_INSTALL_PREFIX@/bin/physisc -DENABLE_DUMP --config $(PHYSISC_CONFIG)
PHYSIS_INCLUDE = -I@CMAKE_INSTALL_PREFIX@/include
CFLAGS = -Wall -g $(PHYSIS_INCLUDE) -O2 -DENABLE_DUMP
NVCC_CFLAGS = -g -Xcompiler -Wall $(PHYSIS_INCLUDE) -arch sm_20 -DENABLE_DUMP --ptxas-options -v
MPI_INCLUDE = $(shell for mpiinc in $(shell echo "@MPI_INCLUDE_PATH@" | sed 's/;/ /g'); do echo -n "-I$$mpiinc "; done)
NVCC_LDFLAGS= -lcudart -L@CUDA_RT_DIR@
ifeq ("@AUTO_TUNING@", "TRUE")
NVCC_LD_FLAGS_AT = -Xcompiler -rdynamic -ldl
else
NVCC_LD_FLAGS_AT =
endif

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
	nvcc $^ -o $@ $(LDFLAGS) $(NVCC_LD_FLAGS_AT) @CMAKE_INSTALL_PREFIX@/lib/libphysis_rt_cuda.a

cuda_at: cuda
	$(MAKE) cuda_at_dynamiclinklibraries
CUDA_DL:= $(patsubst %.cu,%.so,$(wildcard himenobmtxpa_physis.*.cuda_dl.cu))
.PHONY: cuda_at_dynamiclinklibraries
cuda_at_dynamiclinklibraries:
	[ "" = "$(CUDA_DL)" ] || $(MAKE) $(CUDA_DL)
%.cuda_dl.o: %.cuda_dl.cu
	nvcc -c $^ $(NVCC_CFLAGS) -Xcompiler -fPIC
%.cuda_dl.so: %.cuda_dl.o
	nvcc $^ -o $@ -shared $(LDFLAGS) @CMAKE_INSTALL_PREFIX@/lib/libphysis_rt_cuda.a

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
	mpicxx $^ -o $@ $(LDFLAGS) \
	@CMAKE_INSTALL_PREFIX@/lib/libphysis_rt_mpi_cuda.a \
	 $(NVCC_LDFLAGS)

clean:
	-rm -f *.exe *.o *~ himenobmtxpa_physis.*.*
