
CFLAGS = -O3 -Wall -g
CXXFLAGS = -I../.. -O3 -Wall -g
LDFLAGS = -lm

OPENMP_CFLAGS = -fopenmp
OPENMP_LDFLAGS = -fopenmp

# MPI
MPICC = mpicc
MPICXX = mpicxx
MPI_INCLUDE = $(shell mpicc -show | sed 's/.*-I\([\/a-zA-Z0-9_\-]*\).*/\1/g')

# CUDA
NVCC = nvcc
NVCC_CFLAGS = -I../.. -O3 -Xcompiler -Wall -Xptxas -v -arch sm_20 # -keep
CUDA_INC = $(patsubst %bin/nvcc,%include, $(shell which $(NVCC)))
ifeq (,$(findstring Darwin,$(shell uname)))
	CUDA_LDFLAGS = -lcudart -L$(patsubst %bin/nvcc,%lib64, \
		$(shell which $(NVCC)))
else
	NVCC_CFLAGS += -m64
	CUDA_LDFLAGS = -lcudart -L$(patsubst %bin/nvcc,%lib, \
		$(shell which $(NVCC)))
endif

# Physis
PHYSISC_CONFIG ?= opt.conf
PHYSISC_CONFIG_KEY = $(shell basename $(PHYSISC_CONFIG))
PHYSISC_REF = @CMAKE_INSTALL_PREFIX@/bin/physisc-ref --config $(realpath $(PHYSISC_CONFIG))
PHYSISC_CUDA = @CMAKE_INSTALL_PREFIX@/bin/physisc-cuda --config $(realpath $(PHYSISC_CONFIG))
PHYSISC_MPI = @CMAKE_INSTALL_PREFIX@/bin/physisc-mpi --config $(realpath $(PHYSISC_CONFIG))
PHYSISC_MPI_CUDA = @CMAKE_INSTALL_PREFIX@/bin/physisc-mpi-cuda --config $(realpath $(PHYSISC_CONFIG))
PHYSIS_BUILD_DIR_TOP = physis_build
PHYSIS_BUILD_DIR = physis_build/$(PHYSISC_CONFIG_KEY)
ifeq ("@AUTO_TUNING@", "TRUE")
PHYSIS_LD_FLAGS_AT = -Xcompiler -rdynamic -ldl
else
PHYSIS_LD_FLAGS_AT = 
endif

# Minimal configuration
SRC = diffusion3d.cc baseline.cc
OBJ = $(filter %.o,$(SRC:%.cc=%.o)) $(filter %.o,$(SRC:%.c=%.o))

EXE = diffusion3d_baseline.exe \
	diffusion3d_openmp.exe diffusion3d_openmp_temporal_blocking.exe \
	diffusion3d_cuda.exe diffusion3d_cuda_opt1.exe \
	diffusion3d_cuda_opt2.exe \
	diffusion3d_cuda_shared.exe \
	diffusion3d_cuda_temporal_blocking.exe \
	diffusion3d_mic.exe

.SUFFIXES: .cu

.cu.o:
	$(NVCC) -c $< $(NVCC_CFLAGS) -o $@
##################################################

#all: diffusion3d_baseline.exe diffusion3d_openmp.exe diffusion3d_openmp_temporal_blocking.exe
all: physis-cuda

baseline: diffusion3d_baseline.exe
diffusion3d_baseline.exe: $(OBJ) main_baseline.o
	$(CXX) -o $@ $^ $(LDFLAGS)

main_baseline.o: main.cc
	$(CXX) -o $@ -c $^ $(CXXFLAGS)

openmp: diffusion3d_openmp.exe

diffusion3d_openmp.o: CXXFLAGS += $(OPENMP_CFLAGS)
diffusion3d_openmp.exe: $(OBJ) main_openmp.o diffusion3d_openmp.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(OPENMP_LDFLAGS)

main_openmp.o: main.cc
	$(CXX) -o $@ -c $^ $(CXXFLAGS) -DOPENMP

openmp_temporal_blocking: diffusion3d_openmp_temporal_blocking.exe
diffusion3d_openmp_temporal_blocking.exe: $(OBJ) main_openmp_temporal_blocking.o \
	diffusion3d_openmp_temporal_blocking.o diffusion3d_openmp.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(OPENMP_LDFLAGS)

main_openmp_temporal_blocking.o: main.cc
	$(CXX) -o $@ -c $^ $(CXXFLAGS) -DOPENMP_TEMPORAL_BLOCKING

diffusion3d_openmp_temporal_blocking.o: CXXFLAGS += $(OPENMP_CFLAGS)

cuda: diffusion3d_cuda.exe diffusion3d_cuda_opt1.exe \
	diffusion3d_cuda_opt2.exe diffusion3d_cuda_shared.exe \
	diffusion3d_cuda_xy.exe diffusion3d_cuda_temporal_blocking.exe
diffusion3d_cuda.exe: $(OBJ) main_cuda.o diffusion3d_cuda.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
diffusion3d_cuda.o: diffusion3d_cuda.h
main_cuda.o: main.cc diffusion3d_cuda.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA -I$(CUDA_INC)
diffusion3d_cuda_opt1.exe: $(OBJ) main_cuda_opt1.o diffusion3d_cuda.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
main_cuda_opt1.o: main.cc diffusion3d_cuda.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA_OPT1 -I$(CUDA_INC)
diffusion3d_cuda_opt2.exe: $(OBJ) main_cuda_opt2.o diffusion3d_cuda.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
main_cuda_opt2.o: main.cc diffusion3d_cuda.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA_OPT2 -I$(CUDA_INC)
diffusion3d_cuda_shared.exe: $(OBJ) main_cuda_shared.o diffusion3d_cuda.o \
	diffusion3d_cuda_shared.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
main_cuda_shared.o: main.cc diffusion3d_cuda.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA_SHARED -I$(CUDA_INC)
diffusion3d_cuda_shared.o: diffusion3d_cuda.h
# DISABLE L1 load caching
diffusion3d_cuda_shared.o: NVCC_CFLAGS += -Xptxas -dlcm=cg
# XY multi processing
diffusion3d_cuda_xy.exe: $(OBJ) main_cuda_xy.o diffusion3d_cuda.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
main_cuda_xy.o: main.cc
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA_XY -I$(CUDA_INC)
# Temporal blokcing
diffusion3d_cuda_temporal_blocking.o: diffusion3d_cuda.h diffusion3d_cuda_temporal_blocking.h
diffusion3d_cuda_temporal_blocking.exe: $(OBJ) main_cuda_temporal_blocking.o diffusion3d_cuda.o diffusion3d_cuda_temporal_blocking.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
main_cuda_temporal_blocking.o: main.cc diffusion3d_cuda_temporal_blocking.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA_TEMPORAL_BLOCKING -I$(CUDA_INC)


.PHONY: physis
physis: physis-ref physis-cuda physis-mpi physis-mpi-cuda

.PHONY: physis-ref
physis-ref: $(PHYSIS_BUILD_DIR) $(PHYSIS_BUILD_DIR)/diffusion3d_physis.ref.exe

.PHONY: physis-cuda
physis-cuda: $(PHYSIS_BUILD_DIR) $(PHYSIS_BUILD_DIR)/diffusion3d_physis.cuda.exe
.PHONY: physis-cuda_at
physis-cuda_at: physis-cuda
	$(MAKE) physis-cuda_at_dynamiclinklibraries
CUDA_DL:= $(patsubst %.cu,%.so,$(wildcard $(PHYSIS_BUILD_DIR)/diffusion3d_physis.*.cuda_dl.cu))
.PHONY: physis-cuda_at_dynamiclinklibraries
physis-cuda_at_dynamiclinklibraries:
	[ "" = "$(CUDA_DL)" ] || $(MAKE) $(CUDA_DL)
%.cuda_dl.o: %.cuda_dl.cu 
	$(NVCC) -c $< $(NVCC_CFLAGS) -I@CMAKE_INSTALL_PREFIX@/include -o $@ -Xcompiler -fPIC
%.cuda_dl.so: %.cuda_dl.o
	$(CXX) -o $@ $^ -shared $(LDFLAGS) $(CUDA_LDFLAGS) @CUDA_CUT_LIBRARIES@

.PHONY: physis-mpi
physis-mpi: $(PHYSIS_BUILD_DIR) $(PHYSIS_BUILD_DIR)/diffusion3d_physis.mpi.exe

.PHONY: physis-mpi-cuda
physis-mpi-cuda: $(PHYSIS_BUILD_DIR) $(PHYSIS_BUILD_DIR)/diffusion3d_physis.mpi-cuda.exe

$(PHYSIS_BUILD_DIR):
	mkdir -p $(PHYSIS_BUILD_DIR)

main_physis.o: main.cc
	$(CXX) -o $@ -c $^ $(CXXFLAGS) -DPHYSIS
# reference
$(PHYSIS_BUILD_DIR)/diffusion3d_physis.ref.c: diffusion3d_physis.c $(PHYSISC_CONFIG)
	cd $(PHYSIS_BUILD_DIR) && $(PHYSISC_REF) ../../$<
$(PHYSIS_BUILD_DIR)/diffusion3d_physis.ref.o: CFLAGS += -I@CMAKE_INSTALL_PREFIX@/include
$(PHYSIS_BUILD_DIR)/diffusion3d_physis.ref.exe: $(PHYSIS_BUILD_DIR)/diffusion3d_physis.ref.o \
	main_physis.o baseline.o diffusion3d.o @CMAKE_INSTALL_PREFIX@/lib/libphysis_rt_ref.a
	$(CXX) -o $@ $^ $(LDFLAGS)
# cuda
$(PHYSIS_BUILD_DIR)/diffusion3d_physis.cuda.cu: diffusion3d_physis.c $(PHYSISC_CONFIG)
	cd $(PHYSIS_BUILD_DIR) && $(PHYSISC_CUDA) ../../$<
$(PHYSIS_BUILD_DIR)/diffusion3d_physis.cuda.o: NVCC_CFLAGS += -I@CMAKE_INSTALL_PREFIX@/include
$(PHYSIS_BUILD_DIR)/diffusion3d_physis.cuda.exe: $(PHYSIS_BUILD_DIR)/diffusion3d_physis.cuda.o \
	main_physis.o baseline.o diffusion3d.o @CMAKE_INSTALL_PREFIX@/lib/libphysis_rt_cuda.a
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS) $(PHYSIS_LD_FLAGS_AT)

# mpi
$(PHYSIS_BUILD_DIR)/diffusion3d_physis.mpi.c: diffusion3d_physis.c $(PHYSISC_CONFIG)
	cd $(PHYSIS_BUILD_DIR) && $(PHYSISC_MPI) ../../$<
$(PHYSIS_BUILD_DIR)/diffusion3d_physis.mpi.o: CFLAGS += -I@CMAKE_INSTALL_PREFIX@/include -I$(MPI_INCLUDE)
$(PHYSIS_BUILD_DIR)/diffusion3d_physis.mpi.exe: $(PHYSIS_BUILD_DIR)/diffusion3d_physis.mpi.o \
	main_physis.o baseline.o diffusion3d.o @CMAKE_INSTALL_PREFIX@/lib/libphysis_rt_mpi.a
	$(MPICXX) -o $@ $^ $(LDFLAGS)

# mpi-cuda
$(PHYSIS_BUILD_DIR)/diffusion3d_physis.mpi-cuda.cu: diffusion3d_physis.c $(PHYSISC_CONFIG)
	cd $(PHYSIS_BUILD_DIR) && $(PHYSISC_MPI_CUDA) ../../$<
$(PHYSIS_BUILD_DIR)/diffusion3d_physis.mpi-cuda.o: NVCC_CFLAGS += -I@CMAKE_INSTALL_PREFIX@/include \
	-arch sm_20 -I$(MPI_INCLUDE)
$(PHYSIS_BUILD_DIR)/diffusion3d_physis.mpi-cuda.exe: $(PHYSIS_BUILD_DIR)/diffusion3d_physis.mpi-cuda.o \
	main_physis.o baseline.o diffusion3d.o @CMAKE_INSTALL_PREFIX@/lib/libphysis_rt_mpi_cuda.a
	$(MPICXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)

clean:
	-$(RM) *.o $(EXE)
	-$(RM) diffusion3d_result.*.out
	-$(RM) *.cudafe* *.gpu *.stub.c *.pptx *.cubin *.i *.ii *.fatbin *.fatbin.c
	-$(RM) *.exe
	-$(RM) *_physis.ref.* *_physis.cuda.* *_physis.mpi.* \
		*_physis.mpi-cuda.*
	-$(RM) -rf $(PHYSIS_BUILD_DIR_TOP)
