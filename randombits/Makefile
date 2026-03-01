CUDA_DIR := /usr/local/cuda-12.5
NVCC := nvcc -arch=native -I$(CUDA_DIR)/include -L$(CUDA_DIR)/lib64 \
	-Xcompiler -Wall -res-usage
TARGETS := devicecaps test vector_mul distances conv1d

.PHONY : all clean

all : $(TARGETS)

$(TARGETS) : % : %.cu
	$(NVCC) $^ -o $@

clean :
	rm -rf $(TARGETS)
