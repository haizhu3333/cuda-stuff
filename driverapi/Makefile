CC := gcc
CUDA_DIR := /usr/local/cuda-12.5
INCLUDE := $(CUDA_DIR)/include
CCFLAGS := -std=c17 -Wall -I$(CUDA_DIR)/include

.PHONY : clean

launcher : main.c klib.a
	$(CC) $(CCFLAGS) -L$(CUDA_DIR)/lib64/stubs $< -o $@ -lcuda

klib.a : *.cu
	nvcc -arch=native -rdc=true -cubin -dlink $^ -o $@

clean :
	rm -f klib.a launcher
