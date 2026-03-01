CC := gcc
CUDA_DIR := /usr/local/cuda-12.5
CCFLAGS := -std=c17 -Wall -I$(CUDA_DIR)/include
BIN_DIR := bin

.PHONY : clean

$(BIN_DIR)/launcher : main.c $(BIN_DIR)/klib.a | $(BIN_DIR)
	$(CC) $(CCFLAGS) -L$(CUDA_DIR)/lib64/stubs $< -o $@ -lcuda

$(BIN_DIR)/klib.a : *.cu | $(BIN_DIR)
	nvcc -arch=native -rdc=true -cubin -dlink $^ -o $@

$(BIN_DIR) :
	mkdir -p $@

clean :
	rm -rf $(BIN_DIR)
