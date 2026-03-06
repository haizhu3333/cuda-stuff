CUDA_DIR := /usr/local/cuda-12.5
NVCC := nvcc
CCFLAGS := -arch=native -isystem $(CUDA_DIR)/include -MMD -res-usage -Xcompiler -Wall -std=c++20
LDFLAGS := -L$(CUDA_DIR)/lib64
BIN_DIR := bin
SRCS := utils.cu data_load.cu main.cu
OBJS := $(addprefix $(BIN_DIR)/, $(SRCS:.cu=.o))
DEPS := $(addprefix $(BIN_DIR)/, $(SRCS:.cu=.d))
EXE := $(BIN_DIR)/main

.PHONY : clean

$(EXE) : $(OBJS)
	$(NVCC) $^ -o $@ $(LDFLAGS)

$(BIN_DIR)/%.o : %.cu | $(BIN_DIR)
	$(NVCC) $(CCFLAGS) -c $< -o $@

$(BIN_DIR) :
	mkdir -p $@

-include $(DEPS)

clean :
	rm -rf $(BIN_DIR)
