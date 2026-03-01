CUDA_DIR := /usr/local/cuda-12.5
NVCC := nvcc
CCFLAGS := -arch=native -isystem $(CUDA_DIR)/include -MMD -res-usage -Xcompiler -Wall
LDFLAGS := -L$(CUDA_DIR)/lib64 -lGL -lglut
BIN_DIR := bin
SRCS := board_manager.cu render.cu
OBJS := $(addprefix $(BIN_DIR)/, $(SRCS:.cu=.o))
DEPS := $(addprefix $(BIN_DIR)/, $(SRCS:.cu=.d))
EXE := $(BIN_DIR)/life4

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
