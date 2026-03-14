CUDA_DIR := /usr/local/cuda-12.5
NVCC := nvcc
CCFLAGS := -arch=native -isystem $(CUDA_DIR)/include -MMD -res-usage -Xcompiler -Wall -std=c++20
LDFLAGS := -L$(CUDA_DIR)/lib64
BIN_DIR := bin

LIB_SRCS := utils.cu data_load.cu network.cu
EXE_SRCS := data_load_test.cu network_test.cu
SRCS := $(LIB_SRCS) $(EXE_SRCS)

LIB_OBJS := $(addprefix $(BIN_DIR)/, $(LIB_SRCS:.cu=.o))
EXE_OBJS := $(addprefix $(BIN_DIR)/, $(EXE_SRCS:.cu=.o))
DEPS := $(addprefix $(BIN_DIR)/, $(SRCS:.cu=.d))
EXES := $(addprefix $(BIN_DIR)/, $(EXE_SRCS:.cu=))

.PHONY : all clean

all : $(EXES)

$(EXES) : % : %.o $(LIB_OBJS)
	$(NVCC) $^ -o $@ $(LDFLAGS)

$(BIN_DIR)/%.o : %.cu | $(BIN_DIR)
	$(NVCC) $(CCFLAGS) -c $< -o $@

$(BIN_DIR) :
	mkdir -p $@

-include $(DEPS)

clean :
	rm -rf $(BIN_DIR)
