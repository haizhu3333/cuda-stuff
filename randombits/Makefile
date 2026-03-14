CUDA_DIR := /usr/local/cuda-12.5
NVCC := nvcc -arch=native -isystem $(CUDA_DIR)/include -L$(CUDA_DIR)/lib64 \
	-MMD -Xcompiler -Wall -res-usage
BIN_DIR := bin
EXES := devicecaps test vector_mul distances conv1d typecast
TARGETS := $(addprefix $(BIN_DIR)/, $(EXES))
SRCS := $(addsuffix .cu, $(EXES))
DEPS := $(addsuffix .d, $(TARGETS))

.PHONY : all clean

all : $(TARGETS)

$(BIN_DIR)/% : %.cu | $(BIN_DIR)
	$(NVCC) $< -o $@

$(BIN_DIR) :
	mkdir -p $(BIN_DIR)

-include $(DEPS)

clean :
	rm -rf $(TARGETS)
