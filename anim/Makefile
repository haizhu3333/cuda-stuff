CUDA_DIR := /usr/local/cuda-12.5
NVCC := nvcc -arch=native -I$(CUDA_DIR)/include -L$(CUDA_DIR)/lib64 \
	-Xcompiler -Wall -res-usage

.PHONY : clean

anim : board_manager.o render.o
	$(NVCC) $(LDFLAGS) $^ -o $@ -lGL -lglut

%.o : %.cu *.h
	$(NVCC) $(CCFLAGS) -c $< -o $@

clean :
	rm -f anim *.o
