cuddhabrot: cuddhabrot.cu
	nvcc -std=c++11 -arch sm_21 -O3 cuddhabrot.cu -o cuddhabrot

clean:
	rm -rf ./cuddhabrot

