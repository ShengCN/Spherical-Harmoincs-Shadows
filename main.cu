#include <stdio.h>
#include <cassert>
#include <vector>
#include <functional>
#include "helper_cuda.hpp"

bool is_unified = false;

template <typename T>
__global__
void cuda_add(T *a, T *b, int n, T *out) {
	int ind = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for(int i = ind; i < n; i += stride) {
		out[i] = a[i] + b[i];
	}
}

template <typename T>
void vector_add(const std::vector<T> &a, const std::vector<T> &b, std::vector<T> &out) {
	assert (a.size() == b.size());
	
	int n = a.size();
	out = b;
	if(n == 0) {
		return;
	}

	// cuda memory allocate
	T *d_a, *d_b, *d_out;
	size_t ary_size = n * sizeof(T);
	if(!is_unified) {
		GC(cudaMalloc(&d_a, ary_size));
		GC(cudaMalloc(&d_b, ary_size));
		GC(cudaMalloc(&d_out, ary_size));
	} else {
		GC(cudaMallocManaged(&d_a, ary_size));
		GC(cudaMallocManaged(&d_b, ary_size));
		GC(cudaMallocManaged(&d_out, ary_size));
	}
	
	GC(cudaMemcpy(d_a, &a[0], ary_size, cudaMemcpyHostToDevice));
	GC(cudaMemcpy(d_b, &b[0], ary_size, cudaMemcpyHostToDevice));

	int block=512, thread=32;

	timer t;
	t.tic();
	cuda_add<T><<<block, thread>>>(d_a, d_b, n, d_out);
	GC(cudaDeviceSynchronize());
	t.toc();
	printf("total time: %f \n", t.get_time());

	GC(cudaMemcpy(&out[0], d_out, ary_size, cudaMemcpyDeviceToHost));	
	
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_out);
}

int main(int argc, char* argv[]) {
	int n = 100000000;
	std::vector<float> a(n),b(n),out(n);
	
	for(int i = 0; i < n; ++i) {
		a[i] = i;
		b[i] = n-i;
	}
	vector_add<float>(a,b,out);

	return 0;
}