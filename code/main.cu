#include <stdio.h>
#include <cassert>
#include <vector>
#include <functional>
#include <string>
#include <fstream>
#include <iostream>
#include <random>
#include "helper_cuda.hpp"

bool is_unified = false;
bool fix_thread = true;
int block=512, thread=16;
int N=1000;
size_t size;
float t;
std::string output_name = "result.csv";

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
void vector_add(const std::vector<T> &a, const std::vector<T> &b, std::vector<T> &out, float &running_time) {
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
	
	running_time = t.get_time();
}

template<typename T>
float run_vec_add(int n) {
	std::vector<T> a(n),b(n),out(n);
	
	std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<T> dist(1.0, 1e7);

	for(int i = 0; i < n; ++i) {
		a[i] = dist(mt);
		b[i] = dist(mt);
	}

	float t;
	vector_add<T>(a,b,out, t);
	return t;
}

void output_settings(const std::string fname) {
	std::fstream output(fname, std::fstream::out | std::fstream::app);
	if(output.is_open()) {
		output << block << "," << thread << "," << N << "," << t << "," << size << "," << is_unified << "," << fix_thread << std::endl;
	} else {
		std::cerr << "Cannot open file " << fname << std::endl;
	}

	output.close();
}

void exp() {
	int min = 128, max = std::pow(2, 28), iter = 20;
	for(int i = 0; i < iter; ++i) {
		N = min + (max-min)/(iter-1) * i;	
		block = 512;
		thread = fix_thread ? 32 : (N + block -1)/block;
		t = run_vec_add<float>(N);
		size = N * sizeof(float)/1024;
		printf("size: %d MB \n", (int)size);
		output_settings(output_name);
	}

}

int main(int argc, char* argv[]) {
	GC(cudaSetDevice(0));

	// initialize result dataframe header
	std::fstream output(output_name, std::fstream::out);
	if(output.is_open()) {
		output << "block" << "," << "thread" << "," << "N" << "," << "time" << "," << "size" << "," << "unified" << "," << "fix_thread"<< std::endl;
	} else {
		std::cerr << "Cannot open file " << output_name << std::endl;
	}
	output.close();
	
	fix_thread = true;
	is_unified = false;
	exp();

	fix_thread = false;
	is_unified = false;
	exp();

	fix_thread = true;
	is_unified = true;
	exp();
	
	fix_thread = false;
	is_unified = true;
	exp();
	return 0;
}