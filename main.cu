#include <stdio.h>
#include <cassert>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <random>
#include <deque>
#include <cmath>
#include "helper_cuda.hpp"
#include "glm/vec3.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

using namespace glm;
struct image {
	std::vector<unsigned char> pixels;
	int w, h;

	image(int w=0, int h=0) :w(w), h(h) {
		clear();
	}

	void clear() {
		pixels.resize(w * h * 3, 0xff);
	}

	unsigned char* data() {
		return pixels.data();
	}

	bool save(const std::string fname) {
		return stbi_write_png(fname.c_str(), w, h, 3, pixels.data(), w * 3);
	}
	
	bool read(const std::string fname) {
		int channels;
		unsigned char *img_ptr = stbi_load(fname.c_str(), &w, &h, &channels, 0);
		if(img_ptr == nullptr) {
			std::cerr << "image loading failed \n";
			return false;
		}

		pixels.resize(w * h * 3, 0xff);
		for(int u = 0; u < w; ++u) 
			for(int v = 0; v < h; ++v) {
				size_t ind = (h-1-v) * w + u;
				pixels.at(ind * 3 + 0) = img_ptr[ind * 3 + 0];
				pixels.at(ind * 3 + 1) = img_ptr[ind * 3 + 1];
				pixels.at(ind * 3 + 2) = img_ptr[ind * 3 + 2];
			} 
		
		stbi_image_free(img_ptr);
		return true;
	}
};

const char* getFileNameFromPath(const char* path )
{
    if( path == NULL )
        return NULL;

    const char * pFileName = path;
    for( const char * pCur = path; *pCur != '\0'; pCur++)
    {
        if( *pCur == '/' || *pCur == '\\' )
            pFileName = pCur+1;
    }

    return pFileName;
}

enum class algorithm {
	rectangular,
	triangular,
	gaussian
};

__host__ __device__ 
unsigned char clip(float v) {
	if(v<0.0f) v = 0.0f;
	if(v>255.0f) v= 255.0f;
	return (unsigned char)v;
}

// unnormalized gaussian
__host__ __device__ 
float gaussian(float x, float mu, float sig) {
	return exp(-0.5f * (x-mu) * (x-mu)/(sig * sig));
}

__global__ 
void cuda_process(unsigned char **buffers, int n, int w, int h, algorithm alg_type, unsigned char *out_buffer) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;
	int u_stride = blockDim.x * gridDim.x;
	int v_stride = blockDim.y * gridDim.y;
	
	for(int x = u; x < w; x += u_stride) {
		for(int y = v; y < h; y += v_stride) {
			int ind = y * w + x; 
			float r = 0.0f,g = 0.0f,b = 0.0f;
			int ri = ind * 3 + 0, gi = ind * 3 + 1, bi = ind * 3 + 2;
			switch(alg_type) {
				case algorithm::rectangular: {
					float weight = 1.0f/n;
					for(int i = 0; i < n; ++i) {
						r += (float)buffers[i][ri] * weight;
						g += (float)buffers[i][gi] * weight;
						b += (float)buffers[i][bi] * weight;
					}
					out_buffer[ri] = clip(r);
					out_buffer[gi] = clip(g);
					out_buffer[bi] = clip(b);
				}
				break;
				case algorithm::triangular: {
					float dh = 1.0f/n;
					for(int i = 0; i < n; ++i) {
						float d = (float)(i+1);
						if (i + 1 > n/2)
							d = d - n/2;
						float weight = d * dh;
						r += (float)buffers[i][ri] * weight;
						g += (float)buffers[i][gi] * weight;
						b += (float)buffers[i][bi] * weight;
					}
					out_buffer[ri] = clip(r);
					out_buffer[gi] = clip(g);
					out_buffer[bi] = clip(b);
				}
				break;
				case algorithm::gaussian: {
					float sig = .5f;
					float mu = (float)(n/2);
					float w_sum = 0.0f;
					for(int i = 0; i < n; ++i) {
						float weight = gaussian(i, mu, sig);
						w_sum += weight;
						r += (float)buffers[i][ri] * weight;
						g += (float)buffers[i][gi] * weight;
						b += (float)buffers[i][bi] * weight;
					}
					out_buffer[ri] = clip(r/w_sum);
					out_buffer[gi] = clip(g/w_sum);
					out_buffer[bi] = clip(b/w_sum);
				}
				break;
				default:
				break;
			}
		}
	}
}

// sliding window processing
class window_process {
public:
	window_process(const std::vector<std::string> &files) {
		int n = (int)files.size();
		assert(n > 0);
		d_buffers.resize(n);
		
		int w,h;
		// initialize buffers
		for(int i = 0; i < n; ++i) {
			image img;
			img.read(files[i]);
			w = img.w; h = img.h;
			GC(cudaMalloc(&d_buffers[i], sizeof(unsigned char) * w * h * 3));
			to_cuda(img, d_buffers[i]);
		}
		
		// init output buffer
		// assumption: same image size for each image
		h_out_img = image(w, h);
		GC(cudaMalloc(&d_cur_buffer, sizeof(unsigned char) * w * h * 3));	
	}

	~window_process() {
		cudaFree(d_cur_buffer);
		for(auto &p:d_buffers) {
			cudaFree(p);
		}	
	}
	
	bool add_image(const std::string file) {
		image tmp;
		if(tmp.read(file)) {
			auto first_ptr = *d_buffers.begin();
			d_buffers.pop_front();

			to_cuda(tmp, first_ptr);
			d_buffers.push_back(first_ptr);
			return true;
		} else {
			return false;
		}
	}
	
	// compute motion blur for current batch
	void process(algorithm type, image &out) {
		dim3 grid(256,256,1), block(32,32,1);
		
		// image pointer array
		int n = (int)d_buffers.size();
		unsigned char** sequence;
		cudaMallocManaged(&sequence, sizeof(unsigned char *) * n);
		for(int i = 0; i < n; ++i) {
			sequence[i] = d_buffers[i];
		}

		cuda_process<<<grid,block>>>(sequence, n, h_out_img.w, h_out_img.h, type, d_cur_buffer);
		GC(cudaDeviceSynchronize());
		to_local(d_cur_buffer, h_out_img);
		out = h_out_img;

		cudaFree(sequence);
	}

	void to_cuda(image &img, unsigned char *d_ptr) {
		GC(cudaMemcpy(d_ptr, img.data(), sizeof(unsigned char) * img.w * img.h * 3, cudaMemcpyHostToDevice));	
	}

	void to_local(unsigned char *dptr, image &img) {
		GC(cudaMemcpy(img.data(), dptr,  sizeof(unsigned char) * img.w * img.h * 3, cudaMemcpyDeviceToHost));
	}

private:
	std::deque<unsigned char *> d_buffers;
	unsigned char *d_cur_buffer;
	image h_out_img;
	int first_ind;	// first index for sliding window
};

void openmp_woker(std::vector<std::string> &files, const std::string output_folder, const int n, algorithm cur_algorithm) {
	assert((int)files.size() > n);
	assert(n % 2 == 1); 

	// init window
	// n is odd number
	std::vector<std::string> init_files(n/2, files[0]);
	init_files.insert(init_files.end(), files.begin(), files.begin() + n/2 + 1);
	window_process processor(init_files);

	image out_img;
	for(int i = 0; i < files.size(); ++i) {
		processor.process(cur_algorithm, out_img);
		std::string ori_fname = getFileNameFromPath(files[i].c_str());
		std::string out_path = output_folder + "/" + ori_fname;
		if(!out_img.save(out_path)) {
			std::cerr << "Cannot save " << out_path << std::endl;
		} else {
			std::cout << "File " << out_path << " saved" << std::endl;
		}

		// update a sliding window buffer
		// corner cases
		int ending_ind = std::min(i + 1 + n/2, (int)files.size() - 1);
		processor.add_image(files[ending_ind]);
	}
}

void lab3(std::vector<std::string> &files, const std::string output_folder) {
	printf("There are %d files \n", (int)files.size());
	printf("Output folder: %s \n", output_folder.c_str());

	int n = 3;
	int mp_thread = 16;
	int chunk_size = files.size() / mp_thread;
	for(int i = 0; i < mp_thread; ++i) {
		size_t offset = chunk_size * i;
		std::vector<std::string>::const_iterator begin_iter = files.begin() + offset;
		std::vector<std::string>::const_iterator end_iter = begin_iter + chunk_size;
		std::vector<std::string> cur_thread_files(begin_iter, end_iter);
		openmp_woker(cur_thread_files, output_folder, n, algorithm::gaussian);
		break;
	}

}

std::vector<std::string> read_files(const std::string input_folder) {
	int n = 1024;
	// read files from folder
	std::vector<std::string> ret(n);
	
	// fix prefix
	for(int i = 0; i < n; ++i) {
		char buff[100];
		snprintf(buff, sizeof(buff), "/home/ysheng/Downloads/videos/%04d.png", i+450);
		ret[i] = buff;
	}

	return ret;
}

int main(int argc, char* argv[]) {
	GC(cudaSetDevice(1));

	std::string inputs = "inputs";
	std::string out_folder = "video_out";
	std::vector<std::string> video_files = read_files(inputs);

	lab3(video_files, out_folder);

	return 0;
}