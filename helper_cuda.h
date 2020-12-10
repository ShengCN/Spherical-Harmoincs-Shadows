#include <cuda_runtime.h>
#include <sstream>
#include <vector>

class cuda_timer {
public:
    cuda_timer() {
        cudaEventCreate(&m_tic);
        cudaEventCreate(&m_toc);
    }

    ~cuda_timer() {
        cudaEventDestroy(m_tic);
        cudaEventDestroy(m_toc);
    }

    void tic() {
        cudaEventRecord(m_tic, 0);
    }

    void toc() {
        cudaEventRecord(m_toc, 0);
        cudaEventSynchronize(m_toc);
    }

    float get_time() {
        float time;
        cudaEventElapsedTime(&time, m_tic, m_toc);
        return time;
    }

private:
    cudaEvent_t m_tic, m_toc;
};

// src: https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define GC(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

template<typename T>
class container_cuda {
public:
    container_cuda(std::vector<T> &ori):h_ori(ori) {
        size = sizeof(T) * h_ori.size();
        GC(cudaMalloc(&d_ori, size));
        GC(cudaMemcpy(d_ori, h_ori.data(), size, cudaMemcpyHostToDevice));
    }
    ~container_cuda() {
        cudaFree(d_ori);
    }

    T* get_d() {
        return d_ori;
    }

    int get_n() {
        return (int)h_ori.size();
    }

    int mem_copy_back() {
        cudaError_t code = cudaMemcpy(h_ori.data(), d_ori, size, cudaMemcpyDeviceToHost);
        GC(code);
        return code == cudaSuccess;
    }

private:
    std::vector<T> &h_ori;
    T *d_ori;
    size_t size;
};