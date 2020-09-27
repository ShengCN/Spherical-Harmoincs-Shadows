#include <cuda_runtime.h>
#include <sstream>

class timer {
public:
    timer() {
        cudaEventCreate(&m_tic);
        cudaEventCreate(&m_toc);
    }

    ~timer() {
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