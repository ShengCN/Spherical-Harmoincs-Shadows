#include "spherical_harmonics.h"
#include <limits>
#include <omp.h>

using glm::vec2;
using glm::vec3;

std::vector<vec2> uniform_sphere_2d_samples(int n) {
    std::vector<vec2> ret(n);
    for(int i = 0;i < n; ++i) {
        float x = pd::random_float();
        float y = pd::random_float();
        ret[i] = vec2(2.0f * std::acos(std::sqrt(1.0f - x)), 2.0f * pd::pi * y);
    }
    return ret;
}

std::vector<vec3> uniform_sphere_3d_samples(int n) {
    auto samples = uniform_sphere_2d_samples(n);
    std::vector<vec3> ret(n);

    for(int i = 0; i < n; ++i) {
        float a = samples[i].x, b = samples[i].y;
        vec3 p(std::sin(a) * std::cos(b), std::cos(a),std::sin(a) * std::sin(b));
        ret[i] = p;
    }
    return ret;
}

long long factorial(int l) {
    long long ret = 1;
    for(int i = 2; i <= l; ++i) {
        ret = ret * i;
    }
    return ret;
}

long long dfactorial(int l) {
    long long ret = 1;
    for(int i = l; i >= 2.0; i=i-2) {
        ret *= i;
    }
    return ret;
}

float K(int l, int m) {
    float k = ((2.0f * l + 1.0f) * factorial(l-std::abs(m)))/(4.0f * pd::pi * factorial(l + std::abs(m)));
    return std::sqrt(k);
}

float P(int l, int m, float x) {
    if (m == 0 && l == 0) {
        return 1.0f;
    }

    if (l == m) {
        float sqrt_t = std::sqrt(1.0-x * x);
        return std::pow(-1, m) * dfactorial(2 * m - 1) * std::pow(sqrt_t, m);
    }
    
    if (l == m+1) {
        return x * (2.0f * m + 1.0f) * P(m,m,x);
    }

    return (x * (2.0f * l - 1.0f) * P(l-1, m, x) - (l+m-1) * P(l-2, m, x))/(l-m);
}

const float sqrt2 = std::sqrt(2.0f);
float SH(int l, int m, float theta, float phi) {
    if (m==0) 
        return K(l,0) * P(l, 0, cos(theta));
    
    if (m > 0) 
        return sqrt2 * K(l,m) * cos(m * phi) * P(l, m, cos(theta));
    
    return sqrt2 * K(l,-m) * sin(-m * phi) * P(l, -m, cos(theta)); 
}

std::vector<SH_sample> SH_init(int band, int num) {
    int sample_num = num * band * band;
    
    // memory layout: 
    // band, series x N samples 
    std::vector<SH_sample> ret(sample_num);

    auto samples = uniform_sphere_2d_samples(num);
    int sample_i = 0;
    for(int l = 0; l < band; ++l) {
        for(int m = -l; m <= l; ++m) {
            for (int i = 0; i < num; ++i) {
                float a = samples[i].x, b = samples[i].y;
                vec3 p(std::sin(a) * std::cos(b), std::cos(a),std::sin(a) * std::sin(b));
                
                ret[sample_i].sph = samples[i];
                ret[sample_i].vec = p;

                // coefficients
                ret[sample_i].c = SH(l, m, ret[sample_i].sph.x, ret[sample_i].sph.y);
                sample_i += 1;
            }
        }
    }

    return ret;
}

// return a sparse band/series matrix
std::vector<float> SH_func(std::function<float(float theta, float phi)> func, const std::vector<SH_sample> &samples, int band) {
    int band_n = band * band;
    std::vector<float> ret(band_n);
    int sample_num = (int)samples.size() / band_n;
    int ind = 0;

    const float mc_factor = 4.0f * pd::pi /(float)sample_num;
    for (int l = 0; l < band; ++l) {
        for(int m = -l; m <=l; ++m) {
            float c = 0.0f;

            // monte-carlo integration
            for(int si = 0; si < sample_num; ++si) {
                c += func(samples[si].sph.x, samples[si].sph.y) * samples[si].c * mc_factor;
            }

            ret[ind++] = c;
        }
    }

    return ret; 
}

void compute_sh_coeff(std::shared_ptr<mesh> mesh_ptr, int band, int n) {
	if(!mesh_ptr) return;

	if(mesh_ptr->m_norms.empty()) {
		mesh_ptr->recompute_normal();
	}

	mesh_ptr->m_band = band;
	mesh_ptr->m_sh_coeffs.resize(band * band * mesh_ptr->m_norms.size());
	
    auto sh_samples = SH_init(band, n);
	int sample_num = (int)sh_samples.size()/(band * band);
	const float mc_factor = 4.0f * pd::pi / (float)sample_num;

#pragma omp parallel for
	for(int vi = 0; vi < mesh_ptr->m_norms.size(); ++vi) {
        int ind = 0;
		for(int l = 0; l < band; ++l) for(int m = -l; m <=l; ++m) {
			float c = 0.0f;
			// monte-carlo integration
			for(int si = 0; si < sample_num; ++si) {
				c +=
				std::max(glm::dot(mesh_ptr->m_norms[vi], sh_samples[ind * sample_num + si].vec), 0.0f) * 
				sh_samples[ind * sample_num + si].c * 
				mc_factor;
			}
			mesh_ptr->m_sh_coeffs[vi * band * band + ind++] = c;		
		}
	}
	
	mesh_ptr->compute_sh_texture();
}

__global__
void cuda_no_shadow(glm::vec3 *norms, int norm_n, SH_sample *sh_samples, int sh_n, int band, float *d_coeffs) {
    int ind = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    int sample_num = sh_n / (band * band);
    float mc_factor = 4.0f * 3.1415926f/(float)sample_num;
    for(int vi = ind; vi < norm_n; vi += stride) {
        int ind = 0;
        for(int l = 0; l < band; ++l) for(int m = -l; m <= l; ++m) {
            float c = 0.0f;
            for(int si = 0; si < sample_num; ++si) {
                float dot_term = glm::dot(norms[vi], sh_samples[ind * sample_num + si].vec);
                if (dot_term < 0.0) dot_term = 0.0f;

                c += dot_term * sh_samples[ind * sample_num + si].c * mc_factor;
            }
            d_coeffs[vi * band * band + ind++] = c;
        }
    }
}

void cuda_compute_sh_coeff(std::shared_ptr<mesh> mesh_ptr, int band, int n) {
	if(!mesh_ptr) return;

	if(mesh_ptr->m_norms.empty()) {
		mesh_ptr->recompute_normal();
	}

	mesh_ptr->m_band = band;
	mesh_ptr->m_sh_coeffs.resize(band * band * mesh_ptr->m_norms.size());
	
    auto sh_samples = SH_init(band, n);
	int sample_num = (int)sh_samples.size()/(band * band);

    // memory allocation
    glm::vec3 *d_norms;         size_t d_norms_size = sizeof(glm::vec3) * mesh_ptr->m_norms.size();
    SH_sample *d_sh_samples;    size_t d_sh_samples_size = sizeof(SH_sample) * sh_samples.size();
    float *d_coeffs;            size_t d_coeffs_size = sizeof(float) * band * band * mesh_ptr->m_norms.size();

    GC(cudaMalloc(&d_norms, d_norms_size));
    GC(cudaMalloc(&d_sh_samples, d_sh_samples_size));
    GC(cudaMalloc(&d_coeffs, d_coeffs_size));
    GC(cudaMemcpy(d_norms, mesh_ptr->m_norms.data(), d_norms_size, cudaMemcpyHostToDevice));
    GC(cudaMemcpy(d_sh_samples, sh_samples.data(), d_norms_size, cudaMemcpyHostToDevice))

    // cuda computation
    int grid = 512, block = (grid + mesh_ptr->m_norms.size() -1)/grid;
    cuda_no_shadow<<<grid,block>>>(d_norms, mesh_ptr->m_norms.size(), d_sh_samples, sh_samples.size(), band, d_coeffs);
    GC(cudaDeviceSynchronize());

    // memory copy back
    GC(cudaMemcpy(mesh_ptr->m_sh_coeffs.data(),d_coeffs, d_coeffs_size, cudaMemcpyDeviceToHost));

    // memory free
    cudaFree(d_norms);
    cudaFree(d_sh_samples);

    mesh_ptr->compute_sh_texture();
}