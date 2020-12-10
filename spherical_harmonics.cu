#include "common.h"
#include "glm/ext/vector_float3.hpp"
#include "graphics_lib/Render/mesh.h"
#include "spherical_harmonics.h"
#include <iterator>
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
        vec3 p(std::sin(a) * std::cos(b), std::sin(a) * std::sin(b), std::cos(a));
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
    for(int i = 1; i <=l; i += 2) {
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
        return K(l,0) * P(l, 0, std::cos(theta));
    
    if (m > 0) 
        return sqrt2 * K(l,m) * std::cos(m * phi) * P(l, m, std::cos(theta));
    
    return sqrt2 * K(l,-m) * std::sin(-m * phi) * P(l, -m, std::cos(theta)); 
}

std::vector<SH_sample> SH_init(int band, int num) {
    int sample_num = num * band * band;
    
    // memory layout: 
    // band, series x N samples 
    static std::vector<SH_sample> ret;
    if (ret.size() == sample_num)
        return ret;
        
    ret.resize(sample_num);
    auto samples = uniform_sphere_2d_samples(num);
    int sample_i = 0;
    for(int l = 0; l < band; ++l) {
        for(int m = -l; m <= l; ++m) {
            for (int i = 0; i < num; ++i) {
                float theta = samples[i].x, phi = samples[i].y;
                vec3 p(std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), cos(theta));
                
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
std::vector<float> SH_func(std::function<float(float theta, float phi)> func, int band, int n) {
    auto samples = SH_init(band, n);
    int band_n = band * band;
    std::vector<float> ret(band_n);

    const float mc_factor = 4.0f * pd::pi /(float)n;
    for(int l = 0; l < band_n; ++l){
            float c = 0.0f;

            // monte-carlo integration
            for(int si = 0; si < n; ++si) {
                c += func(samples[l * n + si].sph.x, samples[l * n + si].sph.y) * samples[l * n + si].c;
            }

            ret[l] = c * mc_factor;
    }

    return ret; 
}

void omp_no_shadow(std::vector<vec3> &norms, std::vector<SH_sample> &sh_samples, int band, std::vector<float> &coeffs) {
	int sample_num = (int)sh_samples.size()/(band * band);
	const float mc_factor = 4.0f * pd::pi / (float)sample_num;

#pragma omp parallel for
	for(int vi = 0; vi < norms.size(); ++vi) {
		for (int l = 0; l < band * band; ++l) {
			float c = 0.0f;
			// monte-carlo integration
			for(int si = 0; si < sample_num; ++si) {
                float dot_term = std::max(glm::dot(norms[vi], sh_samples[l * sample_num + si].vec), 0.0f);
				c +=
                dot_term *
				sh_samples[l * sample_num + si].c;
			}
			coeffs[vi * band * band + l] = c * mc_factor;		
		}
	}
}

struct ray {
	glm::vec3 ro, rd;
};

__host__ __device__
float ray_triangle_intersect(ray r, vec3 p0, vec3 p1, vec3 p2, bool& ret) {
    glm::vec3 v0v1 = p1 - p0;
	glm::vec3 v0v2 = p2 - p0;
	glm::mat3 m;
	m[0] = -r.rd; m[1] = v0v1; m[2] = v0v2;
	glm::vec3 b = r.ro - p0;

	if (std::abs(glm::determinant(m)) < 1e-6f) {
		ret = false;
		return 0.0f;
	}

	glm::vec3 x = glm::inverse(m) * b;
	float t = x.x, u = x.y, v = x.z;
	if (t <= 0.0 || u < 0.0 || v < 0.0 || u > 1.0 || v >1.0 || u + v < 0.0 || u + v > 1.0) {
		ret = false; return 0.0f;
	}
	
	ret = true;
	return std::sqrt(glm::dot(r.rd * t, r.rd * t));
}

__host__ __device__
int point_visible(vec3 p, vec3 *scene, int N, vec3 dir) {
    float eps = 1e-3f;
    ray r = {p + glm::normalize(dir) * eps, dir};
    for(int i = 0; i < N/3; ++i) {
        bool intersect = 0;
        ray_triangle_intersect(r, scene[3 * i + 0], scene[3 * i + 1], scene[3 * i + 2], intersect);
        if (intersect) return 0;
    }
    return 1;
}

void omp_shadow(std::vector<vec3> &world_verts, std::vector<vec3> &norms, std::vector<glm::vec3> &scene, std::vector<SH_sample> &sh_samples, int band, std::vector<float> &coeffs) {
	int sample_num = (int)sh_samples.size()/(band * band);
	const float mc_factor = 4.0f * pd::pi / (float)sample_num;

#pragma omp parallel for
	for(int vi = 0; vi < norms.size(); ++vi) {
		for (int l = 0; l < band * band; ++l) {
			float c = 0.0f;
			// monte-carlo integration
			for(int si = 0; si < sample_num; ++si) {
                float dot_term = std::max(glm::dot(norms[vi], sh_samples[l * sample_num + si].vec), 0.0f);
                bool visibility = point_visible(world_verts[vi], scene.data(), (int)scene.size(), sh_samples[si].vec);
                if (visibility) {
                    c +=
                    dot_term *
                    sh_samples[l * sample_num + si].c;
                }
			}
			coeffs[vi * band * band + l] = c * mc_factor;		
		}
	}
}
    
void compute_sh_coeff(std::shared_ptr<mesh> mesh_ptr, std::vector<vec3> &scene, int band, int n) {
	if(!mesh_ptr) return;

	if(mesh_ptr->m_norms.empty()) {
		mesh_ptr->recompute_normal();
	}

	mesh_ptr->m_band = band;
	mesh_ptr->m_sh_coeffs.resize(band * band * mesh_ptr->m_norms.size());
	
    auto sh_samples = SH_init(band, n);
    // omp_no_shadow(mesh_ptr->m_norms, sh_samples, band, mesh_ptr->m_sh_coeffs);
    auto mesh_world_verts = mesh_ptr->compute_world_space_coords();

    omp_shadow(mesh_world_verts, mesh_ptr->m_norms, scene, sh_samples, band, mesh_ptr->m_sh_coeffs);
}

__global__
void cuda_no_shadow(glm::vec3 *norms, int norm_n, SH_sample *sh_samples, int sh_n, int band, float *d_coeffs) {
    int ind = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    int sample_num = sh_n / (band * band);
    float mc_factor = 4.0f * 3.1415926f/(float)sample_num;
    for(int vi = ind; vi < norm_n; vi += stride) {
        for(int l = 0; l < band * band; ++l) {
            float c = 0.0f;
            for(int si = 0; si < sample_num; ++si) {
                float dot_term = glm::dot(glm::normalize(norms[vi]), glm::normalize(sh_samples[l * sample_num + si].vec));
                if (dot_term < 0.0) dot_term = 0.0f;

                c += dot_term * sh_samples[l * sample_num + si].c;
            }
            d_coeffs[vi * band * band + l] = c * mc_factor;
        }
    }
}

struct mesh_info {
    vec3 *verts;
    vec3 *norms;
};

__global__
void cuda_shadow(mesh_info cur_mesh, int vn, vec3 *scene, int scene_n, SH_sample *sh_samples, int sh_n, int band, int *shadow_buffer,float *d_coeffs) {
    int ind = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    int sample_num = sh_n / (band * band);
    float mc_factor = 4.0f * 3.1415926f/(float)sample_num;

    for(int vi = ind; vi < vn; vi += stride) {
        for(int l = 0; l < band * band; ++l) {
            float c = 0.0f;
            for(int si = 0; si < sample_num; ++si) {
                if (shadow_buffer[vi * sample_num + si])
                    c += sh_samples[l * sample_num + si].c;
            }
            d_coeffs[vi * band * band + l] = c * mc_factor;
        }
    }
}

void cuda_compute_sh_coeff(std::vector<std::shared_ptr<mesh>> &scene,  int band, int n, bool is_shadow) {
    // auto samples = SH_init(band, n);
    // std::vector<vec3> scene_verts;

    // std::vector<std::vector<vec3>> each_world_verts;
    // for(int i = 0; i < scene.size(); ++i) {
    //     each_world_verts.push_back(scene[i]->compute_world_space_coords());
    //     scene_verts.insert(scene_verts.end(), each_world_verts[i].begin(), each_world_verts[i].end());
    // }

    // container_cuda<vec3> cuda_scene_verts(scene_verts);
    // container_cuda<SH_sample> cuda_sh_samples(samples);

    // for(int mi = 0; mi < scene.size(); ++mi) {
    //     auto cur_mesh = scene[mi];
    //     cur_mesh->m_band = band;
    //     cur_mesh->m_sh_coeffs.resize(band * band * cur_mesh->m_verts.size());

    //     auto world_norms = cur_mesh->compute_world_space_normals();

    //     container_cuda<vec3> cuda_verts(each_world_verts[mi]);
    //     container_cuda<vec3> cuda_norms(world_norms);
    //     container_cuda<float> cuda_coeffs(cur_mesh->m_sh_coeffs);
    //     int grid = 512, block = (grid + cur_mesh->m_norms.size() -1)/grid;
    //     if (is_shadow) {
    //         mesh_info cur_info = {cuda_verts.get_d(), cuda_norms.get_d()};
    //         // compute diffuse-shadow
    //         cuda_shadow<<<grid,block>>>(
    //             cur_info,
    //             cuda_norms.get_n(), 
    //             cuda_scene_verts.get_d(), 
    //             cuda_scene_verts.get_n(),
    //             cuda_sh_samples.get_d(), 
    //             cuda_sh_samples.get_n(), 
    //             band, 
    //             cuda_coeffs.get_d());
    //     } else {
    //         cuda_no_shadow<<<grid, block>>>(cuda_norms.get_d(), cuda_norms.get_n(), cuda_sh_samples.get_d(), cuda_sh_samples.get_n(), band, cuda_coeffs.get_d());
    //     }
    //     GC(cudaDeviceSynchronize());
    //     cuda_coeffs.mem_copy_back();
    // }
}

void cuda_compute_sh_coeff(std::shared_ptr<mesh> mesh_ptr, int band, int n) {
    auto samples = SH_init(band, n);
    container_cuda<SH_sample> cuda_sh_samples(samples);

    auto cur_mesh = mesh_ptr;
    cur_mesh->m_band = band;
    cur_mesh->m_sh_coeffs.resize(band * band * cur_mesh->m_verts.size());

    auto world_norms = cur_mesh->compute_world_space_normals();

    container_cuda<vec3> cuda_norms(world_norms);
    container_cuda<float> cuda_coeffs(cur_mesh->m_sh_coeffs);
    int grid = 512, block = (grid + cur_mesh->m_norms.size() -1)/grid;
    cuda_no_shadow<<<grid, block>>>(cuda_norms.get_d(), cuda_norms.get_n(), cuda_sh_samples.get_d(), cuda_sh_samples.get_n(), band, cuda_coeffs.get_d());
    GC(cudaDeviceSynchronize());
    cuda_coeffs.mem_copy_back();
}

__global__
void precompute_shadow_buffer(vec3 *verts, int vn, vec3 *scene, int n, SH_sample *sh_samples, int sn, int *shadow_buffer) {
    int ind = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for(int i = ind; i < vn; i += stride) {
        for(int si = 0; si < sn; ++si) {
            if (point_visible(verts[i], scene, n, sh_samples[si].vec)) {
                shadow_buffer[i * sn + si] = 1;
            } else {
                shadow_buffer[i * sn + si] = 0;
            }
        }
    }
}

void cuda_compute_shadow_sh_coeff(std::shared_ptr<mesh> mesh_ptr, std::vector<glm::vec3> &scene, int band, int n, std::vector<float> &shadow_coeffs) {
    auto samples = SH_init(band, n);

    container_cuda<vec3> cuda_scene_verts(scene);
    container_cuda<SH_sample> cuda_sh_samples(samples);

    mesh_ptr->m_band = band;
    mesh_ptr->m_sh_coeffs.resize(band * band * mesh_ptr->m_verts.size());

    auto world_norms = mesh_ptr->compute_world_space_normals();
    auto world_verts = mesh_ptr->compute_world_space_coords();

    container_cuda<vec3> cuda_verts(world_verts);
    
    // store sh visible infos for all verts 
    std::vector<int> shadow_buffer(n * (int)world_verts.size());
    container_cuda<int> cuda_shadow_buffer(shadow_buffer);
    int grid = 512, block = (grid + world_verts.size() -1)/grid;

    precompute_shadow_buffer<<<grid, block>>>(
        cuda_verts.get_d(), 
        cuda_verts.get_n(), 
        cuda_scene_verts.get_d(), 
        cuda_scene_verts.get_n(), 
        cuda_sh_samples.get_d(), 
        n, 
        cuda_shadow_buffer.get_d());
    GC(cudaDeviceSynchronize());

    container_cuda<vec3> cuda_norms(world_norms);
    container_cuda<float> cuda_coeffs(shadow_coeffs);

    mesh_info cur_info = {cuda_verts.get_d(), cuda_norms.get_d()};
    cuda_shadow<<<grid,block>>>(
        cur_info,
        cuda_norms.get_n(), 
        cuda_scene_verts.get_d(), 
        cuda_scene_verts.get_n(),
        cuda_sh_samples.get_d(), 
        cuda_sh_samples.get_n(), 
        band, 
        cuda_shadow_buffer.get_d(),
        cuda_coeffs.get_d());
    GC(cudaDeviceSynchronize());
    cuda_coeffs.mem_copy_back();
}

void sh_render(std::shared_ptr<mesh> mesh_ptr, std::vector<float> light_coeffs) {
    if (mesh_ptr->m_sh_coeffs.empty())
        return;

    int vn = (int)mesh_ptr->m_colors.size();
    int coeff_n = (int)light_coeffs.size();

#pragma omp parallel for
    for(int vi = 0; vi < vn; ++vi) {
        
        vec3 color(0.0f);
        for(int ci = 0; ci < coeff_n; ++ci) {
            color += mesh_ptr->m_sh_coeffs[vi * coeff_n + ci] * light_coeffs[ci];
        }

        mesh_ptr->m_colors[vi] = mesh_ptr->m_colors[vi] * color;
    }
}

void sh_render(std::vector<std::shared_ptr<mesh>> meshes, const std::vector<float> &light_coeffs) {
    for(int i = 0; i < meshes.size(); ++i) {
        sh_render(meshes[i], light_coeffs);
    }
}

__host__ __device__ 
float ibl_light(float theta, float phi) {
    float u = phi / (3.1415926f * 2.0f);
    float v = 1.0f - theta / (3.1415926f);
    
	float m_u_min = 0.29f;
	float m_u_max = 0.346f;
    float m_v_min = 0.765f;
	float m_v_max = 0.84f;
	float m_intensity = 11.0f;	

    if (u > m_u_min && u < m_u_max && v > m_v_min && v < m_v_max)
        return m_intensity;
    
    return 0.0f;
}

__global__
void gt_lights(vec3 *verts, int N, vec2 *samples, int sn,vec3 *o_colors) {
    int ind = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = ind; i < N; i += stride) {
        vec3 p = verts[i];

        float v = 0.0f;
        float w = 1.0f/sn;
        for(int si = 0; si < sn; ++si) {
            float a = samples[si].x, b = samples[si].y;
            vec3 dir(std::sin(a) * std::cos(b), std::sin(a) * std::sin(b), std::cos(a));
            bool visibility = point_visible(p, verts, N, dir);
            float l = ibl_light(a, b);

            if (visibility) {
                v += l;
            }
        }
        v = v * w / 11.0f;
        
        o_colors[i] = vec3(v);
    }
}

void gt_render(std::vector<std::shared_ptr<mesh>> meshes) {
    std::vector<vec3> verts;
    std::vector<vec3> colors;
    for(int i = 0; i < meshes.size(); ++i) {
        auto world_verts = meshes[i]->compute_world_space_coords();
        verts.insert(verts.end(), world_verts.begin(), world_verts.end());
        colors.insert(colors.end(), meshes[i]->m_colors.begin(), meshes[i]->m_colors.end());
    }

    auto samples = uniform_sphere_2d_samples(100);
    auto d_verts = container_cuda<vec3>(verts);
    auto d_colors = container_cuda<vec3>(colors);
    auto d_samples = container_cuda<vec2>(samples);

    //todo, figure out how to deal with c++11 std::function<...> with function pointer
    int grid = 512, block = (grid + d_verts.get_n() - 1)/ grid;
    gt_lights<<<grid, block>>>(d_verts.get_d(), d_verts.get_n(), d_samples.get_d(), d_samples.get_n(), d_colors.get_d());
    GC(cudaDeviceSynchronize());
    d_colors.mem_copy_back();

    size_t offset = 0;
    for(int i = 0; i < meshes.size(); ++i) {
        std::copy(colors.begin() + offset,colors.begin() + offset + meshes[i]->m_colors.size(), meshes[i]->m_colors.begin());
        offset += meshes[i]->m_colors.size();
    }
}