#include "lab5.h"
#include <stdio.h>
#include <cassert>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <random>
#include <cmath>
#include <omp.h>
#include "helper_cuda.hpp"
#include "glm/vec3.hpp"
#include <graphics_lib/Utilities/Utils.h>

heightmap hm(0,0);
const int max_n = 1 << 11;
__constant__ line d_lines[max_n];
__constant__ float d_decrease_factor = 0.9f;
float h_decrease_factor = 0.9f;
__constant__ float d_init_elevation = 0.5f;
float h_init_elevation = 0.01f;

// height map -> mesh
void heightmap_mesh(heightmap &hm, std::shared_ptr<mesh> m) {
    if(!m) return;

    m->clear_vertices();
    for(int i = 0; i < hm.w-1; ++i) {
        for(int j = 0; j < hm.h-1; ++j) {
            //  p0 p1
            //  p2 p3
            vec3 p0 = vec3(i, hm.at(i,j),j);
            vec3 p1 = vec3(i + 1, hm.at(i+1, j),j);
            vec3 p2 = vec3(i, hm.at(i, j+1),j + 1);
            vec3 p3 = vec3(i+1, hm.at(i+1, j+1),j+1);

            m->add_face(p0, p2, p1);
            m->add_face(p1, p2, p3);
        }
    }
    
    m->normalize_position_orientation();
    m->set_color(vec3(0.8f));
}

__host__ __device__
bool is_right(line l, float px, float py) {
    glm::vec2 normal = l.normal;
    float x = l.x;
    float y = l.y;
    return glm::dot(normal, glm::vec2(px-x, py-y)) > 0.0f;
} 

line random_line(float w, float h) {
    line ret;
    ret.x = pd::random_float(0, w);
    ret.y = pd::random_float(0, h);
    ret.normal = vec2(pd::random_float(-1.0f,1.0f), pd::random_float(-1.0f,1.0f));
    return ret;
}

void lab5_init(int w, int h, std::shared_ptr<mesh> m) {
    hm = heightmap(w,h);
    heightmap_mesh(hm, m);
}

std::vector<line> random_faults(int n, int w, int h) {
    std::vector<line> ret(n);

#pragma omp parallel for
    for(int i = 0; i < n; ++i) {
        ret[i] = random_line(w, h);
    }

    return ret;
}

__global__
void compute_heightmap(int n, float *map, int w, int h) {
    int ind = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for(int i = ind; i < w * h; i += stride) {
        map[i] = 0;

        float mapx = i / w + 0.5f;
        float mapy = i - mapx * w + 0.5f;
        
        float dt = d_init_elevation * w;
        for(int fi = 0; fi < n; ++fi) {
            float elevation = -dt;
            if(is_right(d_lines[fi], mapx, mapy)) {
                elevation = dt; 
            }             
            
            map[i] += elevation;
            // decrease the intensity of changes
            dt = dt * d_decrease_factor;
        }
    }
}

cuda_timer mem_cpy_timer;
cuda_timer mem_cpy_back_timer;
void faults_heightmap(std::vector<line> faults, float &t0, float &t1) {
    mem_cpy_timer.tic();
    
    // memory set up
    GC(cudaMemcpyToSymbol(d_lines, &faults[0], sizeof(line) * faults.size()));
    GC(cudaMemcpyToSymbol(d_decrease_factor, &h_decrease_factor, sizeof(float)));
    GC(cudaMemcpyToSymbol(d_init_elevation, &h_init_elevation, sizeof(float)));
    float *d_map;
    GC(cudaMalloc(&d_map, sizeof(float) * hm.w * hm.h));
    mem_cpy_timer.toc();

    // kernel 
    int grid = 512, thread = (hm.w * hm.h + grid -1)/grid; 
    compute_heightmap<<<grid,thread>>>(faults.size(), d_map, hm.w, hm.h);
    GC(cudaDeviceSynchronize());

    // memory copyback
    mem_cpy_back_timer.tic();
    GC(cudaMemcpy(hm.heights.data(), d_map, sizeof(float) * hm.w * hm.h, cudaMemcpyDeviceToHost));
    mem_cpy_back_timer.toc();
    GC(cudaFree(d_map));
    
    t0 = mem_cpy_timer.get_time();
    t1 = mem_cpy_back_timer.get_time();
}

void lab5(int n, std::shared_ptr<mesh> m, float &tt, float &t, float &mem_cpy_time, float &mem_cpy_back_time) { 
    printf("There are %d faults \n", n);

    pd::timer cpu_timer;
    cuda_timer gpu_timer;

    cpu_timer.tic();
    GC(cudaSetDevice(1));
    
    // random faults
    std::vector<line> faults = random_faults(n, hm.w, hm.h);
    // CUDA parallel compute heightmap
    gpu_timer.tic();
    faults_heightmap(faults, mem_cpy_time, mem_cpy_back_time);
    gpu_timer.toc();
    cpu_timer.toc();
    
    // faults to mesh
    heightmap_mesh(hm, m);

    tt = cpu_timer.get_elapse() * 1e-9;
    t = gpu_timer.get_time();
}