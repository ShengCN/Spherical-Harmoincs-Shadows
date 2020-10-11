#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <graphics_lib/Render/mesh.h>

struct heightmap {
    std::vector<float> heights;
    int w,h;

    heightmap(int w, int h): w(w), h(h) {
        heights.resize(w * h, 0.0f);
    }

    void set_height(int x, int y, float h) {
        if(x > w - 1 || x < 0 || y > h-1 || y < 0) {
            return;
        }

        heights[y * w + x] = h;
    }
    
    float& at(int x, int y) {
        return heights[y * w + x]; 
    }
};

struct line {
    float x, y;
    glm::vec2 normal;
};

void lab5_init(int w, int h, std::shared_ptr<mesh> m);

// tt: total time
// cuda_time: cuda processing time
void lab5(int n, std::shared_ptr<mesh> m, float &tt, float &cuda_time, float &mem_cpy_time, float &mem_cpy_back_time);
