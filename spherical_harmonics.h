#pragma once
#include <vector>
#include <math.h>
#include <functional>

#include <graphics_lib/Utilities/Utils.h>
#include <glm/common.hpp>

std::vector<glm::vec2> uniform_sphere_2d_samples(int n);
std::vector<glm::vec3> uniform_sphere_3d_samples(int n);

struct SH_sample {
    glm::vec2 sph;  // theta, phi
    glm::vec3 vec;  // x,y,z
    std::vector<float> coeffs;
    int band;
    
    SH_sample(int band):band(band) {
        init_coeffs();
    }

    SH_sample(int band, glm::vec2 sph, glm::vec3 vec):
    band(band), sph(sph), vec(vec) {
        init_coeffs();
    }
private:
    void init_coeffs() {
        coeffs.resize(band * band, 0.0f);
    }
};

float SH(int l, int m, float theta, float phi);
std::vector<SH_sample> SH_init(int band, int num);
std::vector<float> SH_func(std::function<float(float theta, float phi)> func, const std::vector<SH_sample> &samples);