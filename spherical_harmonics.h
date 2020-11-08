#pragma once
#include <vector>
#include <math.h>
#include <functional>

#include "graphics_lib/Render/mesh.h"
#include "graphics_lib/Utilities/Utils.h"
#include <glm/vec3.hpp>
#include <glm/vec2.hpp>

#include "helper_cuda.h"

std::vector<glm::vec2> uniform_sphere_2d_samples(int n);
std::vector<glm::vec3> uniform_sphere_3d_samples(int n);

struct SH_sample {
    glm::vec2 sph;  // theta, phi
    glm::vec3 vec;  // x,y,z
    float c;
};

float SH(int l, int m, float theta, float phi);
std::vector<SH_sample> SH_init(int band, int num);
std::vector<float> SH_func(std::function<float(float theta, float phi)> func, int band, int n);

void compute_sh_coeff(std::shared_ptr<mesh> mesh_ptr, std::vector<glm::vec3> &scene, int band, int n);
void cuda_compute_sh_coeff(std::vector<std::shared_ptr<mesh>> &scene,  int band, int n, bool is_shadow);

// rendering
void sh_render(std::shared_ptr<mesh> mesh_ptr, std::vector<float> &light_coeffs);
void sh_render(std::vector<std::shared_ptr<mesh>> meshes, const std::vector<float> &light_coeffs);
void test_visible(std::vector<std::shared_ptr<mesh>> meshes);