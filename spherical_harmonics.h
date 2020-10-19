#pragma once
#include <vector>
#include <math.h>

#include <graphics_lib/Utilities/Utils.h>
#include <glm/common.hpp>

std::vector<glm::vec2> uniform_sphere_2d_samples(int n);
std::vector<glm::vec3> uniform_sphere_3d_samples(int n);

struct SH_sample {
    glm::vec3 pos;
    glm::vec3 sph;
    float coeffect;
};