#include "spherical_harmonics.h"
#include <limits>

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

int factorial(int l) {
    int ret = 1;
    for(int i = 2; i <= l; ++i) {
        ret = ret * i;
    }
    return ret;
}

int dfactorial(int l) {
    int ret = 1;
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

constexpr float sqrt2 = std::sqrt(2.0f);
float SH(int l, int m, float theta, float phi) {
    if (m==0) 
        return K(l,0) * P(l, 0, cos(theta));
    
    if (m > 0) 
        return sqrt2 * K(l,m) * cos(m * phi) * P(l, m, cos(theta));
    
    return sqrt2 * K(l,-m) * sin(-m * phi) * P(l, -m, cos(theta)); 
}
