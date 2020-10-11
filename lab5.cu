#include "lab5.h"
#include <stdio.h>
#include <cassert>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <random>
#include <cmath>
#include "helper_cuda.hpp"
#include "glm/vec3.hpp"
#include <graphics_lib/Utilities/Utils.h>

// height map -> mesh
void heightmap_mesh(heightmap &hm, std::shared_ptr<mesh> m) {
    if(!m) return;

    m->clear_vertices();
    for(int i = 0; i < hm.w-1; ++i) {
        for(int j = 0; j < hm.h-1; ++j) {
            //  p0 p1
            //  p2 p3
            float h = sin(i) * cos(j);
            // vec3 p0 = vec3(i,hm.at(i,j),j);
            // vec3 p1 = vec3(i,hm.at(i+1,j),j);
            // vec3 p2 = vec3(i,hm.at(i,j+1),j);
            // vec3 p3 = vec3(i,hm.at(i+1,j+1),j);
            vec3 p0 = vec3(i, sin(i) * cos(j),j);
            vec3 p1 = vec3(i + 1, sin(i+1) * cos(j),j);
            vec3 p2 = vec3(i, sin(i) * cos(j+1),j + 1);
            vec3 p3 = vec3(i+1, sin(i+1) * cos(j+1),j+1);

            m->add_face(p0, p2, p1);
            m->add_face(p1, p2, p3);
        }
    }
    
    m->normalize_position_orientation();
    m->set_color(vec3(0.8f));
}

void lab5_init(int w, int h, std::shared_ptr<mesh> m) {
    heightmap hm(w,h);
    heightmap_mesh(hm, m);
}

void lab5(std::shared_ptr<mesh> m) { 
    GC(cudaSetDevice(1));
}