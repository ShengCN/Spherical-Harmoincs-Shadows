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
#include "otb_window.h"

using namespace glm;

int w = 1024, h = 720;
int main(int argc, char* argv[]) {
	GC(cudaSetDevice(1));

    otb_window wnd;
    wnd.create_window(w, h,"OTB");
    wnd.show();

	return 0;
}