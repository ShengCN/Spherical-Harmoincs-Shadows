#include "Utils.h"
#define STB_IMAGE_IMPLEMENTATION
#include "Dep/stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "Dep/stb/stb_image_write.h"

bool purdue::save_image(const std::string fname, unsigned int *pixels, int w, int h, int c /*= 4*/) {
	return stbi_write_png(fname.c_str(), w, h, c, pixels, w*c);
}

std::default_random_engine generator;

float purdue::normal_random(float mean, float sig) {
	std::normal_distribution<float> distribution(mean, sig);
	return distribution(generator);
}