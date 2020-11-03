#pragma once
#include <vector>
#include <string>
#include "common.h"

/*!
 * \class Interface class for materials
 *
 * \brief 
 *
 * \author YichenSheng
 * \date September 2019
 */
class material
{
public:
	material();
	~material();
};

class img_texutre {
public:
	img_texutre();
	~img_texutre() {}
	bool read_img(const std::string path);
	glm::vec3 at(float u, float v) const;
public:
	unsigned int ogl_tex_id;
	int w, h, c;
	std::vector<unsigned char> img;
};