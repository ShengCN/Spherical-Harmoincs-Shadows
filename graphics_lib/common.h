#pragma once

#include <glad/glad.h>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/mat3x3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtx/quaternion.hpp>
#include <string>
#include <sstream>

namespace purdue {
	typedef float deg;
	typedef float rad;
	constexpr float pi = 3.14159265f;

	inline std::string to_string(glm::vec3 v) {
		std::stringstream out;
		out << v.x << "," << v.y << "," << v.z;
		return out.str();
	}

	inline std::string to_string(glm::vec4 v) {
		std::stringstream out;
		out << v.x << "," << v.y << "," << v.z << "," << v.w;
		return out.str();
	}

	inline std::string to_string(glm::mat4 m) {
		std::stringstream out;
		out << to_string(m[0]) << std::endl;
		out << to_string(m[1]) << std::endl;
		out << to_string(m[2]) << std::endl;
		out << to_string(m[3]);
		return out.str();
	}

	inline bool same_point(const glm::vec3 &a, const glm::vec3 &b) {
		return glm::distance(a, b) < 1e-3;
	}
};

namespace pd = purdue;
using glm::vec2;
using glm::vec3;
using glm::vec4;
using glm::mat3;
using glm::mat4;
using glm::quat;

