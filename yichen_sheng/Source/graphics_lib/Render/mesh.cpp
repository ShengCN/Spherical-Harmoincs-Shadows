#include <fstream>
#include <iostream>

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/matrix_decompose.hpp>

#include "mesh.h"
#include "Utilities/Utils.h" 
#include "Utilities/Logger.h"

int mesh::id = 0;

mesh::mesh(){
	cur_id = ++id;
}

mesh::~mesh() {
}

std::vector<vec3> mesh::compute_world_space_coords() {
	std::vector<vec3> world_coords = m_verts;
	for (auto& v : world_coords) {
		vec4 tmp = m_world * vec4(v,1.0);
		v = vec3(tmp) / tmp.w;
	}
	return world_coords;
}

std::vector<vec3> mesh::compute_world_space_normals() {
	std::vector<vec3> world_normals = m_norms;
	auto normal_transform = glm::inverse(glm::transpose(m_world));

	for (auto& n : world_normals) {
		n = glm::vec3(normal_transform * vec4(n,0.0f));
	}
	return world_normals;
}

vec3 mesh::compute_center() {
	vec3 center(0.0f, 0.0f, 0.0f);
	float weights = 1.0f/(float)m_verts.size();
	for (auto& v : m_verts) {
		center += v * weights;
	}
	return center;
}

vec3 mesh::compute_world_center() {
	vec3 center = compute_center();
	vec4 tmp = vec4(center, 1.0f);
	tmp = m_world * tmp;

	return tmp / tmp.w;
}

void mesh::add_world_transate(vec3 v) {
	m_world = glm::translate(v) * m_world;
}

void mesh::add_scale(vec3 s) {
	vec3 center = compute_center();
	mat4 scale_mat = glm::scale(s);
	m_world = m_world * scale_mat;
}

void mesh::add_rotate(pd::deg angle, vec3 axis) {
	mat4 rot_mat = glm::rotate(pd::deg2rad(angle), axis);
	m_world = rot_mat * m_world;
}

void mesh::add_face(vec3 va, vec3 vb, vec3 vc) {
	m_verts.push_back(va);
	m_verts.push_back(vb);
	m_verts.push_back(vc);

	vec3 ba = vb - va, cb = vc - vb;
	vec3 normal = glm::normalize(glm::cross(ba, cb));
	m_norms.push_back(normal);
	m_norms.push_back(normal);
	m_norms.push_back(normal);

	vec3 default_stl_color = vec3(0.7f);
	m_colors.push_back(default_stl_color);
	m_colors.push_back(default_stl_color);
	m_colors.push_back(default_stl_color);
}

void mesh::add_vertex(vec3 v, vec3 n, vec3 c) {
	m_verts.push_back(v);
	m_norms.push_back(n);
	m_colors.push_back(c);
}

void mesh::add_vertex(vec3 v, vec3 n, vec3 c, vec2 uv) {
	add_vertex(v, n, c);
	m_uvs.push_back(uv);
}

void mesh::add_vertices(std::vector<vec3>& verts) {
	m_verts.insert(m_verts.end(), verts.begin(), verts.end());
}

AABB mesh::compute_aabb() const {
	assert(m_verts.size() > 0);

	AABB aabb(m_verts[0]);
	for (auto &v : m_verts) {
		aabb.add_point(v);
	}
	return aabb;
}

AABB mesh::compute_world_aabb() {
	auto world_space_points = compute_world_space_coords();
	if (world_space_points.empty())
		return AABB(vec3(0.0f));

	AABB aabb(world_space_points[0]);
	for (auto& v : world_space_points) {
		aabb.add_point(v);
	}

	return aabb;
}

void mesh::set_color(vec3 col) {
	if (m_verts.empty())
		return;

	if (m_colors.empty()) {
		m_colors.clear();
		m_colors.resize(m_verts.size(), col);
	}
	else {
		for (auto &c : m_colors) {
			c = col;
		}
	}

}

void mesh::set_color(unsigned triangle_id, vec3 col) {
	// ensure there exist colors
	if (m_colors.empty()) {
		set_color(glm::vec3(0.3f));
	}

	unsigned int vi = 3 * triangle_id + 0, vj = 3 * triangle_id + 1, vk = 3 * triangle_id + 2;
	m_colors[vi] = m_colors[vj] = m_colors[vk] = col;
}

void mesh::normalize_position_orientation(vec3 scale/*=vec3(1.0f)*/, glm::quat rot_quant /*= glm::quat(0.0f,0.0f,0.0f,1.0f)*/) {
	// normalize, move to center and align
	vec3 center = compute_center();
	float diagnoal = compute_aabb().diag_length();
	mat4 norm_transform = glm::toMat4(rot_quant) * 
		glm::scale(scale/ diagnoal) *
		glm::translate(-center);

	m_world = norm_transform;
}

void mesh::get_demose_matrix(vec3& scale, quat& rot, vec3& translate) {
	glm::vec3 skew;
	glm::vec4 perspective;
	glm::decompose(m_world, // input
				   scale,
				   rot,
				   translate,
				   skew,
				   perspective);
}

void mesh::set_matrix(const vec3 scale, const quat rot, const vec3 translate) {
	m_world = glm::translate(translate) *
		glm::scale(scale) * glm::toMat4(rot);
}

void mesh::recompute_normal() {
	m_norms.clear();
	size_t triangle_num = m_verts.size() / 3;
	for(int ti = 0; ti < triangle_num; ++ti) {
		vec3 a = m_verts[3 * ti + 0];
		vec3 b = m_verts[3 * ti + 1];
		vec3 c = m_verts[3 * ti + 2];

		vec3 n = glm::normalize(glm::cross(b - a, c - b));
		m_norms.push_back(n);
		m_norms.push_back(n);
		m_norms.push_back(n);
	}
}

void mesh::remove_duplicate_vertices() {
	std::vector<bool> vert_visited(m_verts.size(), false);
	for (int vi = 0; vi < m_verts.size(); ++vi) {
		if (vert_visited[vi])	continue;
		
		for (int ovi = vi + 1; ovi < m_verts.size(); ++ovi) {
			if(pd::same_point(m_verts[vi], m_verts[ovi])) {
				vert_visited[ovi] = true;
				m_verts[ovi] = m_verts[vi];
			}
		}
	}
}

std::vector<glm::vec3> AABB::to_tri_mesh() {
	std::vector<glm::vec3> ret;
	auto add_face = [](std::vector<glm::vec3>& ret, vec3 a, vec3 b, vec3 c) {
		ret.push_back(a); ret.push_back(b); ret.push_back(c);
	};

	vec3 diag = diagonal();
	vec3 x = diag * vec3(1.0f, 0.0f, 0.0f), y = diag * vec3(0.0f, 1.0f, 0.0f), z = diag * vec3(0.0f, 0.0f, 1.0f);
	
	//		a--b
	//		c--d
	//  h--e
	//  f--g
	vec3 h = p0,    e = h + x, f = h + z, g = h + x + z;
	vec3 a = h + y, b = e + y, c = f + y, d = g + y;

	// bottom
	add_face(ret, h, e, f); 
	add_face(ret, f, e, g);

	// front
	add_face(ret, f, g, d);
	add_face(ret, d, c, f);

	// left
	add_face(ret, f, c, a);
	add_face(ret, a, h, f);

	// back
	add_face(ret, a, b, e);
	add_face(ret, e, h, a);

	// right
	add_face(ret, b, d, g);
	add_face(ret, g, e, b);

	// top
	add_face(ret, a, c, d);
	add_face(ret, d, b, a);

	return ret;
}

std::vector<glm::vec3> AABB::to_line_mesh() {
	std::vector<glm::vec3> ret;
	auto add_line = [](std::vector<glm::vec3>& ret, vec3 a, vec3 b) {
		ret.push_back(a); ret.push_back(b);
	};

	vec3 diag = diagonal();
	vec3 x = diag * vec3(1.0f, 0.0f, 0.0f), y = diag * vec3(0.0f, 1.0f, 0.0f), z = diag * vec3(0.0f, 0.0f, 1.0f);

	//		a--b
	//		c--d
	//  h--e
	//  f--g
	vec3 h = p0, e = h + x, f = h + z, g = h + x + z;
	vec3 a = h + y, b = e + y, c = f + y, d = g + y;

	// bottom
	add_line(ret, h, e);
	add_line(ret, e, g);
	add_line(ret, g, f);
	add_line(ret, f, h);

	// vertical
	add_line(ret, h, a);
	add_line(ret, e, b);
	add_line(ret, f, c);
	add_line(ret, g, d);

	// top
	add_line(ret, a, b);
	add_line(ret, b, d);
	add_line(ret, d, c);
	add_line(ret, c, a);

	return ret;
}
