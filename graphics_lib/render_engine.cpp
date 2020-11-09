#include <glad/glad.h>
#include "render_engine.h"
#include "Utilities/voxelization.h"
#include <glm/gtx/transform.hpp>
#include "Utilities/Utils.h"
#include "Render/material.h"

render_engine::render_engine() {
	m_draw_render = true;
	m_draw_visualize = false;
	m_vis_frame_mode = m_vis_verts_mode = false;
	m_current_draw_type = draw_type::triangle;
}

void render_engine::test_scene(int w, int h) {
	auto plane_mesh = add_plane();
	vec3 plane_center = plane_mesh->compute_world_center();	

	// int id = load_mesh("Meshes/cube.obj");
	// INFO("finish loading mesh cube");

	// auto mesh_ptr = get_mesh(id);
	// mesh_ptr->normalize_position_orientation();
	// set_mesh_color(mesh_ptr, vec3(0.6f));
	// recompute_normal(id);
	// stand_on_plane(id, plane_center, vec3(0.0f,1.0f,0.0f));
	// mesh_ptr->add_world_transate(vec3(-0.75f,0.01f,0.0f));

	int id = load_mesh("Meshes/bunny_s.obj");
	INFO("finish loading mesh cube");

	auto mesh_ptr = get_mesh(id);
	mesh_ptr->normalize_position_orientation();
	set_mesh_color(mesh_ptr, vec3(0.6f));
	recompute_normal(id);
	stand_on_plane(id, plane_center, vec3(0.0f,1.0f,0.0f));
	mesh_ptr->add_world_transate(vec3(0.0f,0.01f,0.0f));

	cur_manager.cur_camera = std::make_shared<ppc>(w, h, 50.0f);
	look_at(id);
}

void render_engine::init_camera(int w, int h, float fov) {
	cur_manager.cur_camera = std::make_shared<ppc>(w, h, fov);
}

void render_engine::recompute_normal(int mesh_id) {
	get_mesh(mesh_id)->recompute_normal();
}

void render_engine::stand_on_plane(int mesh_id, vec3 p, vec3 n) {
	auto mesh_ptr = get_mesh(mesh_id);
	if(mesh_ptr == nullptr) return;

	AABB world_aabb = mesh_ptr->compute_world_aabb();
	vec3 normalized_n = glm::normalize(n);
	glm::vec3 trans_vec = -glm::dot(world_aabb.p0 - p, normalized_n) * normalized_n;
	
	mesh_ptr->add_world_transate(trans_vec); 
}

void render_engine::set_cur_render_type(draw_type type) {
	m_current_draw_type = type;
}

std::shared_ptr<mesh> render_engine::add_plane() {
	std::shared_ptr<mesh> mesh_ptr = std::make_shared<mesh>();
	int n = 50;
	for(int i = 0; i < n; ++i)
		for(int j = 0; j < n; ++j) {
			vec3 offset(2.0f * i,0.0f, 2.0f * j);
			mesh_ptr->add_face(offset + vec3(-1.0f,0.0f,-1.0f), offset + vec3(-1.0f,0.0f,1.0f), offset + vec3(1.0f,0.0f,1.0f));
			mesh_ptr->add_face(offset + vec3(1.0f,0.0f,1.0f), offset + vec3(1.0f,0.0f,-1.0f), offset + vec3(-1.0f,0.0f,-1.0f));
		}
	
	mesh_ptr->set_color(vec3(1.0f));
	mesh_ptr->normalize_position_orientation(vec3(5.0f,1.0f,5.0f));
	// mesh_ptr->add_world_transate(vec3(0.0f,-0.25f,0.0f));
	set_shader(mesh_ptr, "template");
	add_mesh(mesh_ptr);
	return mesh_ptr;
}

void render_engine::render(int frame) {	
	rendering_params params = get_cur_rendering_params(frame);

	if (m_draw_render) {
		render_scene(cur_manager.render_scene, params);
	}
	if (m_draw_visualize) {
		params.dtype = draw_type::line_segments;
		if (m_vis_frame_mode) glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		render_scene(cur_manager.visualize_scene, params);
		if (m_vis_frame_mode) glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	}
	if (m_vis_verts_mode) {
		params.dtype = draw_type::points;
		render_scene(cur_manager.render_scene, params);
	}
}

rendering_params render_engine::get_cur_rendering_params(int frame) {
	rendering_params params;
	params.cur_camera = cur_manager.cur_camera;
	params.p_lights =  cur_manager.lights;
	params.frame = frame;
	params.dtype = m_current_draw_type;
	return params;
}

void render_engine::init() {
	//------- initialize rendering states --------//
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_MULTISAMPLE);

	//------- initialize Shaders --------//
	const std::string template_vs = "Shaders/template_vs.glsl";
	const std::string template_fs = "Shaders/template_fs.glsl";
	cur_manager.shaders["template"] = std::make_shared<shader>(template_vs.c_str(), template_fs.c_str());

	INFO("shader finished");

	// const std::string weighted_OIT_vs = "Shaders/transparent_vs.glsl";
	// const std::string weighted_OIT_fs = "Shaders/transparent_fs.glsl";
	// cur_manager.shaders["weight_OIT"] = std::make_shared<shader>(weighted_OIT_vs.c_str(), weighted_OIT_fs.c_str());

	// const std::string quad_vs = "Shaders/quad_vs.glsl";
	// const std::string quad_fs = "Shaders/quad_fs.glsl";
	// cur_manager.shaders["quad"] = std::make_shared<shader>(quad_vs.c_str(), quad_fs.c_str());

	//------- initialize scene --------//
	cur_manager.render_scene = std::make_shared<scene>();
	cur_manager.visualize_scene = std::make_shared<scene>();
	// m_quad_vao = create_quad();
}

void render_engine::render_scene(std::shared_ptr<scene> cur_scene, rendering_params params) {
	// shader render mesh
	auto meshes = cur_scene->get_meshes();

	for(auto &m:meshes) {
		if (params.dtype == draw_type::points) {
			auto old_color = m->m_colors;
			m->set_color(vec3(1.0f, 0.0f, 0.0f));
			cur_manager.rendering_mappings.at(m)->draw_mesh(m, params);
			m->m_colors = old_color;
			continue;
		}
		cur_manager.rendering_mappings.at(m)->draw_mesh(m, params);
	}
}

void render_engine::render_weighted_OIT(std::shared_ptr<scene> cur_scene, rendering_params params) {
	static unsigned int framebuffer = -1;
	static unsigned int accum_texture = -1;
	static unsigned int reveal_texture = -1;
	static unsigned int rbo = -1;
	
	// initialize
	if (framebuffer == -1) {
		glGenFramebuffers(1, &framebuffer);
		glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

		glGenTextures(1, &accum_texture);
		glBindTexture(GL_TEXTURE_2D, accum_texture);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, cur_manager.cur_camera->_width, cur_manager.cur_camera->_height, 0, GL_RGBA, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glBindTexture(GL_TEXTURE_2D, 0);

		glGenTextures(1, &reveal_texture);
		glBindTexture(GL_TEXTURE_2D, reveal_texture);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_R16F, cur_manager.cur_camera->_width, cur_manager.cur_camera->_height, 0, GL_RED, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glBindTexture(GL_TEXTURE_2D, 0);

		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, accum_texture, 0);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, reveal_texture, 0);
		
		if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
			INFO("ERROR::FRAMEBUFFER:: Framebuffer is not complete!");

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	// ------------------------ pass 1, render opaque ------------------------ //
	// todo

	// ------------------------ pass 2, render transparent ------------------------ //
	// clear initial values
	// render transparent
	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
	glDepthMask(GL_FALSE);

	glDrawBuffer(GL_COLOR_ATTACHMENT0); 
	glClearColor(0.0f,0.0f,0.0f,0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glDrawBuffer(GL_COLOR_ATTACHMENT1);
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	GLenum DrawBuffers[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
	glDrawBuffers(2, DrawBuffers);
	glEnable(GL_BLEND);
	glBlendFunci(0, GL_ONE, GL_ONE);
	//glBlendFunci(1, GL_ZERO, GL_ONE_MINUS_SRC_ALPHA);
	glBlendFunci(1, GL_ZERO, GL_ONE_MINUS_SRC_COLOR);

	render_scene(cur_scene, params);
	glDepthMask(GL_TRUE);
	
	// ------------------------ pass 3, merge results ------------------------ //
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glDrawBuffer(GL_BACK);

	cur_manager.shaders.at("quad")->bind();
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, accum_texture);
	glUniform1i(glGetUniformLocation(cur_manager.shaders.at("quad")->get_shader_program(), "accum_tex"), 0);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, reveal_texture);
	glUniform1i(glGetUniformLocation(cur_manager.shaders.at("quad")->get_shader_program(), "weight_tex"), 1);

	glViewport(0, 0, cur_manager.cur_camera->_width, cur_manager.cur_camera->_height);
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glBlendFunc(GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA);

	draw_quad();
}

std::shared_ptr<mesh> render_engine::vis_new_mesh() {
	auto ret_mesh = cur_manager.visualize_scene->new_mesh();
	cur_manager.rendering_mappings[ret_mesh] = cur_manager.shaders.at("template");
	return ret_mesh;
}

GLuint render_engine::create_quad() {
	GLuint vao, vbo;
	//quad is in z=0 plane, and goes from -1.0 to +1.0 in x,y directions.
	const float quad_verts[] = { -1.0f, -1.0f, 0.0f, 1.0f, -1.0f, 0.0f, -1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f };

	//generate vao id to hold the mapping from attrib variables in shader to memory locations in vbo
	glGenVertexArrays(1, &vao);
	//binding vao means that bindbuffer, enablevertexattribarray and vertexattribpointer 
	// state will be remembered by vao
	glBindVertexArray(vao);

	glGenBuffers(1, &vbo); // Generate vbo to hold vertex attributes for triangle
	glBindBuffer(GL_ARRAY_BUFFER, vbo); //specify the buffer where vertex attribute data is stored
	//upload from main memory to gpu memory
	glBufferData(GL_ARRAY_BUFFER, sizeof(quad_verts), &quad_verts[0], GL_STATIC_DRAW);

	//get a reference to an attrib variable name in a shader
	const GLint pos_loc = 0;
	glEnableVertexAttribArray(pos_loc); //enable this attribute

	//tell opengl how to get the attribute values out of the vbo (stride and offset)
	glVertexAttribPointer(pos_loc, 3, GL_FLOAT, false, 0, 0);
	glBindVertexArray(0); //unbind the vao

	return vao;
}

std::vector<std::shared_ptr<mesh>> render_engine::get_rendering_meshes() {
	return cur_manager.render_scene->get_meshes();
}

std::shared_ptr<mesh> render_engine::get_mesh(int id) {
	return cur_manager.render_scene->get_mesh(id);
}

std::vector<std::shared_ptr<mesh>> render_engine::get_meshes(std::vector<int> ids) {
	std::vector<std::shared_ptr<mesh>> ret;
	for(int id : ids) {
		ret.push_back(get_mesh(id));
	}
	return ret;
}

int render_engine::load_mesh(const std::string model_fname, std::string shader_name) {
	auto cur_mesh = cur_manager.render_scene->load_mesh(model_fname, cur_manager.shaders[shader_name]);
	if (cur_mesh) {
		cur_manager.rendering_mappings[cur_mesh] = cur_manager.shaders[shader_name];
		return cur_mesh->get_id();
	}

	return -1;
}

int render_engine::load_mesh(const std::string model_fname, vec3 c, std::string shader_name) {
	auto cur_mesh = cur_manager.render_scene->load_mesh(model_fname, cur_manager.shaders[shader_name]);

	if (cur_mesh) {
		cur_manager.rendering_mappings[cur_mesh] = cur_manager.shaders[shader_name];

		cur_mesh->set_color(c);
		return cur_mesh->get_id();
	}

	return -1;
}

bool render_engine::save_mesh(std::shared_ptr<mesh> m, const std::string model_fname) {
	return false;
}

bool render_engine::load_render_scene(const std::string scene_file) {
	return false;
}

bool render_engine::reload_shaders() {
	bool success = true;
	for(auto &s:cur_manager.shaders) {
		success = success && s.second->reload_shader();
	}
	return success;
}

void render_engine::add_mesh(std::shared_ptr<mesh> m) {
	cur_manager.render_scene->add_mesh(m);
}

void render_engine::camera_press(int x, int y) {
	cur_manager.cur_camera->mouse_press(x, y);
}

void render_engine::camera_release(int x, int y) {
	cur_manager.cur_camera->mouse_release(x, y);
}

void render_engine::camera_move(int x, int y) {
	cur_manager.cur_camera->mouse_move(x, y);
}

void render_engine::camera_scroll(int offset) {
	cur_manager.cur_camera->scroll((double)offset);
}

void render_engine::camera_keyboard(char m, bool shift) {
	float speed = 1.0f * 0.1f;
	if(shift) 
		speed *= 10.0f;
	
	switch (m)
	{
	case 'w': 
		cur_manager.cur_camera->Keyboard(CameraMovement::forward, speed);
		break;
	case 'a': 
		cur_manager.cur_camera->Keyboard(CameraMovement::left, speed);
		break;
	case 's': 
		cur_manager.cur_camera->Keyboard(CameraMovement::backward, speed);
		break;
	case 'd': 
		cur_manager.cur_camera->Keyboard(CameraMovement::right, speed);
		break;
	case 'q': 
		cur_manager.cur_camera->Keyboard(CameraMovement::up, speed);
		break;
	case 'e': 
		cur_manager.cur_camera->Keyboard(CameraMovement::down, speed);
		break;
	default:
		break;
	}
}

void render_engine::camera_resize(int w, int h) {
	cur_manager.cur_camera->camera_resize(w,h);
}

void render_engine::set_trackball(bool trackball) {
	cur_manager.cur_camera->set_trackball(trackball);
}

void render_engine::look_at(int mesh_id, vec3 relative) {
	auto at_mesh = get_mesh(mesh_id);
	if(at_mesh) {
		cur_manager.render_scene->focus_at(cur_manager.cur_camera, at_mesh, relative);
	} else {
		WARN("Cannot find mesh " + std::to_string(mesh_id));
	}
}

void render_engine::norm_render_scene() {
	//#todo_normalize_render_scene
}

void render_engine::draw_line(glm::vec3 t, glm::vec3 h, vec3 tc, vec3 hc) {
	std::shared_ptr<mesh> line_mesh = std::make_shared<mesh>();
	line_mesh->add_vertex(t, vec3(0.0f), tc);
	line_mesh->add_vertex(h, vec3(0.0f), hc);
	
	rendering_params params = get_cur_rendering_params(0);
	params.dtype = draw_type::line_segments;

	glDisable(GL_DEPTH_TEST);
	cur_manager.shaders.at("template") ->draw_mesh(line_mesh, params);
	glEnable(GL_DEPTH_TEST);

	mesh::id--;
}

void render_engine::set_mesh_color(std::shared_ptr<mesh> m, vec3 c) {
	if(m)
		m->set_color(c);
}

void render_engine::mesh_add_transform(std::shared_ptr<mesh> m, glm::mat4 mat) {
	m->m_world = mat * m->m_world;
}

void render_engine::mesh_add_transform(int id, glm::mat4 mat) {
	std::shared_ptr<mesh> m = get_mesh(id);
	if(m) {
		mesh_add_transform(m, mat);
	}
}

void render_engine::mesh_set_transform(std::shared_ptr<mesh> m, glm::mat4 mat) {
	m->set_world_mat(mat);
}

void render_engine::mesh_apply_transform(std::shared_ptr<mesh> m, glm::mat3 mat) {
	if(m == nullptr) {
		return;
	}

	vec3 trans = mat[0];
	vec3 rotate = mat[1];
	vec3 scale = mat[2];

	glm::mat4 new_scale = glm::scale(scale);
	glm::mat4 new_rotation = glm::rotate(pd::deg2rad(rotate[0]), glm::vec3(1.0f, 0.0f, 0.0f)) *
		glm::rotate(pd::deg2rad(rotate[1]), glm::vec3(0.0f, 1.0f, 0.0f)) *
		glm::rotate(pd::deg2rad(rotate[2]), glm::vec3(0.0f, 0.0f, 1.0f));
	glm::mat4 new_translate = glm::translate(trans);

	glm::mat4 new_transform = new_translate * new_rotation * new_scale;
	m->set_world_mat(new_transform);
}

glm::mat4 render_engine::get_mesh_world(std::shared_ptr<mesh> m) {
	return m->get_world_mat();
}

void render_engine::cut_mesh(std::shared_ptr<mesh> m, vec3 p, vec3 n) {
}

void render_engine::add_point_light(glm::vec3 lp) {
	cur_manager.lights.push_back(lp);
}

void render_engine::set_render_camera(int w, int h, float fov) {
	if (cur_manager.cur_camera == nullptr) {
		cur_manager.cur_camera = std::make_shared<ppc>(w, h, fov);
		return;
	} 
	cur_manager.cur_camera->_width = w;
	cur_manager.cur_camera->_height = h;
	cur_manager.cur_camera->_fov = fov;
}

void render_engine::set_shader(std::shared_ptr<mesh> m, const std::string shader_name) {
	cur_manager.rendering_mappings[m] = cur_manager.shaders.at(shader_name);
}

void render_engine::remove_mesh(int mesh_id) {
	cur_manager.render_scene->remove_mesh(mesh_id);
}

void render_engine::draw_visualize_voxels(std::vector<AABB> voxels) {
	auto vis_mesh = vis_new_mesh();
	if (vis_mesh) {
		for(auto& cur_bb : voxels) {
			auto tri_meshes = cur_bb.to_tri_mesh();
			vis_mesh->add_vertices(tri_meshes);
		}
	}
	vis_mesh->set_color(vec3(0.0f, 0.0f, 0.8f));
}

void render_engine::voxel_vis(int mesh_id) {
	auto mesh_ptr = get_mesh(mesh_id);
	if (mesh_ptr) {
		std::vector<AABB> voxels; voxelizater::voxelize(mesh_ptr, 10, voxels);
		draw_visualize_voxels(voxels);
	}
}

std::shared_ptr<ppc> render_engine::get_render_ppc() {
	return cur_manager.cur_camera;
}

std::shared_ptr<scene> render_engine::get_cur_scene() {
	return cur_manager.render_scene;
}

void render_engine::draw_visualize_line(glm::vec3 t, glm::vec3 h) {
}


void render_engine::draw_quad() {
	glBindVertexArray(m_quad_vao);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
}

void render_engine::clear_visualize() {

}
