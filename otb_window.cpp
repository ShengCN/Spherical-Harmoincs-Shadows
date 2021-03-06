#include <memory>
#include "otb_window.h"
#include <imgui/imgui.h>
#include <imgui/examples/imgui_impl_glfw.h>
#include <imgui/examples/imgui_impl_opengl3.h>

#include "graphics_lib/Render/shader.h"
#include "graphics_lib/Utilities/Utils.h"
#include "graphics_lib/Utilities/model_loader.h"
#include "graphics_lib/Render/material.h"
#include "spherical_harmonics.h"

using namespace purdue;

otb_window::otb_window() {
	m_band = 4;
	
	m_u_min = 0.29f;
	m_u_max = 0.346f;
	m_v_min = 0.765f;
	m_v_max = 0.84f;
	m_intensity = 11.0f;	
}

otb_window::~otb_window() {

}

render_engine otb_window::m_engine;;

void otb_window::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	char m;
	bool shift = false;
	if(key == GLFW_KEY_W && action == GLFW_PRESS) {
		m = 'w';
	}

	if(key == GLFW_KEY_A && action == GLFW_PRESS) { m = 'a';
	}

	if(key == GLFW_KEY_S && action == GLFW_PRESS) {
		m = 's';
	}

	if(key == GLFW_KEY_D && action == GLFW_PRESS) {
		m = 'd';
	}

	if(key == GLFW_KEY_Q && action == GLFW_PRESS) {
		m = 'q';
	}

	if(key == GLFW_KEY_E && action == GLFW_PRESS) {
		m = 'e';
	}

	if(key == GLFW_KEY_LEFT_SHIFT && action == GLFW_PRESS) {
		shift = true;	
	}

	m_engine.camera_keyboard(m,shift);
}

void otb_window::scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {	
	m_engine.camera_scroll(yoffset);
}

void otb_window::mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
	if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) {
		double xpos, ypos;
		glfwGetCursorPos(window, &xpos, &ypos);

		INFO("Right button clicked, " + std::to_string(xpos) + " " + std::to_string(ypos));
		m_engine.camera_press((int)xpos, (int)ypos);
	}

	if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_RELEASE) {
		double xpos, ypos;
		glfwGetCursorPos(window, &xpos, &ypos);

		INFO("Right button released, " + std::to_string(xpos) + " " + std::to_string(ypos));
		m_engine.camera_release((int)xpos, (int)ypos);
	}
}

void otb_window::cursor_position_callback(GLFWwindow* window,  double xpos, double ypos) {
	// INFO("x: " + std::to_string(xpos) + " y: "+ std::to_string(ypos));
	m_engine.camera_move(xpos, ypos);
}

int otb_window::create_window(int w, int h, const std::string title) {
	/* Initialize the library */
	if (!glfwInit())
		return -1;

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);
	glfwWindowHint(GLFW_DEPTH_BITS, 32);
	glfwWindowHint(GLFW_SAMPLES, 16);

	/* Create a windowed mode window and its OpenGL context */
	_window = glfwCreateWindow(w, h, title.c_str(), NULL, NULL);
	if (!_window) {
		glfwTerminate();
		return -1;
	}

	// callbacks
	glfwSetErrorCallback(error_callback);
	glfwSetKeyCallback(_window, key_callback);
	glfwSetScrollCallback(_window, scroll_callback);
	glfwSetCursorPosCallback(_window, cursor_position_callback);
	glfwSetMouseButtonCallback(_window, mouse_button_callback);
	glfwSetWindowSizeCallback(_window, window_resize_callback);

	// set up environment
	glfwMakeContextCurrent(_window);
	glfwSwapInterval(1);

	if (gladLoadGL() == 0) {
		std::cout << "Failed to initialize OpenGL context" << std::endl;
		return -1;
	}

	printf("OpenGL %d.%d\n", GLVersion.major, GLVersion.minor);
	if (GLVersion.major < 2) {
		printf("Your system doesn't support OpenGL >= 2!\n");
		return -1;
	}

	m_engine.init();
	init_gui();
	init_scene();

	return 1;
}

void otb_window::init_scene() {
	int h, w;
	glfwGetWindowSize(_window, &w, &h);
	m_engine.test_scene(w,h);

	const std::string light_img = "lights/2_0.png";
	m_ibl.read_img(light_img);
}

void otb_window::show() {
	glfwMakeContextCurrent(_window);
	static int iter = 0;

	/* Loop until the user closes the window */
	while (!glfwWindowShouldClose(_window)) {
		glfwPollEvents();
		
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		render(iter++);

		draw_gui();
		glfwSwapBuffers(_window);
	}

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glfwTerminate();
}

void otb_window::save_framebuffer(const std::string output_file) {
	unsigned int *pixels;
	int w = width(), h = height();
	pixels = new unsigned int[w * h * 4];
	for (int i = 0; i < (w * h * 4); i++) {
		pixels[i] = 0;
	}

	glReadPixels(0, 0, width(), height(), GL_RGBA, GL_UNSIGNED_BYTE, pixels);

	// filp pixels
	for (int j = 0; j < h / 2; ++j) for (int i = 0; i < w; ++i) {
		std::swap(pixels[w * j + i], pixels[w * (h-1-j) + i]);
	}

	save_image(output_file, pixels, w, h, 4);
	delete[] pixels;
}

int otb_window::width() {
	int display_w, display_h;
	glfwGetFramebufferSize(_window, &display_w, &display_h);
	return display_w;
}

int otb_window::height() {
	int display_w, display_h;
	glfwGetFramebufferSize(_window, &display_w, &display_h);
	return display_h;
}

void otb_window::init_gui() {
	// imgui
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	ImGui::StyleColorsDark();
	ImGui_ImplGlfw_InitForOpenGL(_window, true);
	const char* glsl_version = "#version 460";
	if(ImGui_ImplOpenGL3_Init(glsl_version)) {
		INFO("ImGui init success");
	} else {
		WARN("ImGui init failed");
	}
}

void otb_window::reload_all_shaders() {
	m_engine.reload_shaders();
}

static bool draw_vertex = true;
void otb_window::draw_gui() {
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	//// ------------------------ Window ------------------------ //
	ImGui::Begin("PC control");
	// ImGui::SliderFloat("fov", &asset_manager::instance).cur_camera->_fov, 30.0f, 120.0f);
	ImGui::SliderFloat("fov", &m_engine.get_render_ppc()->_fov, 5.0f, 120.0f);
	if(ImGui::Button("reload shader")) {
		reload_all_shaders();
	}

	static int point_size = 1;
	ImGui::SliderInt("Point size", &point_size, 1, 100);
	glPointSize(point_size);
	ImGui::SliderInt("band", &m_band, 0, 20);
	ImGui::Checkbox("Draw Vertex", &draw_vertex);
	
	ImGui::SliderFloat("u min", &m_u_min, 0.0f, 1.0f);
	ImGui::SliderFloat("u max", &m_u_max, 0.0f, 1.0f);
	ImGui::SliderFloat("v min", &m_v_min, 0.0f, 1.0f);
	ImGui::SliderFloat("v max", &m_v_max, 0.0f, 1.0f);
	
	ImGui::SliderFloat("Light intensity", &m_intensity, 0.0f, 100.0f);

	if (ImGui::Button("dbg")) {
		int n = 10000;

		cuda_timer clc;
		clc.tic();
		// exp_bands(m_band, n, true);
		gt_render(m_engine.get_rendering_meshes());
		clc.toc();
		INFO("exp time: " + std::to_string(clc.get_time()));
	}
	ImGui::SameLine();
	if(ImGui::Button("save")) {
		pd::safe_create_folder("output");
		char buff[100];
		snprintf(buff, sizeof(buff), "%04d", m_band);
		std::string buff_str = buff;

		std::string out_fname = "output/" + buff_str + ".png";
		save_framebuffer(out_fname);

		// aslo save basis here
		out_fname = "output/" + buff_str + ".bin";
		auto plane_mesh = m_engine.get_rendering_meshes()[0];
		std::fstream oss(out_fname, std::fstream::out | std::fstream::binary);
		if (oss.is_open()) {
			oss.write((char*)&plane_mesh->m_sh_coeffs[0], sizeof(float) * plane_mesh->m_sh_coeffs.size());
			INFO("Writing " + out_fname + " succeed!");
		}
		oss.close();
	}
	ImGui::End();

	 //ImGui::ShowTestWindow();
	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void otb_window::window_resize_callback(GLFWwindow* window, int w, int h) {
	glViewport(0,0,w,h);
	m_engine.camera_resize(w,h);
}

void otb_window::render(int iter) {
	glClearColor(0.0f,0.0f,0.0f,1.0f);
	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

	m_engine.render(iter);
	
	if(draw_vertex) {
		m_engine.set_vis_verts(true);
		m_engine.render(iter);
		m_engine.set_vis_verts(false);
	}
}

void otb_window::gt() {
	auto func = [&](float theta, float phi) {
		float u = phi / (3.1415926f * 2.0f);
		float v = 1.0f - theta / (3.1415926f);
		
		if (u > m_u_min && u < m_u_max && v > m_v_min && v < m_v_max)
			return m_intensity;
		
		return 0.0f;
	};

	auto meshes = m_engine.get_rendering_meshes();
	meshes[0]->set_color(vec3(1.0f));
	meshes[1]->set_color(vec3(0.7f));
}

void otb_window::exp_bands(int band, int n, bool is_shadow) {
	auto func = [&](float theta, float phi) {
		float u = phi / (3.1415926f * 2.0f);
		float v = 1.0f - theta / (3.1415926f);
		
		if (u > m_u_min && u < m_u_max && v > m_v_min && v < m_v_max)
			return m_intensity;
		
		return 0.0f;
	};

	auto meshes = m_engine.get_rendering_meshes();
	meshes[0]->set_color(vec3(1.0f));
	meshes[1]->set_color(vec3(0.7f));

	meshes[0]->m_band = band;
	meshes[0]->m_sh_coeffs.resize(meshes[0]->m_verts.size() * band * band);
	char buff[100];
	snprintf(buff, sizeof(buff), "%04d", band);
	std::string buff_str = buff;
	std::string sh_weight_fname = "output/" + buff_str + ".bin";
	std::fstream iss(sh_weight_fname, std::fstream::in | std::fstream::binary);
	if (iss.is_open()) {
		iss.read((char*)&meshes[0]->m_sh_coeffs[0], sizeof(float) * meshes[0]->m_sh_coeffs.size());
		INFO("Reading from " + sh_weight_fname + " succeed");
	}

	// diffuse
	cuda_compute_sh_coeff(meshes[1], band, n);

	auto sp_map_coeff = SH_func(func, m_band, n);
	for(auto f:sp_map_coeff) {
		INFO("light coeff: " + std::to_string(f));
	}

	sh_render(meshes, sp_map_coeff);
}