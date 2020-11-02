#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <string>
#include <iostream>
#include <graphics_lib/render_engine.h>
#include "spherical_harmonics.h"

class otb_window {
	// public variables
public:
	GLFWwindow* _window;

public:
	otb_window();
	~otb_window();

	// callback functions
public:
	static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
	//static void mouse_callback()
	static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

	static void error_callback(int error, const char* description) {
		std::cerr << "Error: %s\n" << description << std::endl;
	}

	static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
	static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
	static void window_resize_callback(GLFWwindow* window, int w, int h);

	// public functions
public:
	int create_window(int w, int h, const std::string title);
	
	void init_scene();
	void show();		// one thread one window
	void save_framebuffer(const std::string output_file);
	int width();
	int height();

private:
	void init_gui();
	void draw_gui();

private:
	void reload_all_shaders();
	void render(int iter);
	
private:
	static render_engine m_engine;
	float m_distance = 2.0f;
	int m_band;
};

