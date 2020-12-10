#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include "common.h"

/*
 *
 *	Planer Pinhole camera.
 *	Provide basic controls.
 *
 */
enum class CameraMovement
{
	forward,
	backward,
	left,
	right,
	up,
	down
};

class ppc
{
public:
	glm::vec3 _position;
	glm::vec3 _front;
	glm::vec3 _up;	// default is (0.0f,1.0f,0.0f)
	float _fov, _near, _far;
	int _width, _height;

	ppc(int w, int h, float fov, float p_near=0.001f, float p_far=100000.0f);
	~ppc();

	bool save(const std::string file) {
		std::ofstream output(file, std::ofstream::binary);
		if (output.is_open()) {
			output.write((char*)&_position[0], sizeof(glm::vec3));
			output.write((char*)&_front[0], sizeof(glm::vec3));
			output.write((char*)&_up[0], sizeof(glm::vec3));
			output.write((char*)&_fov, sizeof(float));
			output.write((char*)&_near, sizeof(float));
			output.write((char*)&_far, sizeof(float));
			output.write((char*)&_width, sizeof(int));
			output.write((char*)&_height, sizeof(int));
			output.close();
			return true;
		}
		else {
			std::cout << "File " << file << " cannot save.\n";
			return false;
		}
	}

	bool load(const std::string file) {
		std::ifstream input(file, std::ifstream::binary);
		if (input.is_open()) {
			input.read((char*)&_position[0], sizeof(glm::vec3));
			input.read((char*)&_front[0], sizeof(glm::vec3));
			input.read((char*)&_up[0], sizeof(glm::vec3));
			input.read((char*)&_fov, sizeof(float));
			input.read((char*)&_near, sizeof(float));
			input.read((char*)&_far, sizeof(float));
			input.read((char*)&_width, sizeof(int));
			input.read((char*)&_height, sizeof(int));
			input.close();
			return true;
		}
		else {
			std::cout << "File " << file << " cannot save.\n";
			return false;
		}
	}

	vec3 GetRight();
	vec3 GetUp();
	vec3 GetViewVec() { return _front; }
	void PositionAndOrient(vec3 p, vec3 lookatP, vec3 up);
	glm::mat4 GetP();
	glm::mat4 GetV();
	void Rotate_Axis(glm::vec3 O, glm::vec3 axis, float angled);	// only rotate ppc position
	void Zoom(float delta);
	void Keyboard(CameraMovement cm, float speed); // keyboard reactions
	void scroll(double delta) {
		_fov += float(delta);
	}

	void pan(double deg);
	void tilt(double deg);
	void pitch(double deg);
	void mouse_press(int x, int y);
	void mouse_release(int x, int y);
	void mouse_move(int x, int y);
	bool mouse_inside_frame(int x, int y);
	void set_trackball(bool trackball);
	void camera_resize(int w, int h);

	std::string to_string();

private:
	float GetFocal();

private:
	int m_last_x, m_last_y;
	vec3 m_last_orientation, m_last_position;
	bool m_pressed;
	bool m_trackball;
};