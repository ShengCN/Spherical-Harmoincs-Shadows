#pragma once
#include <vector>
#include <memory>
#include <unordered_map>

#include "ppc.h"
#include "mesh.h"
#include "shader.h"

class scene {
public:
	scene();
	~scene();

	//------- Shared Functions --------//
	std::shared_ptr<mesh> load_mesh(const std::string mesh_file, std::shared_ptr<shader> render_shader);
	std::shared_ptr<mesh> new_mesh();
	void remove_mesh(int mesh_id);

	virtual void add_mesh(std::shared_ptr<mesh> m);
	virtual void load_scene(std::string scene_file);
	virtual void clean_up();
	virtual vec3 scene_center();
	virtual std::vector<std::shared_ptr<mesh>> get_meshes();
	virtual std::shared_ptr<mesh> get_mesh(int mesh_id);
	virtual AABB scene_aabb();
	virtual bool save_scene(const std::string filename);
	virtual void reset_camera(vec3 &look, vec3 &at);
	virtual void reset_camera(std::shared_ptr<ppc> camera);
	void focus_at(std::shared_ptr<ppc> camera, std::shared_ptr<mesh> m, glm::vec3 relative_vec);
	void stand_on_plane(std::shared_ptr<mesh> m);
	void add_light(std::shared_ptr<mesh> l) { m_lights.push_back(l); }
	std::vector<std::shared_ptr<mesh>> get_lights() { return m_lights; }

	//------- Protected Variables --------//
protected:
	std::vector<std::shared_ptr<mesh>> m_meshes;
	std::vector<std::shared_ptr<mesh>> m_lights;
};

