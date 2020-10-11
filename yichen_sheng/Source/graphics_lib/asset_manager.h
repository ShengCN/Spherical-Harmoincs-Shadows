#pragma once
#include "Render/shader.h"
#include "Render/scene.h"
#include "Render/ppc.h"

// rendering related assets
struct asset_manager {

public:
	asset_manager();

public:
	//------- Variables --------//
	std::shared_ptr<scene> render_scene;
	std::shared_ptr<scene> visualize_scene;

	std::vector<glm::vec3> lights;
	std::unordered_map<std::string, std::shared_ptr<shader>> shaders;
	std::unordered_map<std::shared_ptr<mesh>, std::shared_ptr<shader>> rendering_mappings; // mesh_id -> shader
	std::shared_ptr<ppc> cur_camera=nullptr;

	float shadow_ppc_fov = 30.0f;
};