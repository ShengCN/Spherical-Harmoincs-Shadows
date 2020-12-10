#pragma once
#include <memory>
#include "Render/mesh.h"

class voxelizater {
public:
	voxelizater() = default;
	static void voxelize(std::shared_ptr<mesh> m,
				  int steps,
				  std::vector<AABB>& out_voxels);
	
private:
};