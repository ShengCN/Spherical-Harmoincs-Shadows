#include "voxelization.h"

void voxelizater::voxelize(std::shared_ptr<mesh> m, int steps, std::vector<AABB>& out_voxels) {
	// uniform voxelize the space
	AABB mesh_aabb = m->compute_world_aabb();
	vec3 diag = mesh_aabb.diagonal();
	vec3 unit_diag = diag / (float)steps;

	float min_length = std::min(unit_diag.x, std::min(unit_diag.y, unit_diag.z));
	
	unit_diag = vec3(min_length);
	vec3 x = vec3(diag.x, 0.0f, 0.0f), y = vec3(0.0f, diag.y, 0.0f), z = vec3(0.0f, 0.0f, diag.z);
	mat3 basis = mat3(x, y, z);

	out_voxels.clear();

	vec3 xyz_steps = diag / unit_diag;
	xyz_steps.x = (int)xyz_steps.x;
	xyz_steps.y = (int)xyz_steps.y;
	xyz_steps.z = (int)xyz_steps.z;
	for (int xi = 0; xi < xyz_steps.x; ++xi) {
		float x_fract = (float)xi / xyz_steps.x;
		for (int yi = 0; yi < xyz_steps.y; ++yi) {
			float y_fract = (float)yi / xyz_steps.y;
			for (int zi = 0; zi < xyz_steps.z; ++zi) {
				float z_fract = (float)zi / xyz_steps.z;
				vec3 cell_p0 = mesh_aabb.p0 + basis * vec3(x_fract, y_fract, z_fract);
				out_voxels.push_back(AABB(cell_p0, cell_p0 + unit_diag));
			}
		}
	}
}
