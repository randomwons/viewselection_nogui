#ifndef __RAY_CASTING_H__
#define __RAY_CASTING_H__

#include <limits>
#include "mathutils.h"

class FastVoxelTraversalSample {
public:
    FastVoxelTraversalSample(
        const int index,
        const double voxel_size,
        const point3d<double>& voxel_origin,
        const point3d<double>& start_point, 
        const int32_t height, const int32_t width, 
        const double* intrinsic,
        const double* pose,
        const double range) : max_range(range) {

        cur_voxel_index = ((start_point - voxel_origin) / voxel_size).floor().convertTo<int32_t>();

        double xp = (index % width - intrinsic[2]) / intrinsic[0];
        double yp = (index / width - intrinsic[5]) / intrinsic[4];

        point3d<double> ray_camera = {xp, yp, 1.0};

        point3d<double> ray_dir = {
            pose[0] * ray_camera.x + pose[1] * ray_camera.y + pose[2],
            pose[4] * ray_camera.x + pose[5] * ray_camera.y + pose[6],
            pose[8] * ray_camera.x + pose[9] * ray_camera.y + pose[10]
        };

        point3d<double> end_point = start_point + ray_dir * range;
        end_voxel_index = ((end_point - voxel_origin) / voxel_size).floor().convertTo<int32_t>();

        step = ray_dir.sign();
        
        point3d<double> next_voxel_bnds = (cur_voxel_index + (step + 1) / 2).convertTo<double>() * voxel_size + voxel_origin;

        t_max = {
            ray_dir.x != 0 ? (next_voxel_bnds.x - start_point.x) / ray_dir.x : std::numeric_limits<double>::infinity(),
            ray_dir.y != 0 ? (next_voxel_bnds.y - start_point.y) / ray_dir.y : std::numeric_limits<double>::infinity(),
            ray_dir.z != 0 ? (next_voxel_bnds.z - start_point.z) / ray_dir.z : std::numeric_limits<double>::infinity()
        };

        t_delta = {
            ray_dir.x != 0 ? voxel_size / ray_dir.x * step.x : 0,
            ray_dir.y != 0 ? voxel_size / ray_dir.y * step.y : 0,
            ray_dir.z != 0 ? voxel_size / ray_dir.z * step.z : 0  
        };

    }

    bool is_hit() const {
        return cur_voxel_index == end_voxel_index;
    }

    bool is_valid(const point3d<int32_t>& bounds) const {
        return (cur_voxel_index.x >=0 && cur_voxel_index.y >= 0 && cur_voxel_index.z >= 0 &&
                cur_voxel_index.x < bounds.x && cur_voxel_index.y < bounds.y && cur_voxel_index.z < bounds.z);
    }


    void one_step() {
        if (t_max.x < t_max.y && t_max.x < t_max.z) {
            t_current = t_max.x;
            cur_voxel_index.x += step.x;
            cur_voxel_face = step.x >= 0 ? 2 : 0;
            t_max.x += t_delta.x;
        } else if (t_max.y < t_max.z) {
            t_current = t_max.y;
            cur_voxel_index.y += step.y;
            cur_voxel_face = step.y >= 0 ? 5 : 4;
            t_max.y += t_delta.y;
        } else {
            t_current = t_max.z;
            cur_voxel_index.z += step.z;
            cur_voxel_face = step.z >= 0 ? 1 : 3;
            t_max.z += t_delta.z;
        }
        n_travesed++;
    }

    bool is_over() const {
        return t_current > max_range;
    }

    double get_current_t() const {
        return t_current;
    }
    
    int32_t get_n_traversed() const {
        return n_travesed;
    } 

    int32_t get_cur_face() const {
        return cur_voxel_face;
    }

    point3d<int32_t> get_cur_voxel_index() const {
        return cur_voxel_index;
    }

private:
    point3d<int32_t> cur_voxel_index;
    point3d<int32_t> end_voxel_index;
    int32_t cur_voxel_face;

    int32_t n_travesed = 0;
    double t_current = 0;
    double max_range;
    point3d<double> t_max;
    point3d<double> t_delta;
    point3d<int32_t> step;
    
};

#endif // __RAY_CASTING_H__