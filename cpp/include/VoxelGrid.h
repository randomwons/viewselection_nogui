#ifndef __VOXELGRID_H__
#define __VOXELGRID_H__

#include "common.h"
#include "fastvoxeltraversal.h"

struct Voxel {

    double occLogOdd { 0 };
    int32_t face[6] { 0 };

};

class VoxelGrid {
public:
    VoxelGrid(py::array_t<double> bounds_, const double voxelSize);

    void integrate(int32_t height, int32_t width,
                   py::array_t<uint8_t> color, py::array_t<double> depth,
                   py::array_t<double> intrinsic, py::array_t<double> pose);

    py::array_t<double> raycasting(int32_t height, int32_t width,
                                   py::array_t<double> intrinsic, py::array_t<double> pose) const;
    py::array_t<double> raycasting_face(int32_t height, int32_t width,
                                        py::array_t<double> intrinsic, py::array_t<double> pose) const;                                   


    // 기존 POLICIES
    py::array_t<double> occlusion_aware(int32_t height, int32_t width,
                                        py::array_t<double> intrinsic, py::array_t<double> pose) const;
    py::array_t<double> unobserved_voxel(int32_t height, int32_t width,
                                        py::array_t<double> intrinsic, py::array_t<double> pose) const;                                        
    py::array_t<double> rear_side_voxel(int32_t height, int32_t width,
                                        py::array_t<double> intrinsic, py::array_t<double> pose) const;
    py::array_t<double> rear_side_entropy(int32_t height, int32_t width,
                                        py::array_t<double> intrinsic, py::array_t<double> pose) const;
    py::array_t<double> proximity_count(int32_t height, int32_t width,
                                        py::array_t<double> intrinsic, py::array_t<double> pose) const;
    py::array_t<double> krigel(int32_t height, int32_t width,
                                        py::array_t<double> intrinsic, py::array_t<double> pose) const;


    // FACE 적용 POLICIES
    py::array_t<double> occlusion_aware_face(int32_t height, int32_t width,
                                             py::array_t<double> intrinsic, py::array_t<double> pose) const;
    py::array_t<double> unobserved_voxel_face(int32_t height, int32_t width,
                                        py::array_t<double> intrinsic, py::array_t<double> pose) const;                                        
    py::array_t<double> rear_side_voxel_face(int32_t height, int32_t width,
                                        py::array_t<double> intrinsic, py::array_t<double> pose) const;
    py::array_t<double> rear_side_entropy_face(int32_t height, int32_t width,
                                        py::array_t<double> intrinsic, py::array_t<double> pose) const;
    py::array_t<double> proximity_count_face(int32_t height, int32_t width,
                                        py::array_t<double> intrinsic, py::array_t<double> pose) const;
    py::array_t<double> krigel_face(int32_t height, int32_t width,
                                        py::array_t<double> intrinsic, py::array_t<double> pose) const;                                             

private:
    double voxelSize;
    std::array<point3d<double>, 2> bounds;
    point3d<int32_t> dims;
    point3d<double> origin;
    std::vector<Voxel> voxels;

};


#endif // __VOXELGRID_H__