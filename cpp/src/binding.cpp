#include "common.h"
#include "VoxelGrid.h"


PYBIND11_MODULE(DirectionVoxelGrid, m) {

    py::class_<VoxelGrid>(m, "VoxelGrid")
        .def(py::init<py::array_t<double>, double>())
        .def("integrate", &VoxelGrid::integrate)
        .def("raycasting", &VoxelGrid::raycasting)
        .def("raycasting_face", &VoxelGrid::raycasting_face)
        .def("occlusion_aware", &VoxelGrid::occlusion_aware)
        .def("unobserved_voxel", &VoxelGrid::unobserved_voxel)
        .def("rear_side_voxel", &VoxelGrid::rear_side_voxel)
        .def("rear_side_entropy", &VoxelGrid::rear_side_entropy)
        .def("krigel", &VoxelGrid::krigel)
        .def("occlusion_aware_face", &VoxelGrid::occlusion_aware_face)
        .def("unobserved_voxel_face", &VoxelGrid::unobserved_voxel_face)
        .def("rear_side_voxel_face", &VoxelGrid::rear_side_voxel_face)
        .def("rear_side_entropy_face", &VoxelGrid::rear_side_entropy_face)
        .def("krigel_face", &VoxelGrid::krigel_face);

}