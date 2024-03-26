#include "VoxelGrid.h"

#include <iostream>

VoxelGrid::VoxelGrid(py::array_t<double> bounds_, const double voxelSize) : voxelSize(voxelSize) {

    py::buffer_info bndsBuf = bounds_.request();
    if(bndsBuf.ndim != 2 || bndsBuf.shape[0] != 2 || bndsBuf.shape[1] != 3){
        throw std::runtime_error{"Expected voxel bounds shape 2x3"};
    }
    double* bndsPtr = static_cast<double*>(bndsBuf.ptr);
    
    for(int i = 0; i < 2; i++){
        bounds[i] = {bndsPtr[i * 3], bndsPtr[i * 3 + 1], bndsPtr[i * 3 + 2]};
    }
    dims = ((bounds[1] - bounds[0]) / voxelSize).ceil().convertTo<int32_t>();
    origin = bounds[0];
    bounds[1] = bounds[0] + dims.convertTo<double>() * voxelSize;

    printf("==============================================\n");
    printf("         PYBIND Volumetric map Configs ! \n");
    printf("  Voxel Size : %.4f\n", voxelSize);
    printf("  Voxel Min Bounds : %.3f, %.3f, %.3f\n", bounds[0].x, bounds[0].y, bounds[0].z);
    printf("  Voxel Max Bounds : %.3f, %.3f, %.3f\n", bounds[1].x, bounds[1].y, bounds[1].z);
    printf("  Voxel Dimensions : %d, %d, %d\n", dims.x, dims.y, dims.z);
    printf("         Done constructor\n");
    printf("==============================================\n");

    voxels.resize(dims.x * dims.y * dims.z);
}

void VoxelGrid::integrate(int32_t height, int32_t width,
                          py::array_t<uint8_t> color_, py::array_t<double> depth_,
                          py::array_t<double> intrinsic_, py::array_t<double> pose_) {

    
    uint8_t* color = request_buffer_ptr(color_, {-1, -1, 3});
    double* depth = request_buffer_ptr(depth_, {-1, -1});
    double* intrinsic = request_buffer_ptr(intrinsic_, {3, 3});
    double* pose = request_buffer_ptr(pose_, {4, 4});

    point3d<double> startPoint = {pose[3], pose[7], pose[11]};
    
    std::vector<Voxel> temp(dims.x * dims.y * dims.z);

    #pragma omp parallel for
    for(int i = 0; i < height * width; i++){

        double d = depth[i];

        auto sample = FastVoxelTraversalSample(i, voxelSize, origin, startPoint, height, width, intrinsic, pose, MAX_RANGE);

        while(!sample.is_hit()) {

            if(sample.is_valid(dims)) {
                if(d == 0) {
                    point3d<int32_t> index = sample.get_cur_voxel_index();
                    Voxel& curVoxel = temp[index.x + dims.x * index.y + dims.x * dims.y * index.z];
                    curVoxel.occLogOdd = L_UPDATE_FREE;
                    curVoxel.face[sample.get_cur_face()] = L_UPDATE_FREE;
                }
                else if (sample.get_current_t() > d) {
                    point3d<int32_t> index = sample.get_cur_voxel_index();
                    Voxel& curVoxel = temp[index.x + dims.x * index.y + dims.x * dims.y * index.z];
                    curVoxel.occLogOdd = L_UPDATE_OCCUPIED;
                    curVoxel.face[sample.get_cur_face()] = L_UPDATE_OCCUPIED;
                    break;
                } else {
                    point3d<int32_t> index = sample.get_cur_voxel_index();
                    Voxel& curVoxel = temp[index.x + dims.x * index.y + dims.x * dims.y * index.z];
                    curVoxel.occLogOdd = L_UPDATE_FREE;
                    curVoxel.face[sample.get_cur_face()] = L_UPDATE_FREE;
                }
            }

            sample.one_step();
            if(sample.is_over()) break;
        }
    }
    for(int i = 0; i < temp.size(); i++){
        voxels[i].occLogOdd += temp[i].occLogOdd;
        for(int j = 0; j < 6; j++){
            voxels[i].face[j] += temp[i].face[j];
        }
    }

    printf("Integrating Done\n");
}

py::array_t<double> VoxelGrid::raycasting_face(int32_t height, int32_t width,
                                          py::array_t<double> intrinsic_, py::array_t<double> pose_) const {

    double* intrinsic = request_buffer_ptr(intrinsic_, {3, 3});
    double* pose = request_buffer_ptr(pose_, {4, 4});

    std::vector<double> values(height * width);
    point3d<double> startPoint = {pose[3], pose[7], pose[11]};

    #pragma omp parallel for
    for(int i = 0; i < height * width; i++){

        auto sample = FastVoxelTraversalSample(i, voxelSize, origin, startPoint, height, width, intrinsic, pose, MAX_RANGE);

        while(!sample.is_hit()){
            if(sample.is_valid(dims)){
                point3d<int32_t> index = sample.get_cur_voxel_index();
                Voxel curVoxel = voxels[index.x + dims.x * index.y + dims.x * dims.y * index.z];

                if(curVoxel.face[sample.get_cur_face()] > L_OCCUPIED) {
                    values[i] = sample.get_current_t();
                    break;
                }


            }
            sample.one_step();
            if(sample.is_over()){
                values[i] = 0;
            }
        }
    }
    return vector_to_pyarray(values, {height, width});
}

py::array_t<double> VoxelGrid::raycasting(int32_t height, int32_t width,
                                          py::array_t<double> intrinsic_, py::array_t<double> pose_) const {

    double* intrinsic = request_buffer_ptr(intrinsic_, {3, 3});
    double* pose = request_buffer_ptr(pose_, {4, 4});

    std::vector<double> values(height * width);
    point3d<double> startPoint = {pose[3], pose[7], pose[11]};

    #pragma omp parallel for
    for(int i = 0; i < height * width; i++){

        auto sample = FastVoxelTraversalSample(i, voxelSize, origin, startPoint, height, width, intrinsic, pose, MAX_RANGE);

        while(!sample.is_hit()){
            if(sample.is_valid(dims)){
                point3d<int32_t> index = sample.get_cur_voxel_index();
                Voxel curVoxel = voxels[index.x + dims.x * index.y + dims.x * dims.y * index.z];
                if(curVoxel.occLogOdd > L_OCCUPIED) {
                    values[i] = sample.get_current_t();
                    break;
                }
            }
            sample.one_step();
            if(sample.is_over()){
                values[i] = 0;
            }
        }
    }
    return vector_to_pyarray(values, {height, width});
}

