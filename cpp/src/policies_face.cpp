#include "VoxelGrid.h"

py::array_t<double> VoxelGrid::occlusion_aware_face(int32_t height, int32_t width,
                                                    py::array_t<double> intrinsic_, py::array_t<double> pose_) const {


    double* intrinsic = request_buffer_ptr(intrinsic_, {3, 3});
    double* pose = request_buffer_ptr(pose_, {4, 4});

    std::vector<double> values(height * width);
    point3d<double> startPoint = {pose[3], pose[7], pose[11]};

    #pragma omp parallel for
    for(int i = 0; i < height * width; i++){

        auto sample = FastVoxelTraversalSample(i, voxelSize, origin, startPoint, height, width, intrinsic, pose, MAX_RANGE);

        double total_Iv = 0;
        double Pv_xi = 1;

        while(!sample.is_hit()){
            if(sample.is_valid(dims)){
                point3d<int32_t> index = sample.get_cur_voxel_index();
                Voxel curVoxel = voxels[index.x + dims.x * index.y + dims.x * dims.y * index.z];
                double Po_xi = probability(curVoxel.face[sample.get_cur_face()]);
                double Io_xi = entropy(Po_xi);
                double Iv_xi = Io_xi * Pv_xi;
                total_Iv += Iv_xi;

                if(curVoxel.face[sample.get_cur_face()] >= L_OCCUPIED) break;;

                Pv_xi *= (1 - Po_xi);
            }
            sample.one_step();
            if(sample.is_over()) break;
        }
        values[i] = total_Iv;
    }
    return vector_to_pyarray(values, {height, width});
}

py::array_t<double> VoxelGrid::unobserved_voxel_face(int32_t height, int32_t width,
                                               py::array_t<double> intrinsic_, py::array_t<double> pose_) const {


    double* intrinsic = request_buffer_ptr(intrinsic_, {3, 3});
    double* pose = request_buffer_ptr(pose_, {4, 4});

    std::vector<double> values(height * width);
    point3d<double> startPoint = {pose[3], pose[7], pose[11]};

    #pragma omp parallel for
    for(int i = 0; i < height * width; i++){

        auto sample = FastVoxelTraversalSample(i, voxelSize, origin, startPoint, height, width, intrinsic, pose, MAX_RANGE);

        double total_Ik = 0;
        double Pv_xi = 1;

        while(!sample.is_hit()){
            if(sample.is_valid(dims)){
                point3d<int32_t> index = sample.get_cur_voxel_index();
                Voxel curVoxel = voxels[index.x + dims.x * index.y + dims.x * dims.y * index.z];
                double L = curVoxel.face[sample.get_cur_face()];

                double Iu_xi = (L > L_FREE && L < L_OCCUPIED) ? 1 : 0;
                double Po_xi = probability(L);
                double Io_xi = entropy(Po_xi);
                double Iv_xi = Io_xi * Pv_xi;
                double Ik_xi = Iu_xi * Iv_xi;
                total_Ik += Ik_xi;

                if(L >= L_OCCUPIED) break;;

                Pv_xi *= (1 - Po_xi);
            }
            sample.one_step();
            if(sample.is_over()) break;
        }
        values[i] = total_Ik;
    }
    return vector_to_pyarray(values, {height, width});
}

py::array_t<double> VoxelGrid::rear_side_voxel_face(int32_t height, int32_t width,
                                               py::array_t<double> intrinsic_, py::array_t<double> pose_) const {


    double* intrinsic = request_buffer_ptr(intrinsic_, {3, 3});
    double* pose = request_buffer_ptr(pose_, {4, 4});

    std::vector<double> values(height * width);
    point3d<double> startPoint = {pose[3], pose[7], pose[11]};

    #pragma omp parallel for
    for(int i = 0; i < height * width; i++){

        auto sample = FastVoxelTraversalSample(i, voxelSize, origin, startPoint, height, width, intrinsic, pose, MAX_RANGE);

        double total_Ib = 0;
        double previous_L = L_FREE;

        while(!sample.is_hit()){
            if(sample.is_valid(dims)){
                point3d<int32_t> index = sample.get_cur_voxel_index();
                Voxel curVoxel = voxels[index.x + dims.x * index.y + dims.x * dims.y * index.z];
                double L = curVoxel.face[sample.get_cur_face()];

                if(L >= L_OCCUPIED){
                    if(previous_L > L_FREE && previous_L < L_OCCUPIED) total_Ib++;
                    break;
                }
;               previous_L = L;

            }
            sample.one_step();
            if(sample.is_over()) break;
        }
        values[i] = total_Ib;
    }
    return vector_to_pyarray(values, {height, width});
}

py::array_t<double> VoxelGrid::rear_side_entropy_face(int32_t height, int32_t width,
                                               py::array_t<double> intrinsic_, py::array_t<double> pose_) const {


    double* intrinsic = request_buffer_ptr(intrinsic_, {3, 3});
    double* pose = request_buffer_ptr(pose_, {4, 4});

    std::vector<double> values(height * width);
    point3d<double> startPoint = {pose[3], pose[7], pose[11]};

    #pragma omp parallel for
    for(int i = 0; i < height * width; i++){

        auto sample = FastVoxelTraversalSample(i, voxelSize, origin, startPoint, height, width, intrinsic, pose, MAX_RANGE);

        double total_In = 0;
        double previous_L = L_FREE;
        double Pv_xi = 1;

        while(!sample.is_hit()){
            if(sample.is_valid(dims)){
                point3d<int32_t> index = sample.get_cur_voxel_index();
                Voxel curVoxel = voxels[index.x + dims.x * index.y + dims.x * dims.y * index.z];
                double L = curVoxel.face[sample.get_cur_face()];

                double Po_xi = probability(L);
                double Io_xi = entropy(Po_xi);
                double Iv_xi = Io_xi * Pv_xi;

                if(L >= L_OCCUPIED){
                    if(previous_L > L_FREE && previous_L < L_OCCUPIED) total_In += Iv_xi;
                    break;
                }

                previous_L = L;
                Pv_xi *= (1 - Po_xi);
            }
            sample.one_step();
            if(sample.is_over()) break;
        }
        values[i] = total_In;
    }
    return vector_to_pyarray(values, {height, width});
}

py::array_t<double> VoxelGrid::krigel_face(int32_t height, int32_t width,
                                               py::array_t<double> intrinsic_, py::array_t<double> pose_) const {


    double* intrinsic = request_buffer_ptr(intrinsic_, {3, 3});
    double* pose = request_buffer_ptr(pose_, {4, 4});

    std::vector<double> values(height * width);
    point3d<double> startPoint = {pose[3], pose[7], pose[11]};

    #pragma omp parallel for
    for(int i = 0; i < height * width; i++){

        auto sample = FastVoxelTraversalSample(i, voxelSize, origin, startPoint, height, width, intrinsic, pose, MAX_RANGE);

        double total_Ie = 0;

        int32_t n_traversed = 0;
        while(!sample.is_hit()){
            if(sample.is_valid(dims)){
                n_traversed++;

                point3d<int32_t> index = sample.get_cur_voxel_index();
                Voxel curVoxel = voxels[index.x + dims.x * index.y + dims.x * dims.y * index.z];
                double L = curVoxel.face[sample.get_cur_face()];

                double Po_xi = probability(L);
                double Ie = entropy(Po_xi);
                total_Ie += Ie;

                if(L >= L_OCCUPIED) break;
            
            }
            sample.one_step();
            if(sample.is_over()) break;
        }
        values[i] = n_traversed != 0 ? total_Ie / n_traversed : 0;
    }
    return vector_to_pyarray(values, {height, width});
}

py::array_t<double> VoxelGrid::proximity_count_face(int32_t height, int32_t width,
                                               py::array_t<double> intrinsic_, py::array_t<double> pose_) const {


    double* intrinsic = request_buffer_ptr(intrinsic_, {3, 3});
    double* pose = request_buffer_ptr(pose_, {4, 4});

    std::vector<double> values(height * width);
    point3d<double> startPoint = {pose[3], pose[7], pose[11]};

    #pragma omp parallel for
    for(int i = 0; i < height * width; i++){

        auto sample = FastVoxelTraversalSample(i, voxelSize, origin, startPoint, height, width, intrinsic, pose, MAX_RANGE);

        double total_Iv = 0;
        double Pv_xi = 1;

        while(!sample.is_hit()){
            if(sample.is_valid(dims)){
                point3d<int32_t> index = sample.get_cur_voxel_index();
                Voxel curVoxel = voxels[index.x + dims.x * index.y + dims.x * dims.y * index.z];
                double Po_xi = probability(curVoxel.occLogOdd);
                double Io_xi = entropy(Po_xi);
                double Iv_xi = Io_xi * Pv_xi;
                total_Iv += Iv_xi;

                if(curVoxel.occLogOdd >= L_OCCUPIED) break;;

                Po_xi *= (1 - Po_xi);
            }
            sample.one_step();
            if(sample.is_over()){
                values[i] = 0;
            }
        }
    }
    return vector_to_pyarray(values, {height, width});
}