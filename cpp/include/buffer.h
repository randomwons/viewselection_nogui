#ifndef __BUFFER_H__
#define __BUFFER_H__

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

namespace py = pybind11;

template<typename T>
T* request_buffer_ptr(py::array_t<T>& array, const std::vector<ssize_t>& expected_shape){
    py::buffer_info info = array.request();
    if(info.ndim != expected_shape.size()) {
        throw std::runtime_error{"Unexpected number of dimensions"};
    }

    for(ssize_t i = 0; i < expected_shape.size(); ++i){
        if(expected_shape[i] >= 0 && info.shape[i] != expected_shape[i]){
            throw std::runtime_error{"Shape mismatch"};
        }
    }
    return static_cast<T*>(info.ptr);
}

template<typename T>
py::array_t<T> vector_to_pyarray(std::vector<T>& vector, const std::vector<ssize_t>& size){
    py::array_t<T> array = py::array_t<T>(size);
    py::buffer_info info = array.request();
    T* ptr = static_cast<T*>(info.ptr);
    std::copy(vector.begin(), vector.end(), ptr);

    return array;
}

#endif // __BUFFER_H__