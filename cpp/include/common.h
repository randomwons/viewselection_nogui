#ifndef __COMMON_H__
#define __COMMON_H__

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <omp.h>

#include <cmath>
#include <limits>
#include <memory>
#include <vector>
#include <array>
#include <stdexcept>

#include "mathutils.h"
#include "buffer.h"

namespace py = pybind11;

constexpr double MAX_RANGE = 7.0;

constexpr double L_UPDATE_OCCUPIED = 2.0;
constexpr double L_UPDATE_FREE = -2.0;
constexpr double L_OCCUPIED = 1.0;
constexpr double L_FREE = -1.0;
constexpr double L_UNOBSERVED = 0.0;


#endif // __COMMON_H__