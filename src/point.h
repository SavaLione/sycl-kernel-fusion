/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * Copyright (C) 2026 Savelii Pototskii (savalione.com)
 *
 * Author: Savelii Pototskii <savelii.pototskii@gmail.com>
 *
 * This file is part of sycl-kernel-fusion.
 *
 * sycl-kernel-fusion is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation, either version 3
 * of the License, or (at your option) any later version.
 *
 * sycl-kernel-fusion is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with sycl-kernel-fusion. If not, see <https://www.gnu.org/licenses/>.
 */
/**
 * @file point.h
 * @brief Defines the fundamental three-dimensional point structures and memory layout requirements.
 * @author Savelii Pototskii
 * @copyright Copyright (C) 2026 Savelii Pototskii (savalione.com)
 * @copyright SPDX-License-Identifier: GPL-3.0-or-later
 */
#ifndef POINT_H
#define POINT_H

#include <cstdint>
#include <iostream>
#include <type_traits>
#include <vector>

/**
 * @brief Representation of a basic 3D point in space.
 */
struct point_3d
{
    float x, y, z;

    friend std::ostream &operator<<(std::ostream &os, point_3d const &p)
    {
        os << "x = " << std::to_string(p.x) << " y = " << std::to_string(p.y) << " z = " << std::to_string(p.z);
        return os;
    }
};

static_assert(sizeof(point_3d) == 3 * sizeof(float), "point_3d layout must match binary standard layout.");
static_assert(std::is_standard_layout_v<point_3d>, "point_3d must have standard layout.");

/**
 * @brief Representation of a 3D point with an intensity attribute.
 */
struct point_intensity
{
    float x, y, z;
    float intensity;

    friend std::ostream &operator<<(std::ostream &os, point_intensity const &p)
    {
        os << "x = " << std::to_string(p.x) << " y = " << std::to_string(p.y) << " z = " << std::to_string(p.z) << " intensity = " << std::to_string(p.intensity);
        return os;
    }
};

static_assert(sizeof(point_intensity) == 4 * sizeof(float), "point_intensity layout must match binary standard layout.");
static_assert(std::is_standard_layout_v<point_intensity>, "point_intensity must have standard layout.");

/**
 * @brief Representation of a 3D point with 8-bit RGBA color attributes.
 */
struct point_rgba
{
    float x, y, z;
    std::uint8_t r, g, b, a;

    friend std::ostream &operator<<(std::ostream &os, point_rgba const &p)
    {
        os << "x = " << std::to_string(p.x) << " y = " << std::to_string(p.y) << " z = " << std::to_string(p.z) << " rgba = (" << static_cast<int>(p.r) << ", "
           << static_cast<int>(p.g) << ", " << static_cast<int>(p.b) << ", " << static_cast<int>(p.a) << ")";
        return os;
    }
};

static_assert(sizeof(point_rgba) == 3 * sizeof(float) + 4 * sizeof(std::uint8_t), "point_rgba layout must match binary standard layout.");
static_assert(std::is_standard_layout_v<point_rgba>, "point_rgba must have standard layout.");

/**
 * @brief Representation of a 3D discrete voxel grid coordinate.
 */
struct voxel_index_3d
{
    std::int32_t x, y, z;

    friend std::ostream &operator<<(std::ostream &os, voxel_index_3d const &idx)
    {
        os << "vx = " << std::to_string(idx.x) << " vy = " << std::to_string(idx.y) << " vz = " << std::to_string(idx.z);
        return os;
    }
};

static_assert(sizeof(voxel_index_3d) == 3 * sizeof(std::int32_t), "voxel_index_3d layout must match binary standard layout.");
static_assert(std::is_standard_layout_v<voxel_index_3d>, "voxel_index_3d must have standard layout.");

/**
 * @brief Converts a standard vector of 3D points into extended points with default intensity.
 * 
 * @param[in] input 3D point cloud vector.
 * @return std::vector<point_intensity> Populated vector of point_intensity.
 */
std::vector<point_intensity> convert_to_intensity_cloud(std::vector<point_3d> const &input);

/**
 * @brief Converts a vector of extended intensity points back to standard 3D points.
 * 
 * @param[in] input Extended intensity point cloud.
 * @return std::vector<point_3d> Extracted 3D points.
 */
std::vector<point_3d> convert_to_point_3d_cloud(std::vector<point_intensity> const &input);

#endif // POINT_H