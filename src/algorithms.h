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
 * @file algorithms.h
 * @brief Defines fused pointwise transformation functors and filters for high-throughput spatial execution.
 * @author Savelii Pototskii
 * @copyright Copyright (C) 2026 Savelii Pototskii (savalione.com)
 * @copyright SPDX-License-Identifier: GPL-3.0-or-later
 */
#ifndef ALGORITHMS_H
#define ALGORITHMS_H

#include "point.h"

#include <cmath>

/**
 * @brief Sentinel value indicating an invalid or pruned point in fused kernels.
 */
inline constexpr float invalid_point_sentinel = -1.0f;

/**
 * @brief Functor composition wrapper enabling kernel fusion across pointwise passes.
 */
template<typename F1, typename F2>
struct composed_functor
{
    F1 f1;
    F2 f2;

    template<typename T>
    constexpr auto operator()(T const &p) const
    {
        return f2(f1(p));
    }
};

/**
 * @brief Chains two pointwise functors into a single fusible functor.
 * 
 * @tparam F1 First unary functor.
 * @tparam F2 Second unary functor.
 * @param[in] f1 First functor instance.
 * @param[in] f2 Second functor instance.
 * @return composed_functor<F1, F2> Composed functor.
 */
template<typename F1, typename F2>
constexpr composed_functor<F1, F2> compose(F1 f1, F2 f2)
{
    return {f1, f2};
}

/**
 * @brief Radial pass-through filter (sphere crop).
 * 
 * Prunes points exceeding distance R from a defined center.
 * Marks filtered points by setting their intensity to @ref invalid_point_sentinel.
 */
struct sphere_crop_filter
{
    float center_x {0.0f};
    float center_y {0.0f};
    float center_z {0.0f};
    float radius_sq {0.0f};

    constexpr sphere_crop_filter(point_3d const &center, float const radius) : center_x {center.x}, center_y {center.y}, center_z {center.z}, radius_sq {radius * radius} {}

    constexpr point_intensity operator()(point_intensity const &p) const
    {
        float const dx = p.x - center_x;
        float const dy = p.y - center_y;
        float const dz = p.z - center_z;

        float const dist_sq = dx * dx + dy * dy + dz * dz;
        if(dist_sq > radius_sq)
        {
            return {p.x, p.y, p.z, invalid_point_sentinel};
        }

        return p;
    }
};

/**
 * @brief Coordinate translation / offset.
 * 
 * Applies spatial translation vector (dx, dy, dz) to coordinates.
 */
struct translation_transform
{
    float dx {0.0f};
    float dy {0.0f};
    float dz {0.0f};

    constexpr translation_transform(float const x_offset, float const y_offset, float const z_offset) : dx {x_offset}, dy {y_offset}, dz {z_offset} {}

    template<typename PointT>
    constexpr PointT operator()(PointT const &p) const
    {
        PointT result = p;
        result.x += dx;
        result.y += dy;
        result.z += dz;
        return result;
    }
};

/**
 * @brief Color space conversion / luminance normalization.
 * 
 * Converts RGBA point attributes to a normalized intensity (ITU-R BT.601 standard).
 */
struct rgba_to_intensity
{
    constexpr point_intensity operator()(point_rgba const &p) const
    {
        // Y = 0.299 * R + 0.587 * G + 0.114 * B
        constexpr float r_coeff = 0.299f / 255.0f;
        constexpr float g_coeff = 0.587f / 255.0f;
        constexpr float b_coeff = 0.114f / 255.0f;

        float const lum = static_cast<float>(p.r) * r_coeff + static_cast<float>(p.g) * g_coeff + static_cast<float>(p.b) * b_coeff;

        return {p.x, p.y, p.z, lum};
    }
};

/**
 * @brief Coordinate quantization (pre-voxelization grid mapping).
 * 
 * Computes integer voxel indices without intermediate reduction or synchronization.
 */
struct prevoxelization_grid_mapping
{
    float inv_voxel_size {1.0f};
    float min_x {0.0f};
    float min_y {0.0f};
    float min_z {0.0f};

    constexpr prevoxelization_grid_mapping(float const voxel_size, point_3d const &min_bound)
        : inv_voxel_size {voxel_size > 0.0f ? (1.0f / voxel_size) : 1.0f},
          min_x {min_bound.x},
          min_y {min_bound.y},
          min_z {min_bound.z}
    {
    }

    template<typename PointT>
    voxel_index_3d operator()(PointT const &p) const
    {
        return {static_cast<std::int32_t>(std::floor((p.x - min_x) * inv_voxel_size)), static_cast<std::int32_t>(std::floor((p.y - min_y) * inv_voxel_size)),
                static_cast<std::int32_t>(std::floor((p.z - min_z) * inv_voxel_size))};
    }
};

/**
 * @brief Finite point validity check.
 * 
 * Validates that all coordinate components are finite real numbers (neither NaN nor Inf).
 */
struct finite_point_check
{
    template<typename PointT>
    bool operator()(PointT const &p) const
    {
        return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
    }
};

/**
 * @brief Intensity scaling and normalization.
 * 
 * Scales intensity values by a factor. Precomputes reciprocal to replace division with multiplication.
 */
struct intensity_scale
{
    float factor {1.0f};

    constexpr explicit intensity_scale(float const scale_factor) : factor {scale_factor} {}

    constexpr point_intensity operator()(point_intensity const &p) const
    {
        return {p.x, p.y, p.z, p.intensity * factor};
    }
};

/**
 * @brief Bounding box spatial filter along the Z-axis.
 * 
 * Sets the intensity of points outside the specified Z-range to @ref invalid_point_sentinel.
 */
struct bounding_box_filter
{
    float min_z {0.0f};
    float max_z {0.0f};

    constexpr bounding_box_filter(float const minimum_z, float const maximum_z) : min_z {minimum_z}, max_z {maximum_z} {}

    constexpr point_intensity operator()(point_intensity const &p) const
    {
        if(p.z < min_z || p.z > max_z)
        {
            return {p.x, p.y, p.z, invalid_point_sentinel};
        }
        return p;
    }
};

#endif // ALGORITHMS_H