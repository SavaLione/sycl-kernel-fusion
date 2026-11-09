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
 * @file utils.h
 * @brief Provides auxiliary routines for command-line argument parsing, environment inspection, and application configuration.
 * @author Savelii Pototskii
 * @copyright Copyright (C) 2026 Savelii Pototskii (savalione.com)
 * @copyright SPDX-License-Identifier: GPL-3.0-or-later
 */
#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <sycl/sycl.hpp>

#include "point.h"

/**
 * @brief Execution mode for kernel dispatching.
 */
enum class pipeline_mode
{
    fused,   ///< Run compile-time fused single-pass pipeline.
    unfused, ///< Run multi-pass eager execution with intermediate buffers.
    compare  ///< Run both modes, verify numerical parity, and output comparative performance metrics.
};

/**
 * @brief Holds the application's configuration settings, parsed from command-line arguments.
 */
struct configuration
{
    // General options
    std::string input_cloud  = "";                     ///< Path to input point cloud file. If empty, synthetic data is used.
    std::string output_cloud = "";                     ///< Path to output transformed cloud file (empty skips saving).
    std::string device_name  = "";                     ///< Target SYCL device substring match.
    std::size_t num_points   = 10000000;               ///< Number of synthetic points to generate if no input file is supplied.
    std::size_t iterations   = 10;                     ///< Number of measurement iterations for execution timing.
    pipeline_mode mode       = pipeline_mode::compare; ///< Pipeline execution mode.
    bool verify_results      = false;                  ///< Verify numerical parity between fused and unfused output buffers.

    // Algorithm switches and parameters
    bool apply_translation = false; ///< Flag indicating if coordinate translation is active.
    float translate_x      = 0.0f;  ///< Translation offset along X axis.
    float translate_y      = 0.0f;  ///< Translation offset along Y axis.
    float translate_z      = 0.0f;  ///< Translation offset along Z axis.

    bool apply_sphere_crop = false;  ///< Flag indicating if radial crop filter is active.
    float sphere_radius    = 100.0f; ///< Maximum distance from origin for sphere crop.

    bool apply_intensity_scale = false;         ///< Flag indicating if intensity scaling is active.
    float intensity_factor     = 1.0f / 255.0f; ///< Intensity scaling multiplier.

    bool apply_z_crop = false;  ///< Flag indicating if Z-axis bounding box filter is active.
    float min_z       = -50.0f; ///< Minimum Z boundary.
    float max_z       = 50.0f;  ///< Maximum Z boundary.
};

/**
 * @brief Parses command-line arguments and populates a configuration struct.
 * @param argc Argument count from `main`.
 * @param argv Argument vector from `main`.
 * @return A populated `configuration` struct.
 * @throws std::runtime_error on parsing failure or invalid arguments.
 */
configuration parse_arguments(int argc, char **argv);

/**
 * @brief Prints help information that is invoked by `-h` or `--help`.
*/
void print_help();

/**
 * @brief Prints version information that is invoked by `-v` or `--version`.
*/
void print_version();

/**
 * @brief Prints a list of all available SYCL devices and platforms to the console.
 * 
 * @note This implementation is mostly adapted from an open-source example.
 * @see https://github.com/SavaLione/sycl-acpp-example/blob/main/src/sycl-acpp-example.cpp
*/
void print_all_sycl_devices();

/**
 * @brief Selects a SYCL device based on a name substring or defaults to the standard device selector.
 * 
 * @param[in] device_substring Substring to match against available device names.
 * @return sycl::device Selected SYCL device.
 */
sycl::device select_sycl_device(std::string const &device_substring);

/**
 * @brief Verifies that two device buffers contain numerically equivalent points.
 * 
 * @param[in] q SYCL queue used for memory transfers.
 * @param[in] d_a First device buffer.
 * @param[in] d_b Second device buffer.
 * @param[in] count Number of points to compare.
 * @return bool True if all points match within tolerance, false otherwise.
 */
bool verify_parity(sycl::queue &q, point_intensity const *d_a, point_intensity const *d_b, std::size_t const count);

/**
 * @brief Generates synthetic input points directly inside device memory.
 * 
 * @param[in] q In-order SYCL queue.
 * @param[out] d_buffer Device memory pointer.
 * @param[in] count Total number of points to synthesize.
 */
void synthesize_points_on_device(sycl::queue &q, point_intensity *d_buffer, std::size_t const count);

#endif // UTILS_H