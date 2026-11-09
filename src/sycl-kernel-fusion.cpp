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
 * @file sycl-kernel-fusion.cpp
 * @brief High-throughput spatial pointwise processing and compile-time kernel fusion evaluation in SYCL.
 * @author Savelii Pototskii
 * @copyright Copyright (C) 2026 Savelii Pototskii (savalione.com)
 * @copyright SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <sycl/sycl.hpp>

#include "utils.h"
#include "point.h"
#include "io.h"
#include "algorithms.h"

int main(int argc, char **argv)
{
    configuration config;

    // Parse config
    try
    {
        config = parse_arguments(argc, argv);
    }
    catch(std::exception const &e)
    {
        std::stringstream ss;
        ss << "sycl-kernel-fusion: " << e.what() << std::endl;
        std::cerr << ss.str();
        return EXIT_FAILURE;
    }

    try
    {
        sycl::device const device = select_sycl_device(config.device_name);
        sycl::queue q {device, sycl::property::queue::in_order {}};
        std::cout << "Target Device: " << device.get_info<sycl::info::device::name>() << std::endl;

        // Load or generate dataset
        std::size_t num_points = config.num_points;
        std::vector<point_intensity> host_input;

        if(!config.input_cloud.empty())
        {
            std::cout << "Loading cloud from: " << config.input_cloud << std::endl;
            std::vector<point_3d> const raw_cloud = load_point_cloud(config.input_cloud);
            num_points                            = raw_cloud.size();
            host_input                            = convert_to_intensity_cloud(raw_cloud);
            std::cout << "Loaded " << num_points << " points." << std::endl;
        }
        else
        {
            std::cout << "Synthesizing " << num_points << " points in VRAM" << std::endl;
        }

        // If no algorithmic stages were explicitly activated, enable standard defaults
        if(!config.apply_translation && !config.apply_sphere_crop && !config.apply_intensity_scale && !config.apply_z_crop)
        {
            std::cout << "No specific filters specified. Applying default benchmarking pipeline." << std::endl;
            config.apply_translation     = true;
            config.translate_x           = 5.0f;
            config.translate_y           = -2.5f;
            config.translate_z           = 1.0f;
            config.apply_sphere_crop     = true;
            config.sphere_radius         = 5000.0f;
            config.apply_intensity_scale = true;
            config.intensity_factor      = 1.0f / 255.0f;
        }

        // Pipeline operators
        translation_transform const op_trans {config.translate_x, config.translate_y, config.translate_z};
        sphere_crop_filter const op_crop {point_3d {0.0f, 0.0f, 0.0f}, config.sphere_radius};
        intensity_scale const op_scale {config.intensity_factor};
        bounding_box_filter const op_zcrop {config.min_z, config.max_z};

        // Determine active pipeline depth K
        std::size_t pipeline_k = 0;
        if(config.apply_translation)
        {
            pipeline_k++;
        }
        if(config.apply_sphere_crop)
        {
            pipeline_k++;
        }
        if(config.apply_intensity_scale)
        {
            pipeline_k++;
        }
        if(config.apply_z_crop)
        {
            pipeline_k++;
        }

        if(pipeline_k == 0)
        {
            throw std::runtime_error("Pipeline depth is zero. No operations to execute.");
        }

        // Allocate device memory
        point_intensity *d_in          = sycl::malloc_device<point_intensity>(num_points, q);
        point_intensity *d_out_fused   = sycl::malloc_device<point_intensity>(num_points, q);
        point_intensity *d_out_unfused = nullptr;

        // Allocate intermediate buffers if running unfused or compare mode
        std::vector<point_intensity *> d_intermediates;
        if(config.mode == pipeline_mode::unfused || config.mode == pipeline_mode::compare)
        {
            d_intermediates.resize(pipeline_k);
            for(std::size_t i = 0; i < pipeline_k; ++i)
            {
                d_intermediates[i] = sycl::malloc_device<point_intensity>(num_points, q);
            }
            d_out_unfused = d_intermediates.back();
        }

        // Initialize device memory
        if(!host_input.empty())
        {
            q.copy(host_input.data(), d_in, num_points).wait();
        }
        else
        {
            synthesize_points_on_device(q, d_in, num_points);
        }

        std::size_t const iterations = std::max<std::size_t>(1, config.iterations);
        double const buffer_bytes    = static_cast<double>(num_points * sizeof(point_intensity));

        // ---------------------------------------------------------------------
        // Fused execution
        // ---------------------------------------------------------------------
        double fused_avg_ms = 0.0;
        if(config.mode == pipeline_mode::fused || config.mode == pipeline_mode::compare)
        {
            q.wait();
            auto const start = std::chrono::high_resolution_clock::now();

            switch(pipeline_k)
            {
                case 1: {
                    for(std::size_t iter = 0; iter < iterations; ++iter)
                    {
                        q.parallel_for(sycl::range<1> {num_points}, [=](sycl::id<1> const idx) {
                             d_out_fused[idx] = op_trans(d_in[idx]);
                         }).wait();
                    }
                    break;
                }
                case 2: {
                    auto const fused = compose(op_trans, op_crop);
                    for(std::size_t iter = 0; iter < iterations; ++iter)
                    {
                        q.parallel_for(sycl::range<1> {num_points}, [=](sycl::id<1> const idx) {
                             d_out_fused[idx] = fused(d_in[idx]);
                         }).wait();
                    }
                    break;
                }
                case 3: {
                    auto const fused = compose(op_trans, compose(op_crop, op_scale));
                    for(std::size_t iter = 0; iter < iterations; ++iter)
                    {
                        q.parallel_for(sycl::range<1> {num_points}, [=](sycl::id<1> const idx) {
                             d_out_fused[idx] = fused(d_in[idx]);
                         }).wait();
                    }
                    break;
                }
                case 4: {
                    auto const fused = compose(op_trans, compose(op_crop, compose(op_scale, op_zcrop)));
                    for(std::size_t iter = 0; iter < iterations; ++iter)
                    {
                        q.parallel_for(sycl::range<1> {num_points}, [=](sycl::id<1> const idx) {
                             d_out_fused[idx] = fused(d_in[idx]);
                         }).wait();
                    }
                    break;
                }
                default:
                    throw std::runtime_error("Unsupported pipeline depth: " + std::to_string(pipeline_k));
            }

            auto const end = std::chrono::high_resolution_clock::now();
            fused_avg_ms   = std::chrono::duration<double, std::milli>(end - start).count() / static_cast<double>(iterations);
        }

        // ---------------------------------------------------------------------
        // Unfused execution
        // ---------------------------------------------------------------------
        double unfused_avg_ms = 0.0;
        if(config.mode == pipeline_mode::unfused || config.mode == pipeline_mode::compare)
        {
            q.wait();
            auto const start = std::chrono::high_resolution_clock::now();

            for(std::size_t iter = 0; iter < iterations; ++iter)
            {
                point_intensity const *current_in = d_in;
                std::size_t stage                 = 0;

                if(config.apply_translation)
                {
                    point_intensity *current_out = d_intermediates[stage];
                    q.parallel_for(sycl::range<1> {num_points}, [=](sycl::id<1> const idx) {
                        current_out[idx] = op_trans(current_in[idx]);
                    });
                    current_in = current_out;
                    ++stage;
                }

                if(config.apply_sphere_crop)
                {
                    point_intensity *current_out = d_intermediates[stage];
                    q.parallel_for(sycl::range<1> {num_points}, [=](sycl::id<1> const idx) {
                        current_out[idx] = op_crop(current_in[idx]);
                    });
                    current_in = current_out;
                    ++stage;
                }

                if(config.apply_intensity_scale)
                {
                    point_intensity *current_out = d_intermediates[stage];
                    q.parallel_for(sycl::range<1> {num_points}, [=](sycl::id<1> const idx) {
                        current_out[idx] = op_scale(current_in[idx]);
                    });
                    current_in = current_out;
                    ++stage;
                }

                if(config.apply_z_crop)
                {
                    point_intensity *current_out = d_intermediates[stage];
                    q.parallel_for(sycl::range<1> {num_points}, [=](sycl::id<1> const idx) {
                        current_out[idx] = op_zcrop(current_in[idx]);
                    });
                }

                q.wait();
            }

            auto const end = std::chrono::high_resolution_clock::now();
            unfused_avg_ms = std::chrono::duration<double, std::milli>(end - start).count() / static_cast<double>(iterations);
        }

        // ---------------------------------------------------------------------
        // Numerical verification
        // ---------------------------------------------------------------------
        if(config.verify_results && config.mode == pipeline_mode::compare)
        {
            std::cout << "Verifying numerical parity between fused and unfused buffers: ";
            bool const matched = verify_parity(q, d_out_fused, d_out_unfused, num_points);
            if(matched)
            {
                std::cout << "PASSED (outputs are identical)." << std::endl;
            }
            else
            {
                std::cout << "FAILED (numerical divergence detected)." << std::endl;
            }
        }

        // ---------------------------------------------------------------------
        // Output reporting
        // ---------------------------------------------------------------------
        std::cout << std::fixed << std::setprecision(3);
        std::cout << std::endl << "================ Execution Metrics ================" << std::endl;
        std::cout << "Dataset size:        " << num_points << " points" << std::endl;
        std::cout << "Pipeline depth (K):  " << pipeline_k << " operations" << std::endl;
        std::cout << "Iterations averaged: " << iterations << std::endl << std::endl;

        if(config.mode == pipeline_mode::unfused || config.mode == pipeline_mode::compare)
        {
            double const unfused_traffic = (2.0 * static_cast<double>(pipeline_k) * buffer_bytes) / 1e9;
            double const unfused_bw      = unfused_traffic / (unfused_avg_ms / 1000.0);
            std::cout << "Unfused pipeline latency: " << unfused_avg_ms << " ms" << std::endl;
            std::cout << "  -> DRAM traffic:         " << unfused_traffic << " GB" << std::endl;
            std::cout << "  -> Effective bandwidth:  " << unfused_bw << " GB/s" << std::endl << std::endl;
        }

        if(config.mode == pipeline_mode::fused || config.mode == pipeline_mode::compare)
        {
            double const fused_traffic = (2.0 * buffer_bytes) / 1e9;
            double const fused_bw      = fused_traffic / (fused_avg_ms / 1000.0);
            std::cout << "Fused pipeline latency:   " << fused_avg_ms << " ms" << std::endl;
            std::cout << "  -> DRAM traffic:         " << fused_traffic << " GB" << std::endl;
            std::cout << "  -> Effective bandwidth:  " << fused_bw << " GB/s" << std::endl << std::endl;
        }

        if(config.mode == pipeline_mode::compare)
        {
            double const speedup           = unfused_avg_ms / fused_avg_ms;
            double const traffic_reduction = static_cast<double>(pipeline_k);
            std::cout << "---------------------------------------------------" << std::endl;
            std::cout << "Speedup factor:         " << speedup << "x" << std::endl;
            std::cout << "DRAM traffic reduction: " << traffic_reduction << "x" << std::endl;
            std::cout << "===================================================" << std::endl;
        }

        // ---------------------------------------------------------------------
        // Save output to disk
        // ---------------------------------------------------------------------
        if(!config.output_cloud.empty())
        {
            std::cout << std::endl << "Writing result to a disk: " << config.output_cloud << std::endl;
            std::vector<point_intensity> host_output(num_points);
            point_intensity const *result_ptr = (config.mode == pipeline_mode::unfused) ? d_out_unfused : d_out_fused;

            q.copy(result_ptr, host_output.data(), num_points).wait();
            std::vector<point_3d> const out_points = convert_to_point_3d_cloud(host_output);

            if(config.output_cloud.ends_with(".ply"))
            {
                save_point_cloud_ply(config.output_cloud, out_points);
            }
            else
            {
                save_point_cloud(config.output_cloud, out_points);
            }
            std::cout << "Export completed." << std::endl;
        }

        sycl::free(d_in, q);
        sycl::free(d_out_fused, q);
        for(auto *buf : d_intermediates)
        {
            sycl::free(buf, q);
        }
    }
    catch(std::exception const &e)
    {
        std::stringstream ss;
        ss << "sycl-kernel-fusion: " << e.what() << std::endl;
        std::cerr << ss.str();
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
