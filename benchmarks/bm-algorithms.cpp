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
 * @file bm-algorithms.cpp
 * @brief Benchmark suite evaluating individual and fused pointwise SYCL kernel throughput and memory traffic.
 * @author Savelii Pototskii
 * @copyright Copyright (C) 2026 Savelii Pototskii (savalione.com)
 * @copyright SPDX-License-Identifier: GPL-3.0-or-later
 */
#include "bm-algorithms.h"

#include <sycl/sycl.hpp>
#include <cstddef>

#include "algorithms.h"
#include "point.h"

void bm_sycl_translation(benchmark::State &state)
{
    std::size_t const num_points = state.range(0);
    sycl::queue q {sycl::default_selector_v, sycl::property::queue::in_order {}};

    point_3d *d_in  = sycl::malloc_device<point_3d>(num_points, q);
    point_3d *d_out = sycl::malloc_device<point_3d>(num_points, q);

    q.parallel_for(sycl::range<1> {num_points}, [=](sycl::id<1> const idx) {
         float const val = static_cast<float>(idx[0]);
         d_in[idx]       = point_3d {val, val, val};
     }).wait();

    translation_transform const op {1.5f, -2.0f, 3.5f};

    for(auto _ : state)
    {
        q.parallel_for(sycl::range<1> {num_points}, [=](sycl::id<1> const idx) {
             d_out[idx] = op(d_in[idx]);
         }).wait();
    }

    std::size_t const bytes_per_iteration = num_points * sizeof(point_3d) * 2; // 1 Read + 1 Write
    state.SetBytesProcessed(state.iterations() * bytes_per_iteration);
    state.SetItemsProcessed(state.iterations() * num_points);

    sycl::free(d_in, q);
    sycl::free(d_out, q);
}

void bm_sycl_sphere_crop(benchmark::State &state)
{
    std::size_t const num_points = state.range(0);
    sycl::queue q {sycl::default_selector_v, sycl::property::queue::in_order {}};

    point_intensity *d_in  = sycl::malloc_device<point_intensity>(num_points, q);
    point_intensity *d_out = sycl::malloc_device<point_intensity>(num_points, q);

    q.parallel_for(sycl::range<1> {num_points}, [=](sycl::id<1> const idx) {
         float const val = static_cast<float>(idx[0]);
         d_in[idx]       = point_intensity {val, val, val, 1.0f};
     }).wait();

    sphere_crop_filter const op {point_3d {0.0f, 0.0f, 0.0f}, 1000.0f};

    for(auto _ : state)
    {
        q.parallel_for(sycl::range<1> {num_points}, [=](sycl::id<1> const idx) {
             d_out[idx] = op(d_in[idx]);
         }).wait();
    }

    std::size_t const bytes_per_iteration = num_points * sizeof(point_intensity) * 2;
    state.SetBytesProcessed(state.iterations() * bytes_per_iteration);
    state.SetItemsProcessed(state.iterations() * num_points);

    sycl::free(d_in, q);
    sycl::free(d_out, q);
}

void bm_sycl_rgba_to_intensity(benchmark::State &state)
{
    std::size_t const num_points = state.range(0);
    sycl::queue q {sycl::default_selector_v, sycl::property::queue::in_order {}};

    point_rgba *d_in       = sycl::malloc_device<point_rgba>(num_points, q);
    point_intensity *d_out = sycl::malloc_device<point_intensity>(num_points, q);

    q.parallel_for(sycl::range<1> {num_points}, [=](sycl::id<1> const idx) {
         float const val = static_cast<float>(idx[0]);
         d_in[idx]       = point_rgba {val, val, val, 255, 128, 64, 255};
     }).wait();

    rgba_to_intensity const op {};

    for(auto _ : state)
    {
        q.parallel_for(sycl::range<1> {num_points}, [=](sycl::id<1> const idx) {
             d_out[idx] = op(d_in[idx]);
         }).wait();
    }

    std::size_t const bytes_per_iteration = num_points * (sizeof(point_rgba) + sizeof(point_intensity));
    state.SetBytesProcessed(state.iterations() * bytes_per_iteration);
    state.SetItemsProcessed(state.iterations() * num_points);

    sycl::free(d_in, q);
    sycl::free(d_out, q);
}

void bm_sycl_prevoxelization(benchmark::State &state)
{
    std::size_t const num_points = state.range(0);
    sycl::queue q {sycl::default_selector_v, sycl::property::queue::in_order {}};

    point_3d *d_in        = sycl::malloc_device<point_3d>(num_points, q);
    voxel_index_3d *d_out = sycl::malloc_device<voxel_index_3d>(num_points, q);

    q.parallel_for(sycl::range<1> {num_points}, [=](sycl::id<1> const idx) {
         float const val = static_cast<float>(idx[0]);
         d_in[idx]       = point_3d {val * 0.1f, val * 0.1f, val * 0.1f};
     }).wait();

    prevoxelization_grid_mapping const op {0.05f, point_3d {0.0f, 0.0f, 0.0f}};

    for(auto _ : state)
    {
        q.parallel_for(sycl::range<1> {num_points}, [=](sycl::id<1> const idx) {
             d_out[idx] = op(d_in[idx]);
         }).wait();
    }

    std::size_t const bytes_per_iteration = num_points * (sizeof(point_3d) + sizeof(voxel_index_3d));
    state.SetBytesProcessed(state.iterations() * bytes_per_iteration);
    state.SetItemsProcessed(state.iterations() * num_points);

    sycl::free(d_in, q);
    sycl::free(d_out, q);
}

void bm_sycl_finite_check(benchmark::State &state)
{
    std::size_t const num_points = state.range(0);
    sycl::queue q {sycl::default_selector_v, sycl::property::queue::in_order {}};

    point_3d *d_in = sycl::malloc_device<point_3d>(num_points, q);
    bool *d_out    = sycl::malloc_device<bool>(num_points, q);

    q.parallel_for(sycl::range<1> {num_points}, [=](sycl::id<1> const idx) {
         float const val = static_cast<float>(idx[0]);
         d_in[idx]       = point_3d {val, val, val};
     }).wait();

    finite_point_check const op {};

    for(auto _ : state)
    {
        q.parallel_for(sycl::range<1> {num_points}, [=](sycl::id<1> const idx) {
             d_out[idx] = op(d_in[idx]);
         }).wait();
    }

    std::size_t const bytes_per_iteration = num_points * (sizeof(point_3d) + sizeof(bool));
    state.SetBytesProcessed(state.iterations() * bytes_per_iteration);
    state.SetItemsProcessed(state.iterations() * num_points);

    sycl::free(d_in, q);
    sycl::free(d_out, q);
}

void bm_sycl_intensity_scale(benchmark::State &state)
{
    std::size_t const num_points = static_cast<std::size_t>(state.range(0));
    sycl::queue q {sycl::default_selector_v, sycl::property::queue::in_order {}};

    point_intensity *d_in  = sycl::malloc_device<point_intensity>(num_points, q);
    point_intensity *d_out = sycl::malloc_device<point_intensity>(num_points, q);

    q.parallel_for(sycl::range<1> {num_points}, [=](sycl::id<1> const idx) {
         float const val = static_cast<float>(idx[0]);
         d_in[idx]       = point_intensity {val, val, val, 128.0f};
     }).wait();

    intensity_scale const op {1.0f / 255.0f};

    for(auto _ : state)
    {
        q.parallel_for(sycl::range<1> {num_points}, [=](sycl::id<1> const idx) {
             d_out[idx] = op(d_in[idx]);
         }).wait();
    }

    // Memory traffic: 1 read + 1 write of point_intensity
    std::size_t const bytes_per_iteration = num_points * sizeof(point_intensity) * 2;
    state.SetBytesProcessed(state.iterations() * bytes_per_iteration);
    state.SetItemsProcessed(state.iterations() * num_points);

    sycl::free(d_in, q);
    sycl::free(d_out, q);
}

void bm_sycl_bounding_box_filter(benchmark::State &state)
{
    std::size_t const num_points = static_cast<std::size_t>(state.range(0));
    sycl::queue q {sycl::default_selector_v, sycl::property::queue::in_order {}};

    point_intensity *d_in  = sycl::malloc_device<point_intensity>(num_points, q);
    point_intensity *d_out = sycl::malloc_device<point_intensity>(num_points, q);

    q.parallel_for(sycl::range<1> {num_points}, [=](sycl::id<1> const idx) {
         float const val = static_cast<float>(idx[0]);
         d_in[idx]       = point_intensity {val, val, val, 100.0f};
     }).wait();

    bounding_box_filter const op {-500.0f, 500.0f};

    for(auto _ : state)
    {
        q.parallel_for(sycl::range<1> {num_points}, [=](sycl::id<1> const idx) {
             d_out[idx] = op(d_in[idx]);
         }).wait();
    }

    // Memory traffic: 1 read + 1 write of point_intensity
    std::size_t const bytes_per_iteration = num_points * sizeof(point_intensity) * 2;
    state.SetBytesProcessed(state.iterations() * bytes_per_iteration);
    state.SetItemsProcessed(state.iterations() * num_points);

    sycl::free(d_in, q);
    sycl::free(d_out, q);
}

void bm_sycl_unfused_pipeline(benchmark::State &state)
{
    std::size_t const num_points = static_cast<std::size_t>(state.range(0));
    std::size_t const pipeline_k = static_cast<std::size_t>(state.range(1));
    sycl::queue q {sycl::default_selector_v, sycl::property::queue::in_order {}};

    // Allocate K + 1 device buffers: buffer[0] is input, buffer[K] is final output
    std::vector<point_intensity *> buffers(pipeline_k + 1, nullptr);
    for(std::size_t i = 0; i <= pipeline_k; ++i)
    {
        buffers[i] = sycl::malloc_device<point_intensity>(num_points, q);
    }

    q.parallel_for(sycl::range<1> {num_points}, [=, d_in = buffers[0]](sycl::id<1> const idx) {
         float const val = static_cast<float>(idx[0]);
         d_in[idx]       = point_intensity {val, val, val, 100.0f};
     }).wait();

    // Define K pipeline operators
    translation_transform const op1 {5.0f, -2.5f, 1.0f};
    sphere_crop_filter const op2 {point_3d {0.0f, 0.0f, 0.0f}, 5000.0f};
    intensity_scale const op3 {1.0f / 255.0f};
    bounding_box_filter const op4 {-1000.0f, 1000.0f};
    translation_transform const op5 {-5.0f, 2.5f, -1.0f};

    for(auto _ : state)
    {
        // Step 1
        q.parallel_for(sycl::range<1> {num_points}, [=, in = buffers[0], out = buffers[1]](sycl::id<1> const idx) {
            out[idx] = op1(in[idx]);
        });

        // Step 2
        q.parallel_for(sycl::range<1> {num_points}, [=, in = buffers[1], out = buffers[2]](sycl::id<1> const idx) {
            out[idx] = op2(in[idx]);
        });

        // Step 3 (if K >= 3)
        if(pipeline_k >= 3)
        {
            q.parallel_for(sycl::range<1> {num_points}, [=, in = buffers[2], out = buffers[3]](sycl::id<1> const idx) {
                out[idx] = op3(in[idx]);
            });
        }

        // Step 4 (if K >= 4)
        if(pipeline_k >= 4)
        {
            q.parallel_for(sycl::range<1> {num_points}, [=, in = buffers[3], out = buffers[4]](sycl::id<1> const idx) {
                out[idx] = op4(in[idx]);
            });
        }

        // Step 5 (if K >= 5)
        if(pipeline_k >= 5)
        {
            q.parallel_for(sycl::range<1> {num_points}, [=, in = buffers[4], out = buffers[5]](sycl::id<1> const idx) {
                out[idx] = op5(in[idx]);
            });
        }

        q.wait();
    }

    double const point_size  = static_cast<double>(sizeof(point_intensity));
    double const total_bytes = static_cast<double>(num_points) * point_size * 2.0 * static_cast<double>(pipeline_k);

    state.SetBytesProcessed(state.iterations() * static_cast<std::size_t>(total_bytes));
    state.SetItemsProcessed(state.iterations() * num_points);

    state.counters["pipeline_k"]      = static_cast<double>(pipeline_k);
    state.counters["working_set_mb"]  = (static_cast<double>(num_points) * point_size * static_cast<double>(pipeline_k + 1)) / (1024.0 * 1024.0);
    state.counters["dram_traffic_gb"] = benchmark::Counter(total_bytes / 1e9, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1000);

    for(std::size_t i = 0; i <= pipeline_k; ++i)
    {
        sycl::free(buffers[i], q);
    }
}

void bm_sycl_fused_pipeline(benchmark::State &state)
{
    std::size_t const num_points = static_cast<std::size_t>(state.range(0));
    std::size_t const pipeline_k = static_cast<std::size_t>(state.range(1));
    sycl::queue q {sycl::default_selector_v, sycl::property::queue::in_order {}};

    point_intensity *d_in  = sycl::malloc_device<point_intensity>(num_points, q);
    point_intensity *d_out = sycl::malloc_device<point_intensity>(num_points, q);

    q.parallel_for(sycl::range<1> {num_points}, [=](sycl::id<1> const idx) {
         float const val = static_cast<float>(idx[0]);
         d_in[idx]       = point_intensity {val, val, val, 100.0f};
     }).wait();

    // Define K pipeline operators
    translation_transform const op1 {5.0f, -2.5f, 1.0f};
    sphere_crop_filter const op2 {point_3d {0.0f, 0.0f, 0.0f}, 5000.0f};
    intensity_scale const op3 {1.0f / 255.0f};
    bounding_box_filter const op4 {-1000.0f, 1000.0f};
    translation_transform const op5 {-5.0f, 2.5f, -1.0f};

    // Statically composed pipelines evaluated according to pipeline_k
    switch(pipeline_k)
    {
        case 2: {
            auto const fused = compose(op1, op2);
            for(auto _ : state)
            {
                q.parallel_for(sycl::range<1> {num_points}, [=](sycl::id<1> const idx) {
                     d_out[idx] = fused(d_in[idx]);
                 }).wait();
            }
            break;
        }
        case 3: {
            auto const fused = compose(op1, compose(op2, op3));
            for(auto _ : state)
            {
                q.parallel_for(sycl::range<1> {num_points}, [=](sycl::id<1> const idx) {
                     d_out[idx] = fused(d_in[idx]);
                 }).wait();
            }
            break;
        }
        case 4: {
            auto const fused = compose(op1, compose(op2, compose(op3, op4)));
            for(auto _ : state)
            {
                q.parallel_for(sycl::range<1> {num_points}, [=](sycl::id<1> const idx) {
                     d_out[idx] = fused(d_in[idx]);
                 }).wait();
            }
            break;
        }
        case 5: {
            auto const fused = compose(op1, compose(op2, compose(op3, compose(op4, op5))));
            for(auto _ : state)
            {
                q.parallel_for(sycl::range<1> {num_points}, [=](sycl::id<1> const idx) {
                     d_out[idx] = fused(d_in[idx]);
                 }).wait();
            }
            break;
        }
        default:
            state.SkipWithError("Unsupported pipeline depth requested.");
            break;
    }

    double const point_size               = static_cast<double>(sizeof(point_intensity));
    double const total_bytes              = static_cast<double>(num_points) * point_size * 2.0;
    double const eliminated_traffic_bytes = static_cast<double>(num_points) * point_size * 2.0 * static_cast<double>(pipeline_k - 1);

    state.SetBytesProcessed(state.iterations() * static_cast<std::size_t>(total_bytes));
    state.SetItemsProcessed(state.iterations() * num_points);

    state.counters["pipeline_k"]            = static_cast<double>(pipeline_k);
    state.counters["working_set_mb"]        = (static_cast<double>(num_points) * point_size * 2.0) / (1024.0 * 1024.0);
    state.counters["dram_traffic_gb"]       = benchmark::Counter(total_bytes / 1e9, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1000);
    state.counters["eliminated_traffic_mb"] = eliminated_traffic_bytes / (1024.0 * 1024.0);

    sycl::free(d_in, q);
    sycl::free(d_out, q);
}