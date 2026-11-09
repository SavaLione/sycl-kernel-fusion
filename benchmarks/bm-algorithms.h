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
 * @file bm-algorithms.h
 * @brief Benchmark suite evaluating individual and fused pointwise SYCL kernel throughput and memory traffic.
 * @author Savelii Pototskii
 * @copyright Copyright (C) 2026 Savelii Pototskii (savalione.com)
 * @copyright SPDX-License-Identifier: GPL-3.0-or-later
 */
#ifndef BM_ALGORITHMS_H
#define BM_ALGORITHMS_H

#include <benchmark/benchmark.h>

#include "bm-common.h"

/**
 * @brief Benchmarks the isolated translation transform kernel.
 * 
 * @param[in,out] state Google Benchmark state object tracking execution iterations and point count.
 */
void bm_sycl_translation(benchmark::State &state);

/**
 * @brief Benchmarks the isolated radial sphere crop pass-through filter.
 * 
 * @param[in,out] state Google Benchmark state object tracking execution iterations and point count.
 */
void bm_sycl_sphere_crop(benchmark::State &state);

/**
 * @brief Benchmarks the isolated RGBA-to-intensity luminance conversion.
 * 
 * @param[in,out] state Google Benchmark state object tracking execution iterations and point count.
 */
void bm_sycl_rgba_to_intensity(benchmark::State &state);

/**
 * @brief Benchmarks coordinate quantization (pre-voxelization grid mapping).
 * 
 * @param[in,out] state Google Benchmark state object tracking execution iterations and point count.
 */
void bm_sycl_prevoxelization(benchmark::State &state);

/**
 * @brief Benchmarks finite coordinate validity checking.
 */
void bm_sycl_finite_check(benchmark::State &state);

/**
 * @brief Benchmarks the isolated intensity scaling and normalization kernel.
 * 
 * @param[in,out] state Google Benchmark state object tracking execution iterations and point count.
 */
void bm_sycl_intensity_scale(benchmark::State &state);

/**
 * @brief Benchmarks the isolated bounding box Z-axis filter kernel.
 * 
 * @param[in,out] state Google Benchmark state object tracking execution iterations and point count.
 */
void bm_sycl_bounding_box_filter(benchmark::State &state);

/**
 * @brief Benchmarks the unfused multi-pass pipeline (3 sequential kernels with intermediate global VRAM round-trips).
 * 
 * @param[in,out] state Google Benchmark state object tracking execution iterations and point count.
 */
void bm_sycl_unfused_pipeline(benchmark::State &state);

/**
 * @brief Benchmarks the compile-time fused single-pass pipeline (1 kernel, zero intermediate VRAM round-trips).
 * 
 * @param[in,out] state Google Benchmark state object tracking execution iterations and point count.
 */
void bm_sycl_fused_pipeline(benchmark::State &state);

BENCHMARK(bm_sycl_translation)->Apply(default_args_pointwise);
BENCHMARK(bm_sycl_sphere_crop)->Apply(default_args_pointwise);
BENCHMARK(bm_sycl_rgba_to_intensity)->Apply(default_args_pointwise);
BENCHMARK(bm_sycl_prevoxelization)->Apply(default_args_pointwise);
BENCHMARK(bm_sycl_finite_check)->Apply(default_args_pointwise);
BENCHMARK(bm_sycl_intensity_scale)->Apply(default_args_pointwise);
BENCHMARK(bm_sycl_bounding_box_filter)->Apply(default_args_pointwise);
BENCHMARK(bm_sycl_unfused_pipeline)->Apply(default_args_pipeline);
BENCHMARK(bm_sycl_fused_pipeline)->Apply(default_args_pipeline);

#endif // BM_ALGORITHMS_H