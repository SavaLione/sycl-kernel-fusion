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
 * @file bm-common.h
 * @brief Provides common utilities, shared configuration, and default argument definitions for the benchmark suites.
 * @author Savelii Pototskii
 * @copyright Copyright (C) 2026 Savelii Pototskii (savalione.com)
 * @copyright SPDX-License-Identifier: GPL-3.0-or-later
 */
#ifndef BM_COMMON_H
#define BM_COMMON_H

#include <benchmark/benchmark.h>

/**
 * @brief Registers point counts for pointwise throughput evaluation.
 * 
 * @param b Benchmark instance pointer.
 */
void default_args_pointwise(benchmark::internal::Benchmark *b);

/**
 * @brief Registers 2D parameter grid (N points x K pipeline depth) for pipeline evaluation.
 * 
 * @param b Benchmark instance pointer.
 */
void default_args_pipeline(benchmark::internal::Benchmark *b);

/**
 * @brief Benchmarks the unfused execution model across varying pipeline depths K.
 * 
 * @param[in,out] state Google Benchmark state object tracking parameters:
 *                      - state.range(0): Total point count (N).
 *                      - state.range(1): Pipeline depth (K).
 */
void bm_sycl_unfused_pipeline(benchmark::State &state);

/**
 * @brief Benchmarks the compile-time fused execution model across varying pipeline depths K.
 * 
 * @param[in,out] state Google Benchmark state object tracking parameters:
 *                      - state.range(0): Total point count (N).
 *                      - state.range(1): Pipeline depth (K).
 */
void bm_sycl_fused_pipeline(benchmark::State &state);

#endif // BM_COMMON_H