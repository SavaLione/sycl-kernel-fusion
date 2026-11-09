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
#include "bm-common.h"

void default_args_pointwise(benchmark::internal::Benchmark *b)
{
    // clang-format off
    b->Args({10000})
     ->Args({100000})
     ->Args({1000000})
     ->Args({10000000})
     ->Args({20000000})
     ->Unit(benchmark::kMillisecond);
    // clang-format on
}

void default_args_pipeline(benchmark::internal::Benchmark *b)
{
    // clang-format off
    b->ArgsProduct({
        {10000, 100000, 1000000, 10000000, 20000000}, // N: Point cloud sizes
        {2, 3, 4, 5}                                  // K: Pipeline depths
    })->Unit(benchmark::kMillisecond);
    // clang-format on
}

BENCHMARK_MAIN();