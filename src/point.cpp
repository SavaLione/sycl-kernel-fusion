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
 * @file point.cpp
 * @brief Defines the fundamental three-dimensional point structures and memory layout requirements.
 * @author Savelii Pototskii
 * @copyright Copyright (C) 2026 Savelii Pototskii (savalione.com)
 * @copyright SPDX-License-Identifier: GPL-3.0-or-later
 */
#include "point.h"

std::vector<point_intensity> convert_to_intensity_cloud(std::vector<point_3d> const &input)
{
    std::vector<point_intensity> result(input.size());
    for(std::size_t i = 0; i < input.size(); ++i)
    {
        result[i] = point_intensity {input[i].x, input[i].y, input[i].z, 100.0f};
    }
    return result;
}

std::vector<point_3d> convert_to_point_3d_cloud(std::vector<point_intensity> const &input)
{
    std::vector<point_3d> result(input.size());
    for(std::size_t i = 0; i < input.size(); ++i)
    {
        result[i] = point_3d {input[i].x, input[i].y, input[i].z};
    }
    return result;
}
