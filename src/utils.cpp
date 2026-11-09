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
 * @file utils.cpp
 * @brief Provides auxiliary routines for command-line argument parsing, environment inspection, and application configuration.
 * @author Savelii Pototskii
 * @copyright Copyright (C) 2026 Savelii Pototskii (savalione.com)
 * @copyright SPDX-License-Identifier: GPL-3.0-or-later
 */
#include "utils.h"

#include <sycl/sycl.hpp>
#include <stdexcept>
#include <string>
#include <iostream>

#include "xgetopt/xgetopt.h"
#include "config.h"

configuration parse_arguments(int argc, char **argv)
{
    if(argc == 1)
    {
        print_help();
        exit(EXIT_SUCCESS);
    }

    configuration result;

    // Accepted parameters
    std::string const short_opts = "c:o:d:n:i:m:r:s:z:lhv";

    // clang-format off
    std::array<xoption, 15> long_options =
        {{
            {"cloud",                 xrequired_argument, nullptr, 'c'},
            {"output",                xrequired_argument, nullptr, 'o'},
            {"device",                xrequired_argument, nullptr, 'd'},
            {"num-points",            xrequired_argument, nullptr, 'n'},
            {"iterations",            xrequired_argument, nullptr, 'i'},
            {"mode",                  xrequired_argument, nullptr, 'm'},
            {"translate",             xrequired_argument, nullptr, 1001},
            {"sphere-crop",           xrequired_argument, nullptr, 'r'},
            {"intensity-scale",       xrequired_argument, nullptr, 's'},
            {"crop-z",                xrequired_argument, nullptr, 'z'},
            {"verify",                xno_argument,       nullptr, 1002},

            {"device-list",           xno_argument,       nullptr, 'l'},
            {"help",                  xno_argument,       nullptr, 'h'},
            {"version",               xno_argument,       nullptr, 'v'},
            {0, 0, 0, 0} // Sentinel
        }};
    // clang-format on

    while(true)
    {
        auto const opt = xgetopt_long(argc, argv, short_opts.c_str(), long_options.data(), nullptr);

        if(opt == -1)
            break;

        // clang-format off
        switch(opt)
        {
            case 'c': result.input_cloud = xoptarg; break;
            case 'o': result.output_cloud = xoptarg; break;
            case 'd': result.device_name = xoptarg; break;
            case 'n': result.num_points = static_cast<std::size_t>(std::stoull(xoptarg)); break;
            case 'i': result.iterations = static_cast<std::size_t>(std::stoull(xoptarg)); break;
            case 'm':
            {
                std::string const mode_str = xoptarg;
                if(mode_str == "fused")
                {
                    result.mode = pipeline_mode::fused;
                }
                else if(mode_str == "unfused")
                {
                    result.mode = pipeline_mode::unfused;
                }
                else if(mode_str == "compare")
                {
                    result.mode = pipeline_mode::compare;
                }
                else
                {
                    throw std::runtime_error("Invalid mode '" + mode_str + "'. Expected: fused, unfused, or compare.");
                }
                break;
            }
            case 1001:
            {
                std::stringstream ss(xoptarg);
                std::string item;
                std::vector<float> coords;
                while(std::getline(ss, item, ','))
                {
                    coords.push_back(std::stof(item));
                }
                if(coords.size() != 3)
                {
                    throw std::runtime_error("--translate requires exactly 3 comma-separated values: dx,dy,dz.");
                }
                result.apply_translation = true;
                result.translate_x       = coords[0];
                result.translate_y       = coords[1];
                result.translate_z       = coords[2];
                break;
            }
            case 'r':
                result.apply_sphere_crop = true;
                result.sphere_radius     = std::stof(xoptarg);
                break;
            case 's':
                result.apply_intensity_scale = true;
                result.intensity_factor      = std::stof(xoptarg);
                break;
            case 'z':
            {
                std::stringstream ss(xoptarg);
                std::string item;
                std::vector<float> bounds;
                while(std::getline(ss, item, ','))
                {
                    bounds.push_back(std::stof(item));
                }
                if(bounds.size() != 2)
                {
                    throw std::runtime_error("--crop-z requires exactly 2 comma-separated values: min_z,max_z.");
                }
                result.apply_z_crop = true;
                result.min_z        = bounds[0];
                result.max_z        = bounds[1];
                break;
            }
            case 1002: result.verify_results = true; break;
            case 'l': print_all_sycl_devices(); std::exit(EXIT_SUCCESS);
            case 'h': print_help(); exit(EXIT_SUCCESS); break;
            case 'v': print_version(); exit(EXIT_SUCCESS); break;
            default: throw std::runtime_error("could not parse parameters, use --help for usage.");
        }
        // clang-format on
    }

    return result;
}

void print_help()
{
    std::string const help =
        R"(sycl-kernel-fusion: High-throughput spatial pointwise processing and compile-time kernel fusion evaluation in SYCL.

Usage:
  sycl-kernel-fusion [options...]
  sycl-kernel-fusion -l
  sycl-kernel-fusion -h | -v

Execution modes:
  -m, --mode <mode>                   Execution mode: 'fused', 'unfused', or 'compare' (default: compare).
                                      - fused:   Executes a single-pass composed kernel with zero intermediate VRAM round-trips.
                                      - unfused: Executes sequential eager kernels with intermediate VRAM memory materialization.
                                      - compare: Executes both modes, benchmarks speedup, and reports DRAM traffic metrics.
      --verify                        Validates bit-level parity between fused and unfused results.

General options:
  -c, --cloud <path>                  Path to input point cloud file. If omitted, synthetic data is generated.
  -o, --output <path>                 Path to write processed point cloud. If omitted, disk write is skipped.
  -d, --device <name>                 Explicitly select a target SYCL device by name substring.
  -n, --num-points <uint>             Point count to generate synthetically if no cloud is supplied (default: 10000000).
  -i, --iterations <uint>             Timing iterations for average performance reporting (default: 10).
  -l, --device-list                   Enumerate and display all available SYCL platforms and devices.
  -h, --help                          Display this help text and exit.
  -v, --version                       Display application version and license information and exit.

Spatial transformations & filters:
      --translate <dx,dy,dz>          Apply 3D translation offset (e.g. --translate 10.0,-5.0,2.5).
  -r, --sphere-crop <radius>          Prune points outside radial distance R from origin.
  -s, --intensity-scale <factor>      Multiply point intensities by scalar factor (e.g. 0.0039215 for 1/255).
  -z, --crop-z <min,max>              Prune points outside Z-axis bounding range (e.g. --crop-z -10.0,50.0).

Examples:
  sycl-kernel-fusion --device-list
  sycl-kernel-fusion -n 10000000 -m compare --translate 1.0,2.0,3.0 -r 500.0 --verify
  sycl-kernel-fusion -c input.bin -o output.bin -m fused --translate 0.0,0.0,-5.0 -s 0.00392 -z -10.0,30.0
  sycl-kernel-fusion -d Arc -n 20000000 -m compare
)";

    std::cout << help << std::endl;
}

void print_version()
{
    std::string version = "sycl-kernel-fusion " PROJECT_VERSION "\n"
                          "Copyright (C) 2026 " PROJECT_AUTHOR "\n"
                          "License " PROJECT_LICENSE ": GNU GPL version 3 or later <https://gnu.org/licenses/gpl.html>.\n"
                          "This is free software: you are free to change and redistribute it.\n"
                          "There is NO WARRANTY, to the extent permitted by law.\n\n"
                          "Homepage: " PROJECT_HOMEPAGE_URL "\n";

    std::cout << version << std::endl;
}

// The code taken mostly from: https://github.com/SavaLione/sycl-acpp-example/blob/main/src/sycl-acpp-example.cpp
void print_all_sycl_devices()
{
    // Retrieve all available SYCL platforms
    std::vector<sycl::platform> platforms = sycl::platform::get_platforms();

    if(platforms.empty())
    {
        std::cout << "No SYCL platforms found on this system." << std::endl;
        return;
    }

    for(const auto &platform : platforms)
    {
        std::cout << "Platform: " << platform.get_info<sycl::info::platform::name>() << std::endl;
        std::cout << "Vendor:   " << platform.get_info<sycl::info::platform::vendor>() << std::endl;
        std::cout << "Version:  " << platform.get_info<sycl::info::platform::version>() << std::endl;

        // Get devices associated with the current platform
        std::vector<sycl::device> devices = platform.get_devices();

        for(const auto &device : devices)
        {
            std::cout << "  Device: " << device.get_info<sycl::info::device::name>() << std::endl;

            // Determine the device type
            std::cout << "  Type:   ";
            if(device.is_gpu())
            {
                std::cout << "GPU" << std::endl;
            }
            else if(device.is_cpu())
            {
                std::cout << "CPU" << (device.is_host() ? " (host)" : "") << std::endl;
            }
            else if(device.is_accelerator())
            {
                std::cout << "Accelerator" << std::endl;
            }
            else
            {
                std::cout << "Unknown" << std::endl;
            }

            std::cout << "  Driver: " << device.get_info<sycl::info::device::driver_version>() << std::endl;
        }
    }
}

sycl::device select_sycl_device(std::string const &device_substring)
{
    if(device_substring.empty())
    {
        return sycl::device {sycl::default_selector_v};
    }

    for(auto const &platform : sycl::platform::get_platforms())
    {
        for(auto const &device : platform.get_devices())
        {
            if(device.get_info<sycl::info::device::name>().find(device_substring) != std::string::npos)
            {
                return device;
            }
        }
    }

    throw std::runtime_error("Could not find any SYCL device matching: '" + device_substring + "'");
}

bool verify_parity(sycl::queue &q, point_intensity const *d_a, point_intensity const *d_b, std::size_t const count)
{
    std::vector<point_intensity> h_a(count);
    std::vector<point_intensity> h_b(count);

    q.copy(d_a, h_a.data(), count).wait();
    q.copy(d_b, h_b.data(), count).wait();

    constexpr float epsilon = 1e-4f;
    for(std::size_t i = 0; i < count; ++i)
    {
        if(std::abs(h_a[i].x - h_b[i].x) > epsilon || std::abs(h_a[i].y - h_b[i].y) > epsilon || std::abs(h_a[i].z - h_b[i].z) > epsilon ||
           std::abs(h_a[i].intensity - h_b[i].intensity) > epsilon)
        {
            return false;
        }
    }
    return true;
}

void synthesize_points_on_device(sycl::queue &q, point_intensity *d_buffer, std::size_t const count)
{
    q.parallel_for(sycl::range<1> {count}, [=](sycl::id<1> const idx) {
         float const val = static_cast<float>(idx[0]);
         d_buffer[idx]   = point_intensity {val * 0.05f, val * 0.02f, val * 0.01f, 128.0f};
     }).wait();
}
