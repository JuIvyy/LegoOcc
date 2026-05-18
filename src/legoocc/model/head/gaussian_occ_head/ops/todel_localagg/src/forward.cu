#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;


// Perform initial steps for each Gaussian prior to rasterization.
__global__ void preprocessCUDA(
	const int P,
	const int* points_xyz,
	const int* radii,
	const dim3 grid,
	uint32_t* tiles_touched)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	tiles_touched[idx] = 0;

	uint3 rect_min, rect_max;
	getRect(points_xyz + 3 * idx, radii[idx], rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) * (rect_max.z - rect_min.z) == 0)
		return;

	tiles_touched[idx] = (rect_max.z - rect_min.z) * (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}


// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
// CUDA kernel: Each thread handles one pixel, for a block of CHANNELS output channels (starting from base_channel)
template <uint32_t CHANNELS>
__global__ void renderCUDA(
    int N,                      // number of points/pixels
    int total_channels,         // total number of output channels
    const float* __restrict__ pts,
    const int* __restrict__ points_int,
    dim3 grid,
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ point_list,
    const float* __restrict__ means3D,
    const float* __restrict__ cov3D,
    const float* __restrict__ opacity,
    const float* __restrict__ semantic,
    float* __restrict__ out,
    int base_channel            // start channel index for this kernel group
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const int* point_int = points_int + idx * 3;
    int voxel_idx = point_int[0] * grid.y * grid.z + point_int[1] * grid.z + point_int[2];

    float3 point = {pts[3 * idx], pts[3 * idx + 1], pts[3 * idx + 2]};
    uint2 range = ranges[voxel_idx];

    float C[CHANNELS] = {0.0f};

    // Accumulate contributions for this pixel
    for (int i = range.x; i < range.y; ++i) {
        int gs_idx = point_list[i];
        float3 d = {
            means3D[gs_idx * 3]     - point.x,
            means3D[gs_idx * 3 + 1] - point.y,
            means3D[gs_idx * 3 + 2] - point.z
        };
        float power =
            cov3D[gs_idx * 6]     * d.x * d.x +
            cov3D[gs_idx * 6 + 1] * d.y * d.y +
            cov3D[gs_idx * 6 + 2] * d.z * d.z;
        power = -0.5f * power
            - (cov3D[gs_idx * 6 + 3] * d.x * d.y
            +  cov3D[gs_idx * 6 + 4] * d.y * d.z
            +  cov3D[gs_idx * 6 + 5] * d.x * d.z);
        power = opacity[gs_idx] * exp(power);

        for (int ch = 0; ch < CHANNELS; ++ch) {
            int global_ch = base_channel + ch;
            if (global_ch < total_channels) { // Don't overflow on last group
                C[ch] += semantic[total_channels * gs_idx + global_ch] * power;
            }
        }
    }

    // Store results
    for (int ch = 0; ch < CHANNELS; ++ch) {
        int global_ch = base_channel + ch;
        if (global_ch < total_channels) {
            out[idx * total_channels + global_ch] = C[ch];
        }
    }
}


void FORWARD::render(
    int N,
    int C, // total channels
    const float* pts,
    const int* points_int,
    dim3 grid,
    const uint2* ranges,
    const uint32_t* point_list,
    const float* means3D,
    const float* cov3D,
    const float* opacity,
    const float* semantic,
    float* out
) {
    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
    int current_channel = 0;

    while (current_channel < C) {
        int left_channels = C - current_channel;

        if (left_channels >= 128) {
            renderCUDA<128><<<blocks_per_grid, threads_per_block>>>(
                N, C, pts, points_int, grid, ranges, point_list,
                means3D, cov3D, opacity, semantic, out, current_channel
            );
            current_channel += 128;
        }
        else if (left_channels >= 64) {
            renderCUDA<64><<<blocks_per_grid, threads_per_block>>>(
                N, C, pts, points_int, grid, ranges, point_list,
                means3D, cov3D, opacity, semantic, out, current_channel
            );
            current_channel += 64;
        }
        else if (left_channels >= 32) {
            renderCUDA<32><<<blocks_per_grid, threads_per_block>>>(
                N, C, pts, points_int, grid, ranges, point_list,
                means3D, cov3D, opacity, semantic, out, current_channel
            );
            current_channel += 32;
        }
        else if (left_channels >= 16) {
            renderCUDA<16><<<blocks_per_grid, threads_per_block>>>(
                N, C, pts, points_int, grid, ranges, point_list,
                means3D, cov3D, opacity, semantic, out, current_channel
            );
            current_channel += 16;
        }
        else if (left_channels == 13) { // special case for number of classes
            renderCUDA<13><<<blocks_per_grid, threads_per_block>>>(
                N, C, pts, points_int, grid, ranges, point_list,
                means3D, cov3D, opacity, semantic, out, current_channel
            );
            current_channel += 13;
        }
        else if (left_channels >= 8) {
            renderCUDA<8><<<blocks_per_grid, threads_per_block>>>(
                N, C, pts, points_int, grid, ranges, point_list,
                means3D, cov3D, opacity, semantic, out, current_channel
            );
            current_channel += 8;
        }
        else if (left_channels >= 4) {
            renderCUDA<4><<<blocks_per_grid, threads_per_block>>>(
                N, C, pts, points_int, grid, ranges, point_list,
                means3D, cov3D, opacity, semantic, out, current_channel
            );
            current_channel += 4;
        }
        else if (left_channels >= 1) {
            renderCUDA<1><<<blocks_per_grid, threads_per_block>>>(
                N, C, pts, points_int, grid, ranges, point_list,
                means3D, cov3D, opacity, semantic, out, current_channel
            );
            current_channel += 1;
        }
        else {
            // This should never happen - negative channel count
            cudaError_t err = cudaGetLastError();
            printf("ERROR in FORWARD::render: Invalid channel count %d (current_channel=%d, C=%d)\n", 
                   left_channels, current_channel, C);
            cudaDeviceSynchronize();
            return;
        }

        // Check for kernel errors after each launch
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
            cudaDeviceSynchronize();
            return;
        }
    }
}


void FORWARD::preprocess(
	const int P,
	const int* points_xyz,
	const int* radii,
	const dim3 grid,
	uint32_t* tiles_touched)
{
	preprocessCUDA << <(P + 255) / 256, 256 >> > (
		P,
		points_xyz,
		radii,
		grid,
		tiles_touched
	);
}