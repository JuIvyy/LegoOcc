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


// template <uint32_t CHANNELS>
// __global__ void renderCUDA(
//     const int N,
//     const int total_channels,
//     const float* __restrict__ pts,
//     const int* __restrict__ points_int,
//     const dim3 grid,
//     const uint2* __restrict__ ranges,
//     const uint32_t* __restrict__ point_list,
//     const float* __restrict__ means3D,
//     const float* __restrict__ cov3D,
//     const float* __restrict__ opas,
//     const float* __restrict__ semantic,
//     float* __restrict__ out_logits,       // [N, total_channels] : conditional semantics p(c|occ)
//     float* __restrict__ out_bin_logits,   // [N] : occupancy prob α(x) = 1-exp(-z)
//     float* __restrict__ out_density,      // [N] : z (total hazard / intensity)
//     float* __restrict__ out_probability,  // [N] : keep as z for backward compatibility
//     int base_channel
// )
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= N) return;

//     const int* point_int = points_int + idx * 3;
//     int voxel_idx = point_int[0] * grid.y * grid.z + point_int[1] * grid.z + point_int[2];
//     float3 point = {pts[3 * idx + 0], pts[3 * idx + 1], pts[3 * idx + 2]};
//     uint2 range = ranges[voxel_idx];

//     // Accumulators
//     float C[CHANNELS] = {0.f};   // numerator: sum_i h_i * semantic_i
//     float z = 0.f;               // hazard sum: z = sum_i (opa_i * power_i)

//     for (int i = range.x; i < range.y; ++i) {
//         int gs_idx = point_list[i];

//         // Gaussian kernel value φ_i(x) = power
//         float3 cov1 = { cov3D[gs_idx * 6 + 0], cov3D[gs_idx * 6 + 1], cov3D[gs_idx * 6 + 2] };
//         float3 cov2 = { cov3D[gs_idx * 6 + 3], cov3D[gs_idx * 6 + 4], cov3D[gs_idx * 6 + 5] };
//         float3 d = { means3D[gs_idx * 3 + 0] - point.x,
//                      means3D[gs_idx * 3 + 1] - point.y,
//                      means3D[gs_idx * 3 + 2] - point.z };

//         float quad = cov1.x * d.x * d.x + cov1.y * d.y * d.y + cov1.z * d.z * d.z;
//         float cross = (cov2.x * d.x * d.y + cov2.y * d.y * d.z + cov2.z * d.x * d.z);
//         float power = __expf(-0.5f * quad - cross);      // φ_i(x) ≥ 0

//         float h_i = opas[gs_idx] * power;                // hazard contribution h_i = opa * φ

//         // accumulate hazard and semantic numerator
//         z += h_i;
//         #pragma unroll
//         for (int ch = 0; ch < CHANNELS; ++ch) {
//             int gch = base_channel + ch;
//             if (gch < total_channels) {
//                 C[ch] += semantic[gs_idx * total_channels + gch] * h_i;
//             }
//         }
//     }

//     // Produce outputs
//     const float eps = 1e-9f;
//     if (z > eps) {
//         #pragma unroll
//         for (int ch = 0; ch < CHANNELS; ++ch) {
//             int gch = base_channel + ch;
//             if (gch < total_channels) {
//                 out_logits[idx * total_channels + gch] = C[ch] / z;  // conditional semantics p(c|occ)
//             }
//         }
//     } else {
//         // fallback: uniform when no hazard
//         #pragma unroll
//         for (int ch = 0; ch < CHANNELS; ++ch) {
//             int gch = base_channel + ch;
//             if (gch < total_channels) {
//                 out_logits[idx * total_channels + gch] = 1.0f / (float)total_channels;
//             }
//         }
//     }

//     if (base_channel == 0) {
//         // α(x) = 1 - exp(-z)   (survival: S=exp(-H) ⇒ 1-S)
//         float occ = 1.f - __expf(-z);
//         out_bin_logits[idx] = occ;     // in [0,1]
//         out_density[idx]    = z;       // store z for diagnostics/loss
//         out_probability[idx]= z;       // keep z here so backward can read it as "probability"
//     }
// }

template <uint32_t CHANNELS>
__global__ void renderCUDA(
    const int N,
    const int total_channels,
    const float* __restrict__ pts,
    const int* __restrict__ points_int,
    const dim3 grid,
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ point_list,
    const float* __restrict__ means3D,
    const float* __restrict__ cov3D,
    const float* __restrict__ opas,
    const float* __restrict__ semantic,
    float* __restrict__ out_logits,       // [N, C] : p(c | occupied)
    float* __restrict__ out_bin_logits,   // [N]    : o = 1 - T
    float* __restrict__ out_density,      // [N]    : z = sum alpha_i
    float* __restrict__ out_probability,  // [N]    : T = prod (1 - alpha_i)
    int base_channel
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const int* pi = points_int + idx * 3;
    int voxel_idx = pi[0] * grid.y * grid.z + pi[1] * grid.z + pi[2];
    float3 x = {pts[3*idx+0], pts[3*idx+1], pts[3*idx+2]};
    uint2 range = ranges[voxel_idx];

    float C[CHANNELS] = {0.f};  // numerator for semantics
    float z = 0.f;              // z = sum alpha_i
    float logT = 0.f;           // accumulate log(1 - alpha_i) for stability
    const float eps = 1e-9f;

    for (int k = range.x; k < range.y; ++k) {
        int gi = point_list[k];

        float3 c1 = {cov3D[6*gi+0], cov3D[6*gi+1], cov3D[6*gi+2]};
        float3 c2 = {cov3D[6*gi+3], cov3D[6*gi+4], cov3D[6*gi+5]};
        float3 d  = {means3D[3*gi+0] - x.x,
                     means3D[3*gi+1] - x.y,
                     means3D[3*gi+2] - x.z};

        float quad  = c1.x*d.x*d.x + c1.y*d.y*d.y + c1.z*d.z*d.z;
        float cross = c2.x*d.x*d.y + c2.y*d.y*d.z + c2.z*d.x*d.z;
        float phi   = __expf(-0.5f*quad - cross);        // φ_i(x) ≥ 0

        float alpha = opas[gi] * phi;                    // alpha_i = a_i * φ_i
        alpha = fminf(fmaxf(alpha, 0.f), 1.f - 1e-6f);   // clamp to (0,1)

        z += alpha;
        logT += __logf(fmaxf(1.f - alpha, 1e-6f));

        #pragma unroll
        for (int ch = 0; ch < CHANNELS; ++ch) {
            int gch = base_channel + ch;
            if (gch < total_channels) {
                C[ch] += semantic[gi * total_channels + gch] * alpha;
            }
        }
    }

    // conditional semantics
    if (z > eps) {
        #pragma unroll
        for (int ch = 0; ch < CHANNELS; ++ch) {
            int gch = base_channel + ch;
            if (gch < total_channels)
                out_logits[idx * total_channels + gch] = C[ch] / z;
        }
    } else {
        #pragma unroll
        for (int ch = 0; ch < CHANNELS; ++ch) {
            int gch = base_channel + ch;
            if (gch < total_channels)
                out_logits[idx * total_channels + gch] = 1.f / (float)total_channels;
        }
    }

    if (base_channel == 0) {
        float T = __expf(logT);      // total transmittance
        float occ = 1.f - T;         // o = 1 - Π(1-α_i)

        out_bin_logits[idx] = occ;
        out_density[idx]    = z;     // sum α_i
        out_probability[idx]= T;     // store T for backward
    }
}


// // Main rasterization method. Collaboratively works on one tile per
// // block, each thread treats one pixel. Alternates between fetching 
// // and rasterizing data.
// template <uint32_t CHANNELS>
// __global__ void renderCUDA(
//     const int N,
//     const int total_channels,                        // total number of channels
//     const float* __restrict__ pts,
//     const int* __restrict__ points_int,
//     const dim3 grid,
//     const uint2* __restrict__ ranges,
//     const uint32_t* __restrict__ point_list,
//     const float* __restrict__ means3D,
//     const float* __restrict__ cov3D,
//     const float* __restrict__ opas,
//     const float* __restrict__ semantic,
//     float* __restrict__ out_logits,
//     float* __restrict__ out_bin_logits,
//     float* __restrict__ out_density,
//     float* __restrict__ out_probability,
//     int base_channel                                  // channel offset for current group
// )
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= N) return;

//     const int* point_int = points_int + idx * 3;
//     int voxel_idx = point_int[0] * grid.y * grid.z + point_int[1] * grid.z + point_int[2];
//     float3 point = {pts[3 * idx], pts[3 * idx + 1], pts[3 * idx + 2]};
//     uint2 range = ranges[voxel_idx];

//     float C[CHANNELS] = { 0 };
//     float bin_logit = 1.0f;
//     float density = 0.0f;
//     float prob_sum = 0.0f;

//     for (int i = range.x; i < range.y; i++) {
//         int gs_idx = point_list[i];
//         float3 cov1 = { cov3D[gs_idx * 6 + 0], cov3D[gs_idx * 6 + 1], cov3D[gs_idx * 6 + 2] };
//         float3 cov2 = { cov3D[gs_idx * 6 + 3], cov3D[gs_idx * 6 + 4], cov3D[gs_idx * 6 + 5] };
//         float3 d = { means3D[gs_idx * 3] - point.x, means3D[gs_idx * 3 + 1] - point.y, means3D[gs_idx * 3 + 2] - point.z };
//         float power = cov1.x * d.x * d.x + cov1.y * d.y * d.y + cov1.z * d.z * d.z;
//         // alpha in eq(4)
//         power = -0.5f * power - (cov2.x * d.x * d.y + cov2.y * d.y * d.z + cov2.z * d.x * d.z);
//         power = exp(power);
//         float deter = cov1.x * cov1.y * cov1.z + 2 * cov2.x * cov2.y * cov2.z
//                       - cov1.x * cov2.y * cov2.y - cov1.y * cov2.z * cov2.z - cov1.z * cov2.x * cov2.x;
//         // p in eq(7) * opacity
//         float prob = powf(2 * 3.1415926535f, -1.5f) * powf(deter, 0.5f) * power * opas[gs_idx];

//         for (int ch = 0; ch < CHANNELS; ch++) {
//             int global_ch = base_channel + ch;
//             if (global_ch < total_channels) {
//                 C[ch] += semantic[total_channels * gs_idx + global_ch] * prob;
//             }
//         }
//         bin_logit = (1 - power) * bin_logit;
//         density += power;
//         prob_sum += prob;
//     }

//     // Write output for this group of channels
//     if (prob_sum > 1e-9f) {
//         for (int ch = 0; ch < CHANNELS; ch++) {
//             int global_ch = base_channel + ch;
//             if (global_ch < total_channels)
//                 out_logits[idx * total_channels + global_ch] = C[ch] / prob_sum;
//         }
//     } else {
//         for (int ch = 0; ch < CHANNELS; ch++) {
//             int global_ch = base_channel + ch;
//             if (global_ch < total_channels)
//                 out_logits[idx * total_channels + global_ch] = 1.0f / float(total_channels);
//         }
//     }
//     // Write other scalars (only once per pixel)
//     if (base_channel == 0) {
//         out_bin_logits[idx] = 1.0f - bin_logit;
//         out_density[idx] = density;
//         out_probability[idx] = prob_sum;
//     }
// }


void FORWARD::render(
    const int N,
    const int C,  // total number of channels
    const float* pts,
    const int* points_int,
    const dim3 grid,
    const uint2* ranges,
    const uint32_t* point_list,
    const float* means3D,
    const float* cov3D,
    const float* opas,
    const float* semantic,
    float* out_logits,
    float* out_bin_logits,
    float* out_density,
    float* out_probability
) {
    const int TEMPLATE_CHANNELS = 128; // block size for template kernel
    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
    int current_channel = 0;

    while (current_channel < C) {
        int left_channels = C - current_channel;

        if (left_channels >= 128) {
            renderCUDA<128><<<blocks_per_grid, threads_per_block>>>(
                N, C, pts, points_int, grid, ranges, point_list, means3D, cov3D, opas, semantic,
                out_logits, out_bin_logits, out_density, out_probability, current_channel
            );
            current_channel += 128;
        }
        else if (left_channels >= 64) {
            renderCUDA<64><<<blocks_per_grid, threads_per_block>>>(
                N, C, pts, points_int, grid, ranges, point_list, means3D, cov3D, opas, semantic,
                out_logits, out_bin_logits, out_density, out_probability, current_channel
            );
            current_channel += 64;
        }
        else if (left_channels >= 32) {
            renderCUDA<32><<<blocks_per_grid, threads_per_block>>>(
                N, C, pts, points_int, grid, ranges, point_list, means3D, cov3D, opas, semantic,
                out_logits, out_bin_logits, out_density, out_probability, current_channel
            );
            current_channel += 32;
        }
        else if (left_channels >= 16) {
            renderCUDA<16><<<blocks_per_grid, threads_per_block>>>(
                N, C, pts, points_int, grid, ranges, point_list, means3D, cov3D, opas, semantic,
                out_logits, out_bin_logits, out_density, out_probability, current_channel
            );
            current_channel += 16;
        }
        else if (left_channels == 13) {  // special case for number of classes
            renderCUDA<13><<<blocks_per_grid, threads_per_block>>>(
                N, C, pts, points_int, grid, ranges, point_list, means3D, cov3D, opas, semantic,
                out_logits, out_bin_logits, out_density, out_probability, current_channel
            );
            current_channel += 13;
        }
        else if (left_channels >= 8) {
            renderCUDA<8><<<blocks_per_grid, threads_per_block>>>(
                N, C, pts, points_int, grid, ranges, point_list, means3D, cov3D, opas, semantic,
                out_logits, out_bin_logits, out_density, out_probability, current_channel
            );
            current_channel += 8;
        }
        else if (left_channels >= 4) {
            renderCUDA<4><<<blocks_per_grid, threads_per_block>>>(
                N, C, pts, points_int, grid, ranges, point_list, means3D, cov3D, opas, semantic,
                out_logits, out_bin_logits, out_density, out_probability, current_channel
            );
            current_channel += 4;
        }
        else if (left_channels >= 2) {
            renderCUDA<2><<<blocks_per_grid, threads_per_block>>>(
                N, C, pts, points_int, grid, ranges, point_list, means3D, cov3D, opas, semantic,
                out_logits, out_bin_logits, out_density, out_probability, current_channel
            );
            current_channel += 2;
        }
        else if (left_channels >= 1) {
            renderCUDA<1><<<blocks_per_grid, threads_per_block>>>(
                N, C, pts, points_int, grid, ranges, point_list, means3D, cov3D, opas, semantic,
                out_logits, out_bin_logits, out_density, out_probability, current_channel
            );
            current_channel += 1;
        }
        else {
            // Error handling for unexpected cases
            cudaError_t err = cudaGetLastError();
            printf("ERROR in FORWARD::render: Invalid channel count %d (current_channel=%d, C=%d)\n",
                   left_channels, current_channel, C);
            cudaDeviceSynchronize();
            return;
        }

        // Error checking after kernel launch
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