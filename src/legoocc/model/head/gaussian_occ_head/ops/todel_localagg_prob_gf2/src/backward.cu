#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Perform initial steps for each Gaussian prior to rasterization.
__global__ void preprocessCUDA(
	const int N,
	const int* points_xyz,
	const dim3 grid,
	int* voxel2pts)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= N)
		return;

	int voxel_idx = points_xyz[3 * idx] * grid.y * grid.z + points_xyz[3 * idx + 1] * grid.z + points_xyz[3 * idx + 2];
	voxel2pts[voxel_idx] = idx;
}


// template <uint32_t CHANNELS>
// __global__ void renderCUDA(
//     const int P,
//     const int total_channels,
//     const uint32_t* __restrict__ offsets,
//     const uint32_t* __restrict__ point_list_keys_unsorted,
//     const int* __restrict__ voxel2pts,
//     const float* __restrict__ pts,
//     const float* __restrict__ means3D,
//     const float* __restrict__ cov3D,
//     const float* __restrict__ opas,
//     const float* __restrict__ semantic,        // [P, total_channels]  (per-Gaussian semantics)
//     const float* __restrict__ logits,          // [N, total_channels]  (fw: conditional semantics)
//     const float* __restrict__ bin_logits,      // [N]  (fw: α = 1 - exp(-z))
//     const float* __restrict__ density,         // [N]  (fw: z)  (unused; keep for parity)
//     const float* __restrict__ probability,     // [N]  (fw: z as well)
//     const float* __restrict__ logits_grad,     // [N, total_channels]  (∂L/∂y_c)   optional per call
//     const float* __restrict__ bin_logits_grad, // [N]  (∂L/∂α)
//     const float* __restrict__ density_grad,    // [N]  (∂L/∂z)          optional per call
//     float* __restrict__ means3D_grad,
//     float* __restrict__ opas_grad,
//     float* __restrict__ semantics_grad,
//     float* __restrict__ cov3D_grad,
//     int base_channel
// )
// {
//     const int idx = blockIdx.x * blockDim.x + threadIdx.x;  // gaussian index
//     if (idx >= P) return;

//     const uint32_t start = (idx == 0) ? 0 : offsets[idx - 1];
//     const uint32_t end   = offsets[idx];

//     // local copies of gaussian params
//     const float3 m  = {means3D[3*idx+0], means3D[3*idx+1], means3D[3*idx+2]};
//     const float3 c1 = {cov3D[6*idx+0],   cov3D[6*idx+1],   cov3D[6*idx+2]}; // diag terms
//     const float3 c2 = {cov3D[6*idx+3],   cov3D[6*idx+4],   cov3D[6*idx+5]}; // cross terms
//     const float  opa = opas[idx];

//     // local semantic slice for this template block
//     float sem[CHANNELS] = {0.f};
//     #pragma unroll
//     for (int ch = 0; ch < CHANNELS; ++ch) {
//         const int gch = base_channel + ch;
//         if (gch < total_channels) sem[ch] = semantic[idx * total_channels + gch];
//     }

//     // gradient accumulators (per gaussian)
//     float g_m[3]   = {0.f, 0.f, 0.f};
//     float g_cov[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
//     float g_opa    = 0.f;
//     float g_sem[CHANNELS] = {0.f};

//     const float eps = 1e-9f;

//     // Iterate all pixels this gaussian contributes to
//     for (uint32_t it = start; it < end; ++it) {
//         const int voxel_idx = (int)point_list_keys_unsorted[it];
//         const int pidx = voxel2pts[voxel_idx];   // pixel id
//         if (pidx < 0) continue;

//         // geometry
//         const float3 d = { m.x - pts[3*pidx+0], m.y - pts[3*pidx+1], m.z - pts[3*pidx+2] };
//         const float  quad  = c1.x*d.x*d.x + c1.y*d.y*d.y + c1.z*d.z*d.z;
//         const float  cross = (c2.x*d.x*d.y + c2.y*d.y*d.z + c2.z*d.x*d.z);
//         const float  power = __expf(-0.5f*quad - cross);      // φ_i(x) ≥ 0
//         const float  h_i   = opa * power;

//         // forward aux
//         const float z   = probability[pidx];                  // == density[pidx]
//         const float occ = bin_logits[pidx];                   // α = 1 - exp(-z)

//         // ---- occupancy/density path (ALWAYS contributes) ----
//         // dL/dz = dL/dα * ∂α/∂z  +  dL/dz(extra)
//         float dL_dz = 0.f;
//         if (bin_logits_grad)  dL_dz += bin_logits_grad[pidx] * (1.f - occ); // ∂α/∂z = e^{-z} = 1-α
//         if (density_grad)     dL_dz += density_grad[pidx];

//         // total gradient wrt h_i via z = Σ_j h_j  (∂z/∂h_i = 1)
//         float g_h_total = dL_dz;

//         // ---- semantic path (ONLY if z is large enough) ----
//         if (z > eps && logits_grad) {
//             float g_h_sem = 0.f;
//             #pragma unroll
//             for (int ch = 0; ch < CHANNELS; ++ch) {
//                 const int gch = base_channel + ch;
//                 if (gch < total_channels) {
//                     const float y_c   = logits[pidx * total_channels + gch];     // y_c = C_c / z
//                     const float dL_dy = logits_grad[pidx * total_channels + gch];
//                     // ∂y_c/∂h_i = (sem_i[c] - y_c) / z
//                     g_h_sem += dL_dy * (sem[ch] - y_c) / z;
//                     // ∂y_c/∂sem_i[c] = h_i / z
//                     g_sem[ch] += dL_dy * (h_i / z);
//                 }
//             }
//             g_h_total += g_h_sem;
//         }
//         // NOTE: 当 z <= eps 时，不走语义分支，但 occupancy 仍然把梯度推回给 h_i

//         // ---- chain to opa / power ----
//         g_opa += g_h_total * power;         // ∂h_i/∂opa = power
//         const float dL_dpower = g_h_total * opa;   // ∂h_i/∂power = opa

//         // ---- chain to means & cov via power = exp(arg) ----
//         // arg = -0.5*(c1.x dx^2 + c1.y dy^2 + c1.z dz^2) - (c2.x dx dy + c2.y dy dz + c2.z dx dz)
//         // ∂power/∂arg = power
//         const float g_arg = dL_dpower * power;

//         // means
//         g_m[0] -= g_arg * (c1.x * d.x + c2.x * d.y + c2.z * d.z);
//         g_m[1] -= g_arg * (c2.x * d.x + c1.y * d.y + c2.y * d.z);
//         g_m[2] -= g_arg * (c2.z * d.x + c2.y * d.y + c1.z * d.z);

//         // cov (c1: diag; c2: cross)
//         g_cov[0] += g_arg * (-0.5f * d.x * d.x);
//         g_cov[1] += g_arg * (-0.5f * d.y * d.y);
//         g_cov[2] += g_arg * (-0.5f * d.z * d.z);
//         g_cov[3] += g_arg * (- d.x * d.y);
//         g_cov[4] += g_arg * (- d.y * d.z);
//         g_cov[5] += g_arg * (- d.x * d.z);
//     }

//     // write-back
//     atomicAdd(&means3D_grad[3*idx+0], g_m[0]);
//     atomicAdd(&means3D_grad[3*idx+1], g_m[1]);
//     atomicAdd(&means3D_grad[3*idx+2], g_m[2]);

//     atomicAdd(&opas_grad[idx], g_opa);

//     #pragma unroll
//     for (int ch = 0; ch < CHANNELS; ++ch) {
//         const int gch = base_channel + ch;
//         if (gch < total_channels) {
//             atomicAdd(&semantics_grad[idx * total_channels + gch], g_sem[ch]);
//         }
//     }

//     #pragma unroll
//     for (int k = 0; k < 6; ++k) {
//         atomicAdd(&cov3D_grad[6*idx + k], g_cov[k]);
//     }
// }


// template <uint32_t CHANNELS>
// __global__ void renderCUDA(
//     const int P,
//     const int total_channels,
//     const uint32_t* __restrict__ offsets,
//     const uint32_t* __restrict__ point_list_keys_unsorted,
//     const int* __restrict__ voxel2pts,
//     const float* __restrict__ pts,
//     const float* __restrict__ means3D,
//     const float* __restrict__ cov3D,
//     const float* __restrict__ opas,
//     const float* __restrict__ semantic,   // [P, total_channels]
//     const float* __restrict__ logits,     // [N, total_channels]  (forward output: conditional semantics)
//     const float* __restrict__ bin_logits, // [N]    (forward output: α = 1 - exp(-z))
//     const float* __restrict__ density,    // [N]    (forward output: z)
//     const float* __restrict__ probability,// [N]    (forward output: z as well)
//     const float* __restrict__ logits_grad,// [N, total_channels]  (∂L/∂logits_out)
//     const float* __restrict__ bin_logits_grad, // [N] (∂L/∂α)
//     const float* __restrict__ density_grad,    // [N] (∂L/∂z) optional
//     float* __restrict__ means3D_grad,
//     float* __restrict__ opas_grad,
//     float* __restrict__ semantics_grad,
//     float* __restrict__ cov3D_grad,
//     int base_channel
// )
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x; // gaussian index
//     if (idx >= P) return;

//     uint32_t start = (idx == 0) ? 0 : offsets[idx - 1];
//     uint32_t end   = offsets[idx];

//     // local copies
//     const float3 m   = {means3D[3*idx+0], means3D[3*idx+1], means3D[3*idx+2]};
//     const float3 c1  = {cov3D[6*idx+0],   cov3D[6*idx+1],   cov3D[6*idx+2]};
//     const float3 c2  = {cov3D[6*idx+3],   cov3D[6*idx+4],   cov3D[6*idx+5]};
//     const float opa  = opas[idx];

//     float sem[CHANNELS] = {0.f};
//     #pragma unroll
//     for (int ch = 0; ch < CHANNELS; ++ch) {
//         int gch = base_channel + ch;
//         if (gch < total_channels) sem[ch] = semantic[idx * total_channels + gch];
//     }

//     // grads accumulators (w.r.t. gaussian i)
//     float g_m[3]    = {0.f, 0.f, 0.f};
//     float g_cov[6]  = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
//     float g_opa     = 0.f;
//     float g_sem[CHANNELS] = {0.f};

//     // Iterate all pixels this gaussian contributes to
//     for (uint32_t it = start; it < end; ++it) {
//         int voxel_idx = (int)point_list_keys_unsorted[it];
//         int pidx = voxel2pts[voxel_idx];         // pixel id
//         if (pidx < 0) continue;

//         // geometry
//         float3 d = { m.x - pts[3*pidx+0], m.y - pts[3*pidx+1], m.z - pts[3*pidx+2] };
//         float quad  = c1.x*d.x*d.x + c1.y*d.y*d.y + c1.z*d.z*d.z;
//         float cross = (c2.x*d.x*d.y + c2.y*d.y*d.z + c2.z*d.x*d.z);
//         float power = __expf(-0.5f*quad - cross);     // φ_i(x) ≥ 0

//         // read forward outputs/aux
//         float z    = probability[pidx];               // == density[pidx]
//         float occ  = bin_logits[pidx];                // α = 1 - exp(-z)
//         const float eps = 1e-9f;

//         if (z <= eps) continue;                       // no gradient if no hazard

//         // h_i and channel-wise logits at pixel p
//         float h_i = opa * power;

//         // ---------- gradients from semantic path ----------
//         // dL/dh_i_sem = sum_c ( dL/dy_c * ∂y_c/∂h_i ) where y_c = logits[p,c]
//         float g_h_sem = 0.f;
//         #pragma unroll
//         for (int ch = 0; ch < CHANNELS; ++ch) {
//             int gch = base_channel + ch;
//             if (gch < total_channels) {
//                 float y_c   = logits[pidx * total_channels + gch];   // \hat y_c
//                 float dL_dy = logits_grad[pidx * total_channels + gch];
//                 // ∂y_c/∂h_i = (sem_i[c] - y_c) / z
//                 g_h_sem += dL_dy * (sem[ch] - y_c) / z;
//                 // ∂y_c/∂sem_i[c] = h_i / z
//                 g_sem[ch] += dL_dy * (h_i / z);
//             }
//         }

//         // ---------- gradients from occupancy & density ----------
//         // α = 1 - exp(-z) ⇒ ∂α/∂z = exp(-z) = 1 - α
//         float dL_dz = 0.f;
//         dL_dz += bin_logits_grad[pidx] * (1.f - occ);  // = e^{-z}
//         dL_dz += density_grad[pidx];                   // if you supervise z directly

//         // total dL/dh_i  (since z = sum_j h_j ⇒ ∂z/∂h_i = 1)
//         float g_h_total = g_h_sem + dL_dz;

//         // ---------- chain to opa and power ----------
//         // h_i = opa * power
//         g_opa   += g_h_total * power;
//         float dL_dpower = g_h_total * opa;

//         // ---------- chain to means & cov via power = exp(arg) ----------
//         // let arg = -0.5*d^T diag(c1) d - d^T C2 d  (where C2 carries cross terms)
//         // ∂power/∂arg = power
//         float g_arg = dL_dpower * power;

//         // means
//         g_m[0] -= g_arg * (c1.x * d.x + c2.x * d.y + c2.z * d.z);
//         g_m[1] -= g_arg * (c2.x * d.x + c1.y * d.y + c2.y * d.z);
//         g_m[2] -= g_arg * (c2.z * d.x + c2.y * d.y + c1.z * d.z);

//         // cov (c1: diag; c2: cross)
//         g_cov[0] += g_arg * (-0.5f * d.x * d.x);
//         g_cov[1] += g_arg * (-0.5f * d.y * d.y);
//         g_cov[2] += g_arg * (-0.5f * d.z * d.z);
//         g_cov[3] += g_arg * (- d.x * d.y);
//         g_cov[4] += g_arg * (- d.y * d.z);
//         g_cov[5] += g_arg * (- d.x * d.z);
//     }

//     // write-back
//     atomicAdd(&means3D_grad[3*idx+0], g_m[0]);
//     atomicAdd(&means3D_grad[3*idx+1], g_m[1]);
//     atomicAdd(&means3D_grad[3*idx+2], g_m[2]);

//     atomicAdd(&opas_grad[idx], g_opa);

//     #pragma unroll
//     for (int ch = 0; ch < CHANNELS; ++ch) {
//         int gch = base_channel + ch;
//         if (gch < total_channels)
//             atomicAdd(&semantics_grad[idx * total_channels + gch], g_sem[ch]);
//     }

//     #pragma unroll
//     for (int k = 0; k < 6; ++k)
//         atomicAdd(&cov3D_grad[6*idx + k], g_cov[k]);
// }


template <uint32_t CHANNELS>
__global__ void renderCUDA(
    const int P,
    const int total_channels, // total number of channels
    const uint32_t* __restrict__ offsets,
    const uint32_t* __restrict__ point_list_keys_unsorted,
    const int* __restrict__ voxel2pts,
    const float* __restrict__ pts,
    const float* __restrict__ means3D,
    const float* __restrict__ cov3D,
    const float* __restrict__ opas,
    const float* __restrict__ semantic,
    const float* __restrict__ logits,
    const float* __restrict__ bin_logits,
    const float* __restrict__ density,
    const float* __restrict__ probability,
    const float* __restrict__ logits_grad,
    const float* __restrict__ bin_logits_grad,
    const float* __restrict__ density_grad,
    float* __restrict__ means3D_grad,
    float* __restrict__ opas_grad,
    float* __restrict__ semantics_grad,
    float* __restrict__ cov3D_grad,
    int base_channel // current group channel offset
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P) return;

    uint32_t start = (idx == 0) ? 0 : offsets[idx - 1];
    uint32_t end = offsets[idx];

    const float3 means = {means3D[3 * idx], means3D[3 * idx + 1], means3D[3 * idx + 2]};
    const float3 cov1 = {cov3D[6 * idx], cov3D[6 * idx + 1], cov3D[6 * idx + 2]};
    const float3 cov2 = {cov3D[6 * idx + 3], cov3D[6 * idx + 4], cov3D[6 * idx + 5]};
    const float opa = opas[idx];
    float sem[CHANNELS] = {0};
    for (int ch = 0; ch < CHANNELS; ch++) {
        int global_ch = base_channel + ch;
        if (global_ch < total_channels)
            sem[ch] = semantic[idx * total_channels + global_ch];
    }

    float means_grad[3] = {0};
    float opa_grad = 0;
    float semantic_grad[CHANNELS] = {0};
    float cov_grad[6] = {0};

    for (int i = start; i < end; i++) {
        int voxel_idx = point_list_keys_unsorted[i];
        int pts_idx = voxel2pts[voxel_idx];
        if (pts_idx >= 0) {
            float3 d = {means.x - pts[pts_idx * 3],
                        means.y - pts[pts_idx * 3 + 1],
                        means.z - pts[pts_idx * 3 + 2]};
            float power = cov1.x * d.x * d.x + cov1.y * d.y * d.y + cov1.z * d.z * d.z;
            power = -0.5f * power - (cov2.x * d.x * d.y + cov2.y * d.y * d.z + cov2.z * d.x * d.z);
            power = exp(power);
            float deter = cov1.x * cov1.y * cov1.z + 2 * cov2.x * cov2.y * cov2.z
                          - cov1.x * cov2.y * cov2.y - cov1.y * cov2.z * cov2.z - cov1.z * cov2.x * cov2.x;
            float prob = powf(2 * 3.1415926535f, -1.5f) * powf(deter, 0.5f) * power;
            float power_grad = 0.f;
            float deter_grad = 0.f;
            float prob_grad = 0.f;
            float prob_sum = probability[pts_idx];

            if (prob_sum > 1e-9f) {
                for (int ch = 0; ch < CHANNELS; ch++) {
                    int global_ch = base_channel + ch;
                    if (global_ch < total_channels) {
                        semantic_grad[ch] += logits_grad[pts_idx * total_channels + global_ch] * prob * opa / prob_sum;
                        prob_grad += logits_grad[pts_idx * total_channels + global_ch] * (sem[ch] - logits[pts_idx * total_channels + global_ch]) * opa / prob_sum;
                        opa_grad += logits_grad[pts_idx * total_channels + global_ch] * (sem[ch] - logits[pts_idx * total_channels + global_ch]) * prob / prob_sum;
                    }
                }
            }
            power_grad += prob_grad * powf(2 * 3.1415926535f, -1.5f) * powf(deter, 0.5f);
            power_grad += (1.0f - bin_logits[pts_idx]) / (1.0f - power + 1e-9f) * bin_logits_grad[pts_idx];
            power_grad += density_grad[pts_idx];
            deter_grad += prob_grad * prob / 2.0f / deter;

            means_grad[0] -= power_grad * power * (cov1.x * d.x + cov2.x * d.y + cov2.z * d.z);
            means_grad[1] -= power_grad * power * (cov2.x * d.x + cov1.y * d.y + cov2.y * d.z);
            means_grad[2] -= power_grad * power * (cov2.z * d.x + cov2.y * d.y + cov1.z * d.z);

            cov_grad[0] += power_grad * power * (-0.5f * d.x * d.x) + deter_grad * (cov1.y * cov1.z - cov2.y * cov2.y);
            cov_grad[1] += power_grad * power * (-0.5f * d.y * d.y) + deter_grad * (cov1.x * cov1.z - cov2.z * cov2.z);
            cov_grad[2] += power_grad * power * (-0.5f * d.z * d.z) + deter_grad * (cov1.x * cov1.y - cov2.x * cov2.x);
            cov_grad[3] += power_grad * power * (-d.x * d.y) + 2.0f * deter_grad * (cov2.y * cov2.z - cov1.z * cov2.x);
            cov_grad[4] += power_grad * power * (-d.y * d.z) + 2.0f * deter_grad * (cov2.x * cov2.z - cov1.x * cov2.y);
            cov_grad[5] += power_grad * power * (-d.x * d.z) + 2.0f * deter_grad * (cov2.x * cov2.y - cov1.y * cov2.z);
        }
    }

    means3D_grad[idx * 3 + 0] += means_grad[0];
    means3D_grad[idx * 3 + 1] += means_grad[1];
    means3D_grad[idx * 3 + 2] += means_grad[2];
    opas_grad[idx] += opa_grad;
    for (int ch = 0; ch < CHANNELS; ch++) {
        int global_ch = base_channel + ch;
        if (global_ch < total_channels)
            semantics_grad[idx * total_channels + global_ch] += semantic_grad[ch];
    }
    for (int ch = 0; ch < 6; ch++)
        cov3D_grad[idx * 6 + ch] += cov_grad[ch];
}



void BACKWARD::render(
    const int P,
    const int C, // total channels
    const uint32_t* offsets,
    const uint32_t* point_list_keys_unsorted,
    const int* voxel2pts,
    const float* pts,
    const float* means3D,
    const float* cov3D,
    const float* opas,
    const float* semantic,
    const float* logits,
    const float* bin_logits,
    const float* density,
    const float* probability,
    const float* logits_grad,
    const float* bin_logits_grad,
    const float* density_grad,
    float* means3D_grad,
    float* opas_grad,
    float* semantics_grad,
    float* cov3D_grad
) {
    int threads_per_block = 256;
    int blocks_per_grid = (P + threads_per_block - 1) / threads_per_block;
    int current_channel = 0;

    while (current_channel < C) {
        int left_channels = C - current_channel;

        if (left_channels >= 128) {
            renderCUDA<128><<<blocks_per_grid, threads_per_block>>>(
                P, C, offsets, point_list_keys_unsorted, voxel2pts, pts, means3D, cov3D, opas,
                semantic, logits, bin_logits, density, probability, logits_grad, bin_logits_grad, density_grad,
                means3D_grad, opas_grad, semantics_grad, cov3D_grad, current_channel
            );
            current_channel += 128;
        }
        else if (left_channels >= 64) {
            renderCUDA<64><<<blocks_per_grid, threads_per_block>>>(
                P, C, offsets, point_list_keys_unsorted, voxel2pts, pts, means3D, cov3D, opas,
                semantic, logits, bin_logits, density, probability, logits_grad, bin_logits_grad, density_grad,
                means3D_grad, opas_grad, semantics_grad, cov3D_grad, current_channel
            );
            current_channel += 64;
        }
        else if (left_channels >= 32) {
            renderCUDA<32><<<blocks_per_grid, threads_per_block>>>(
                P, C, offsets, point_list_keys_unsorted, voxel2pts, pts, means3D, cov3D, opas,
                semantic, logits, bin_logits, density, probability, logits_grad, bin_logits_grad, density_grad,
                means3D_grad, opas_grad, semantics_grad, cov3D_grad, current_channel
            );
            current_channel += 32;
        }
        else if (left_channels >= 16) {
            renderCUDA<16><<<blocks_per_grid, threads_per_block>>>(
                P, C, offsets, point_list_keys_unsorted, voxel2pts, pts, means3D, cov3D, opas,
                semantic, logits, bin_logits, density, probability, logits_grad, bin_logits_grad, density_grad,
                means3D_grad, opas_grad, semantics_grad, cov3D_grad, current_channel
            );
            current_channel += 16;
        }
        else if (left_channels == 13) {
            renderCUDA<13><<<blocks_per_grid, threads_per_block>>>(
                P, C, offsets, point_list_keys_unsorted, voxel2pts, pts, means3D, cov3D, opas,
                semantic, logits, bin_logits, density, probability, logits_grad, bin_logits_grad, density_grad,
                means3D_grad, opas_grad, semantics_grad, cov3D_grad, current_channel
            );
            current_channel += 13;
        }
        else if (left_channels >= 8) {
            renderCUDA<8><<<blocks_per_grid, threads_per_block>>>(
                P, C, offsets, point_list_keys_unsorted, voxel2pts, pts, means3D, cov3D, opas,
                semantic, logits, bin_logits, density, probability, logits_grad, bin_logits_grad, density_grad,
                means3D_grad, opas_grad, semantics_grad, cov3D_grad, current_channel
            );
            current_channel += 8;
        }
        else if (left_channels >= 4) {
            renderCUDA<4><<<blocks_per_grid, threads_per_block>>>(
                P, C, offsets, point_list_keys_unsorted, voxel2pts, pts, means3D, cov3D, opas,
                semantic, logits, bin_logits, density, probability, logits_grad, bin_logits_grad, density_grad,
                means3D_grad, opas_grad, semantics_grad, cov3D_grad, current_channel
            );
            current_channel += 4;
        }
        else if (left_channels >= 1) {
            renderCUDA<1><<<blocks_per_grid, threads_per_block>>>(
                P, C, offsets, point_list_keys_unsorted, voxel2pts, pts, means3D, cov3D, opas,
                semantic, logits, bin_logits, density, probability, logits_grad, bin_logits_grad, density_grad,
                means3D_grad, opas_grad, semantics_grad, cov3D_grad, current_channel
            );
            current_channel += 1;
        }
        else {
            printf("ERROR in BACKWARD::render: Invalid channel count %d (current_channel=%d, C=%d)\n",
                   left_channels, current_channel, C);
            cudaDeviceSynchronize();
            return;
        }

        // Error check
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
            cudaDeviceSynchronize();
            return;
        }
    }
    // (Optional: cudaDeviceSynchronize() for debugging)
}

void BACKWARD::preprocess(
	const int N,
	const int* points_xyz,
	const dim3 grid,
	int* voxel2pts)
{
	preprocessCUDA << <(N + 255) / 256, 256 >> > (
		N,
		points_xyz,
		grid,
		voxel2pts
	);
}