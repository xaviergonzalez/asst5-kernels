#include <torch/extension.h>
#include <cuda_runtime.h>

// 
// Create your kernel functions here
// Example: __global__ void kernel(...) { ... }
//

struct GlobalConstants
{
    float c0, c1, c2, c3, c4;
    float inv_hx2, inv_hy2, inv_hz2;
    float dt;
    int Nz, Ny, Nx;
    int r;
    
};
__constant__ GlobalConstants cuConstParams;

__global__ void compute_laplacian(
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> u,
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> lap,
)
{
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;
    int tz = threadIdx.z + blockIdx.z * blockDim.z;

    // Boundary checks
    if (tx >= Nx - cuConstParams.r || ty >= Ny - cuConstParams.r || tz >= Nz - cuConstParams.r || tx < cuConstParams.r || ty < cuConstParams.r || tz < cuConstParams.r)
        return;

    // Compute 8th-order Laplacian here using stencil coefficients
    // and store result in lap
    float u_xx = (cuConstParams.c0 * u[tz, ty, tx] + 
    cuConstParams.c1 * (u[tz, ty, tx-1] + u[tz, ty, tx+1]) + 
    cuConstParams.c2 * (u[tz, ty, tx-2] + u[tz, ty, tx+2]) + 
    cuConstParams.c3 * (u[tz, ty, tx-3] + u[tz, ty, tx+3]) + 
    cuConstParams.c4 * (u[tz, ty, tx-4] + u[tz, ty, tx+4])) * cuConstParams.inv_hx2;
    float u_yy = (cuConstParams.c0 * u[tz, ty, tx] +
                  cuConstParams.c1 * (u[tz, ty - 1, tx] + u[tz, ty + 1, tx]) +
                  cuConstParams.c2 * (u[tz, ty - 2, tx] + u[tz, ty + 2, tx]) +
                  cuConstParams.c3 * (u[tz, ty - 3, tx] + u[tz, ty + 3, tx]) +
                  cuConstParams.c4 * (u[tz, ty - 4, tx] + u[tz, ty + 4, tx])) *
                 cuConstParams.inv_hy2;
    float u_zz = (cuConstParams.c0 * u[tz, ty, tx] +
                  cuConstParams.c1 * (u[tz-1, ty, tx] + u[tz+1, ty, tx]) +
                  cuConstParams.c2 * (u[tz-2, ty, tx] + u[tz+2, ty, tx]) +
                  cuConstParams.c3 * (u[tz-3, ty, tx] + u[tz+3, ty, tx]) +
                  cuConstParams.c4 * (u[tz-4, ty, tx] + u[tz+4, ty, tx])) *
                 cuConstParams.inv_hz2;
    lap[tz, ty, tx] = u_xx + u_yy + u_zz;
}

__global__ void rk4(
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> u_new,
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> u_old,
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> lap)
{
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;
    int tz = threadIdx.z + blockIdx.z * blockDim.z;

    // Boundary checks
    if (tx >= Nx - cuConstParams.r || ty >= Ny - cuConstParams.r || tz >= Nz - cuConstParams.r || tx < cuConstParams.r || ty < cuConstParams.r || tz < cuConstParams.r)
        return;
    // never implemented bc I realized we could fuse
}

__global__ void fused_kernel(
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> u_new,
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> u_old)
{
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;
    int tz = threadIdx.z + blockIdx.z * blockDim.z;
    // Boundary checks (only update interior points)
    if (tx >= Nx - cuConstParams.r || ty >= Ny - cuConstParams.r || tz >= Nz - cuConstParams.r || tx < cuConstParams.r || ty < cuConstParams.r || tz < cuConstParams.r)
        return;
    // Laplacian computation
    
    float lap = u_xx + u_yy + u_zz;
    // RK4 update
}

// Host function to launch kernel
torch::Tensor custom_kernel(
    torch::Tensor u0,
    float alpha,
    float hx,
    float hy,
    float hz,
    int n_steps
) {
    TORCH_CHECK(u0.device().is_cuda(), "Tensor u0 must be a CUDA tensor");
    TORCH_CHECK(u0.scalar_type() == torch::kFloat32, "u0 must be float32");
    TORCH_CHECK(u0.dim() == 3, "u0 must have shape (Nz, Ny, Nx)");

    const int Nz = u0.size(0);
    const int Ny = u0.size(1);
    const int Nx = u0.size(2);
    
    if (Nx < 9 || Ny < 9 || Nz < 9) {
        throw std::runtime_error("All dimensions must be >= 9 for radius-4 stencil.");
    }

    // 3D 8th-order 2nd-derivative Laplacian coefficients
    float c0 = -205.0 / 72.0
    float c1 =   8.0  /  5.0
    float c2 =  -1.0  /  5.0
    float c3 =   8.0  / 315.0
    float c4 =  -1.0  / 560.0

    // CFL stability (same constant, but now for RK4)
    float c = 0.05

    float inv_hx2 = 1.0 / (hx * hx)
    float inv_hy2 = 1.0 / (hy * hy)
    float inv_hz2 = 1.0 / (hz * hz)

    float S  = inv_hx2 + inv_hy2 + inv_hz2
    float dt = c / (alpha * S)

    // Radius of stencil
    int r = 4

    GlobalConstants params;
    params.c0 = c0;
    params.c1 = c1;
    params.c2 = c2;
    params.c3 = c3;
    params.c4 = c4;
    params.inv_hx2 = inv_hx2;
    params.inv_hy2 = inv_hy2;
    params.inv_hz2 = inv_hz2;
    params.dt = dt;
    params.Nz = Nz;
    params.Ny = Ny;
    params.Nx = Nx;
    params.r = r;

    cudaMemcpyToSymbol(cuConstParams, &params, sizeof(GlobalConstants));


    // Allocate output tensor (or reuse u0 for in-place)
    u0 = u0.contiguous();
    torch::Tensor lap = torch::zeros_like(u0);
    lap = lap.contiguous();
    torch::Tensor result = u0.clone();  // TODO: Modify as needed
    auto u_acc = u0.packed_accessor32<float, 3, torch::RestrictPtrTraits>();
    auto res_acc = result.packed_accessor32<float, 3, torch::RestrictPtrTraits>();
    auto lap_acc = lap.packed_accessor32<float, 3, torch::RestrictPtrTraits>();
    // const float *d_u0 = u0.data_ptr<float>();
    // float *d_result = result.data_ptr<float>();

    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks(
        (Nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (Ny + threadsPerBlock.y - 1) / threadsPerBlock.y,
        (Nz + threadsPerBlock.z - 1) / threadsPerBlock.z);

    ////
    // Launch your kernel here
    ////
    for _ in range(n_steps):
        compute_laplacian<<<numBlocks, threadsPerBlock>>>(
            u_acc,
            lap_acc,
        );
        rk4<<<numBlocks, threadsPerBlock>>>(
            res_acc,
            u_acc,
        );

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    // Synchronize to ensure kernel completion
    cudaDeviceSynchronize();
    
    return result;
}

