extern "C" __global__ void default_function_kernel0(float* __restrict__ A,
                                                    float* __restrict__ B,
                                                    float* __restrict__ C) {
  float C_local[64];
  __shared__ float A_shared[128];
  __shared__ float B_shared[1024];
  float A_shared_local[16];
  float B_shared_local[4];
  for (int x_c_init = 0; x_c_init < 2; ++x_c_init) {
    for (int y_c_init = 0; y_c_init < 2; ++y_c_init) {
      C_local[((x_c_init * 2) + y_c_init)] = 0.000000e+00f;
      C_local[(((x_c_init * 2) + y_c_init) + 8)] = 0.000000e+00f;
      C_local[(((x_c_init * 2) + y_c_init) + 16)] = 0.000000e+00f;
      C_local[(((x_c_init * 2) + y_c_init) + 24)] = 0.000000e+00f;
      C_local[(((x_c_init * 2) + y_c_init) + 32)] = 0.000000e+00f;
      C_local[(((x_c_init * 2) + y_c_init) + 40)] = 0.000000e+00f;
      C_local[(((x_c_init * 2) + y_c_init) + 48)] = 0.000000e+00f;
      C_local[(((x_c_init * 2) + y_c_init) + 56)] = 0.000000e+00f;
      C_local[(((x_c_init * 2) + y_c_init) + 4)] = 0.000000e+00f;
      C_local[(((x_c_init * 2) + y_c_init) + 12)] = 0.000000e+00f;
      C_local[(((x_c_init * 2) + y_c_init) + 20)] = 0.000000e+00f;
      C_local[(((x_c_init * 2) + y_c_init) + 28)] = 0.000000e+00f;
      C_local[(((x_c_init * 2) + y_c_init) + 36)] = 0.000000e+00f;
      C_local[(((x_c_init * 2) + y_c_init) + 44)] = 0.000000e+00f;
      C_local[(((x_c_init * 2) + y_c_init) + 52)] = 0.000000e+00f;
      C_local[(((x_c_init * 2) + y_c_init) + 60)] = 0.000000e+00f;
    }
  }
  for (int k_outer = 0; k_outer < 256; ++k_outer) {
    __syncthreads();
    for (int ax0_inner = 0; ax0_inner < 16; ++ax0_inner) {
      if (((int)threadIdx.y) < 4) {
        A_shared[(((((int)threadIdx.x) * 64) + (ax0_inner * 4)) +
                  ((int)threadIdx.y))] =
            A[(((((((int)blockIdx.x) * 32768) + (((int)threadIdx.x) * 16384)) +
                 (ax0_inner * 1024)) +
                (k_outer * 4)) +
               ((int)threadIdx.y))];
      }
    }
    for (int ax0_inner1 = 0; ax0_inner1 < 2; ++ax0_inner1) {
      for (int ax1_inner = 0; ax1_inner < 4; ++ax1_inner) {
        B_shared[((((((int)threadIdx.x) * 512) + (ax0_inner1 * 256)) +
                   (((int)threadIdx.y) * 4)) +
                  ax1_inner)] =
            B[((((((k_outer * 4096) + (((int)threadIdx.x) * 2048)) +
                  (ax0_inner1 * 1024)) +
                 (((int)blockIdx.y) * 256)) +
                (((int)threadIdx.y) * 4)) +
               ax1_inner)];
      }
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 4; ++k_inner) {
      for (int ax0 = 0; ax0 < 2; ++ax0) {
        A_shared_local[ax0] =
            A_shared[(((((int)threadIdx.x) * 8) + (ax0 * 4)) + k_inner)];
        A_shared_local[(ax0 + 2)] =
            A_shared[((((((int)threadIdx.x) * 8) + (ax0 * 4)) + k_inner) + 16)];
        A_shared_local[(ax0 + 4)] =
            A_shared[((((((int)threadIdx.x) * 8) + (ax0 * 4)) + k_inner) + 32)];
        A_shared_local[(ax0 + 6)] =
            A_shared[((((((int)threadIdx.x) * 8) + (ax0 * 4)) + k_inner) + 48)];
        A_shared_local[(ax0 + 8)] =
            A_shared[((((((int)threadIdx.x) * 8) + (ax0 * 4)) + k_inner) + 64)];
        A_shared_local[(ax0 + 10)] =
            A_shared[((((((int)threadIdx.x) * 8) + (ax0 * 4)) + k_inner) + 80)];
        A_shared_local[(ax0 + 12)] =
            A_shared[((((((int)threadIdx.x) * 8) + (ax0 * 4)) + k_inner) + 96)];
        A_shared_local[(ax0 + 14)] = A_shared[(
            (((((int)threadIdx.x) * 8) + (ax0 * 4)) + k_inner) + 112)];
      }
      for (int ax1 = 0; ax1 < 2; ++ax1) {
        B_shared_local[ax1] =
            B_shared[(((k_inner * 256) + (((int)threadIdx.y) * 2)) + ax1)];
        B_shared_local[(ax1 + 2)] = B_shared[(
            (((k_inner * 256) + (((int)threadIdx.y) * 2)) + ax1) + 128)];
      }
      for (int x_c = 0; x_c < 2; ++x_c) {
        for (int y_c = 0; y_c < 2; ++y_c) {
          C_local[((x_c * 2) + y_c)] =
              (C_local[((x_c * 2) + y_c)] +
               (A_shared_local[x_c] * B_shared_local[y_c]));
          C_local[(((x_c * 2) + y_c) + 8)] =
              (C_local[(((x_c * 2) + y_c) + 8)] +
               (A_shared_local[(x_c + 2)] * B_shared_local[y_c]));
          C_local[(((x_c * 2) + y_c) + 16)] =
              (C_local[(((x_c * 2) + y_c) + 16)] +
               (A_shared_local[(x_c + 4)] * B_shared_local[y_c]));
          C_local[(((x_c * 2) + y_c) + 24)] =
              (C_local[(((x_c * 2) + y_c) + 24)] +
               (A_shared_local[(x_c + 6)] * B_shared_local[y_c]));
          C_local[(((x_c * 2) + y_c) + 32)] =
              (C_local[(((x_c * 2) + y_c) + 32)] +
               (A_shared_local[(x_c + 8)] * B_shared_local[y_c]));
          C_local[(((x_c * 2) + y_c) + 40)] =
              (C_local[(((x_c * 2) + y_c) + 40)] +
               (A_shared_local[(x_c + 10)] * B_shared_local[y_c]));
          C_local[(((x_c * 2) + y_c) + 48)] =
              (C_local[(((x_c * 2) + y_c) + 48)] +
               (A_shared_local[(x_c + 12)] * B_shared_local[y_c]));
          C_local[(((x_c * 2) + y_c) + 56)] =
              (C_local[(((x_c * 2) + y_c) + 56)] +
               (A_shared_local[(x_c + 14)] * B_shared_local[y_c]));
          C_local[(((x_c * 2) + y_c) + 4)] =
              (C_local[(((x_c * 2) + y_c) + 4)] +
               (A_shared_local[x_c] * B_shared_local[(y_c + 2)]));
          C_local[(((x_c * 2) + y_c) + 12)] =
              (C_local[(((x_c * 2) + y_c) + 12)] +
               (A_shared_local[(x_c + 2)] * B_shared_local[(y_c + 2)]));
          C_local[(((x_c * 2) + y_c) + 20)] =
              (C_local[(((x_c * 2) + y_c) + 20)] +
               (A_shared_local[(x_c + 4)] * B_shared_local[(y_c + 2)]));
          C_local[(((x_c * 2) + y_c) + 28)] =
              (C_local[(((x_c * 2) + y_c) + 28)] +
               (A_shared_local[(x_c + 6)] * B_shared_local[(y_c + 2)]));
          C_local[(((x_c * 2) + y_c) + 36)] =
              (C_local[(((x_c * 2) + y_c) + 36)] +
               (A_shared_local[(x_c + 8)] * B_shared_local[(y_c + 2)]));
          C_local[(((x_c * 2) + y_c) + 44)] =
              (C_local[(((x_c * 2) + y_c) + 44)] +
               (A_shared_local[(x_c + 10)] * B_shared_local[(y_c + 2)]));
          C_local[(((x_c * 2) + y_c) + 52)] =
              (C_local[(((x_c * 2) + y_c) + 52)] +
               (A_shared_local[(x_c + 12)] * B_shared_local[(y_c + 2)]));
          C_local[(((x_c * 2) + y_c) + 60)] =
              (C_local[(((x_c * 2) + y_c) + 60)] +
               (A_shared_local[(x_c + 14)] * B_shared_local[(y_c + 2)]));
        }
      }
    }
  }
  for (int x_inner_inner_inner = 0; x_inner_inner_inner < 2;
       ++x_inner_inner_inner) {
    for (int y_inner_inner_inner = 0; y_inner_inner_inner < 2;
         ++y_inner_inner_inner) {
      C[((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.x) * 2048)) +
            (x_inner_inner_inner * 1024)) +
           (((int)blockIdx.y) * 256)) +
          (((int)threadIdx.y) * 2)) +
         y_inner_inner_inner)] =
          C_local[((x_inner_inner_inner * 2) + y_inner_inner_inner)];
      C[(((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.x) * 2048)) +
             (x_inner_inner_inner * 1024)) +
            (((int)blockIdx.y) * 256)) +
           (((int)threadIdx.y) * 2)) +
          y_inner_inner_inner) +
         4096)] =
          C_local[(((x_inner_inner_inner * 2) + y_inner_inner_inner) + 8)];
      C[(((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.x) * 2048)) +
             (x_inner_inner_inner * 1024)) +
            (((int)blockIdx.y) * 256)) +
           (((int)threadIdx.y) * 2)) +
          y_inner_inner_inner) +
         8192)] =
          C_local[(((x_inner_inner_inner * 2) + y_inner_inner_inner) + 16)];
      C[(((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.x) * 2048)) +
             (x_inner_inner_inner * 1024)) +
            (((int)blockIdx.y) * 256)) +
           (((int)threadIdx.y) * 2)) +
          y_inner_inner_inner) +
         12288)] =
          C_local[(((x_inner_inner_inner * 2) + y_inner_inner_inner) + 24)];
      C[(((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.x) * 2048)) +
             (x_inner_inner_inner * 1024)) +
            (((int)blockIdx.y) * 256)) +
           (((int)threadIdx.y) * 2)) +
          y_inner_inner_inner) +
         16384)] =
          C_local[(((x_inner_inner_inner * 2) + y_inner_inner_inner) + 32)];
      C[(((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.x) * 2048)) +
             (x_inner_inner_inner * 1024)) +
            (((int)blockIdx.y) * 256)) +
           (((int)threadIdx.y) * 2)) +
          y_inner_inner_inner) +
         20480)] =
          C_local[(((x_inner_inner_inner * 2) + y_inner_inner_inner) + 40)];
      C[(((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.x) * 2048)) +
             (x_inner_inner_inner * 1024)) +
            (((int)blockIdx.y) * 256)) +
           (((int)threadIdx.y) * 2)) +
          y_inner_inner_inner) +
         24576)] =
          C_local[(((x_inner_inner_inner * 2) + y_inner_inner_inner) + 48)];
      C[(((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.x) * 2048)) +
             (x_inner_inner_inner * 1024)) +
            (((int)blockIdx.y) * 256)) +
           (((int)threadIdx.y) * 2)) +
          y_inner_inner_inner) +
         28672)] =
          C_local[(((x_inner_inner_inner * 2) + y_inner_inner_inner) + 56)];
      C[(((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.x) * 2048)) +
             (x_inner_inner_inner * 1024)) +
            (((int)blockIdx.y) * 256)) +
           (((int)threadIdx.y) * 2)) +
          y_inner_inner_inner) +
         128)] =
          C_local[(((x_inner_inner_inner * 2) + y_inner_inner_inner) + 4)];
      C[(((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.x) * 2048)) +
             (x_inner_inner_inner * 1024)) +
            (((int)blockIdx.y) * 256)) +
           (((int)threadIdx.y) * 2)) +
          y_inner_inner_inner) +
         4224)] =
          C_local[(((x_inner_inner_inner * 2) + y_inner_inner_inner) + 12)];
      C[(((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.x) * 2048)) +
             (x_inner_inner_inner * 1024)) +
            (((int)blockIdx.y) * 256)) +
           (((int)threadIdx.y) * 2)) +
          y_inner_inner_inner) +
         8320)] =
          C_local[(((x_inner_inner_inner * 2) + y_inner_inner_inner) + 20)];
      C[(((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.x) * 2048)) +
             (x_inner_inner_inner * 1024)) +
            (((int)blockIdx.y) * 256)) +
           (((int)threadIdx.y) * 2)) +
          y_inner_inner_inner) +
         12416)] =
          C_local[(((x_inner_inner_inner * 2) + y_inner_inner_inner) + 28)];
      C[(((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.x) * 2048)) +
             (x_inner_inner_inner * 1024)) +
            (((int)blockIdx.y) * 256)) +
           (((int)threadIdx.y) * 2)) +
          y_inner_inner_inner) +
         16512)] =
          C_local[(((x_inner_inner_inner * 2) + y_inner_inner_inner) + 36)];
      C[(((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.x) * 2048)) +
             (x_inner_inner_inner * 1024)) +
            (((int)blockIdx.y) * 256)) +
           (((int)threadIdx.y) * 2)) +
          y_inner_inner_inner) +
         20608)] =
          C_local[(((x_inner_inner_inner * 2) + y_inner_inner_inner) + 44)];
      C[(((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.x) * 2048)) +
             (x_inner_inner_inner * 1024)) +
            (((int)blockIdx.y) * 256)) +
           (((int)threadIdx.y) * 2)) +
          y_inner_inner_inner) +
         24704)] =
          C_local[(((x_inner_inner_inner * 2) + y_inner_inner_inner) + 52)];
      C[(((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.x) * 2048)) +
             (x_inner_inner_inner * 1024)) +
            (((int)blockIdx.y) * 256)) +
           (((int)threadIdx.y) * 2)) +
          y_inner_inner_inner) +
         28800)] =
          C_local[(((x_inner_inner_inner * 2) + y_inner_inner_inner) + 60)];
    }
  }
}