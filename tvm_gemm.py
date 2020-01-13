import tvm
import numpy as np

M = 1024
N = 1024
K = 1024
A = tvm.placeholder((M, K), name='A')
B = tvm.placeholder((K, N), name='B')
k = tvm.reduce_axis((0, K), 'k')
C = tvm.compute((M, N),
                lambda x, y: tvm.sum(A[x, k] * B[k, y], axis=k),
                name='C')
s = tvm.create_schedule(C.op)

share_A = s.cache_read(A, 'shared', [C])
share_B = s.cache_read(B, "shared", [C])
local_A = s.cache_read(share_A, "local", [C])
local_B = s.cache_read(share_B, "local", [C])
local_C = s.cache_write(C, "local")

# parallelism parameters
thread_x = 16
thread_y = 8
block_x = 64
block_y = 128
vthread_x = 2
vthread_y = 2
step = 4

# Get the GPU thread indices
block_dim_x = tvm.thread_axis("blockIdx.x")
block_dim_y = tvm.thread_axis("blockIdx.y")
thread_dim_x = tvm.thread_axis("threadIdx.x")
thread_dim_y = tvm.thread_axis("threadIdx.y")
vthread_dim_x = tvm.thread_axis("vthread")
vthread_dim_y = tvm.thread_axis("vthread")

# Split and bind the workloads
h_C, w_C = s[C].op.axis

bh, rest_h = s[C].split(h_C, factor=block_x)
bw, rest_w = s[C].split(w_C, factor=block_y)
s[C].bind(bh, block_dim_x)
s[C].bind(bw, block_dim_y)

vth, rest_h = s[C].split(rest_h, nparts=vthread_x)
vtw, rest_w = s[C].split(rest_w, nparts=vthread_y)
s[C].bind(vth, vthread_dim_x)
s[C].bind(vtw, vthread_dim_y)

th, rest_h = s[C].split(rest_h, nparts=thread_x)
tw, rest_w = s[C].split(rest_w, nparts=thread_y)
s[C].bind(th, thread_dim_x)
s[C].bind(tw, thread_dim_y)

# Schedule local_C local write
s[C].reorder(bh, bw, vth, vtw, th, tw, rest_h, rest_w)
s[local_C].compute_at(s[C], tw)
hi, wi = s[local_C].op.axis
rk = s[local_C].op.reduce_axis[0]
rko, rki = s[local_C].split(rk, factor=step)
s[local_C].reorder(rko, rki, hi, wi)

# Attach computation to iteration variables
s[share_A].compute_at(s[local_C], rko)
s[share_B].compute_at(s[local_C], rko)
s[local_A].compute_at(s[local_C], rki)
s[local_B].compute_at(s[local_C], rki)

hi, wi = s[share_A].op.axis
th, rest_h = s[share_A].split(hi, nparts=thread_x)
tw, rest_w = s[share_A].split(wi, nparts=thread_y)
s[share_A].reorder(th, tw, rest_h, rest_w)
s[share_A].bind(th, thread_dim_x)
s[share_A].bind(tw, thread_dim_y)

hi, wi = s[share_B].op.axis
th, rest_h = s[share_B].split(hi, nparts=thread_y)
tw, rest_w = s[share_B].split(wi, nparts=thread_x)
s[share_B].reorder(th, tw, rest_h, rest_w)
s[share_B].bind(th, thread_dim_y)
s[share_B].bind(tw, thread_dim_x)

# print(tvm.lower(s, [A, B, C], simple_mode=True))
# exit()
func = tvm.build(s, [A, B, C], 'cuda', target_host='llvm', name="func")
dev_module = func.imported_modules[0]
print("-----GPU code-----")
print(dev_module.get_source())

ctx = tvm.gpu(0)
a_np = np.random.uniform(size=(M, K)).astype(A.dtype)
b_np = np.random.uniform(size=(K, N)).astype(B.dtype)
a = tvm.nd.array(a_np, ctx)
b = tvm.nd.array(b_np, ctx)
c = tvm.nd.array(np.zeros((M, N), dtype=B.dtype), ctx)
func(a, b, c)
tvm.testing.assert_allclose(c.asnumpy(), np.matmul(a.asnumpy(), b.asnumpy()))
print(np.matmul(a.asnumpy(), b.asnumpy()))
evaluator = func.time_evaluator(func.entry_name, ctx, number=1)
print('gemm: %f ms' % (evaluator(a, b, c).mean * 1e3))
