import logging
import sys
import numpy as np

import tvm
import topi
from topi.testing import conv2d_nchw_python

from tvm import autotvm


@autotvm.template
def gemm(M, N, K):
    A = tvm.placeholder((M, K), name='A')
    B = tvm.placeholder((K, N), name='B')
    k = tvm.reduce_axis((0, K), 'k')
    C = tvm.compute((M, N),
                    lambda x, y: tvm.sum(A[x, k] * B[k, y], axis=k),
                    name='C')
    s = tvm.create_schedule([C.op])
    cfg = autotvm.get_config()

    local_C = s.cache_write(C, "local")

    #schedule C
    h_C, w_C = s[C].op.axis
    cfg.define_split("h_C", h_C, num_outputs=4)
    cfg.define_split("w_C", w_C, num_outputs=4)
    bh, vth, th, h = cfg["h_C"].apply(s, C, h_C)
    bw, vtw, tw, w = cfg["w_C"].apply(s, C, w_C)
    s[C].bind(bh, tvm.thread_axis("blockIdx.x"))
    s[C].bind(bw, tvm.thread_axis("blockIdx.y"))
    s[C].bind(vth, tvm.thread_axis("vthread"))
    s[C].bind(vtw, tvm.thread_axis("vthread"))
    s[C].bind(th, tvm.thread_axis("threadIdx.x"))
    s[C].bind(tw, tvm.thread_axis("threadIdx.y"))
    s[C].reorder(bh, bw, vth, vtw, th, tw, h, w)

    #schedule local_C
    s[local_C].compute_at(s[C], tw)
    hi, wi = s[local_C].op.axis
    rk = s[local_C].op.reduce_axis[0]
    cfg.define_split("rk", rk, num_outputs=2)
    rko, rki = cfg["rk"].apply(s, local_C, rk)

    s[local_C].reorder(rko, rki, hi, wi)

    #schedule share_A and share_B
    share_A = s.cache_read(A, 'shared', local_C)
    s[share_A].compute_at(s[local_C], rko)
    sh_h, sh_w = s[share_A].op.axis
    th, sh_h = s[share_A].split(sh_h, nparts=cfg["h_C"].size[2])
    tw, sh_w = s[share_A].split(sh_w, nparts=cfg["w_C"].size[2])
    s[share_A].bind(th, tvm.thread_axis("threadIdx.x"))
    s[share_A].bind(tw, tvm.thread_axis("threadIdx.y"))

    share_B = s.cache_read(B, "shared", local_C)
    s[share_B].compute_at(s[local_C], rko)
    sh_h, sh_w = s[share_B].op.axis
    th, sh_h = s[share_B].split(sh_h, nparts=cfg["h_C"].size[2])
    tw, sh_w = s[share_B].split(sh_w, nparts=cfg["w_C"].size[2])
    s[share_B].bind(th, tvm.thread_axis("threadIdx.x"))
    s[share_B].bind(tw, tvm.thread_axis("threadIdx.y"))

    #schedule local_A and local_B
    local_A = s.cache_read(share_A, "local", local_C)
    s[local_A].compute_at(s[local_C], rki)
    local_B = s.cache_read(share_B, "local", local_C)
    s[local_B].compute_at(s[local_C], rki)
    return s, [A, B, C]


logging.getLogger('autotvm').setLevel(logging.DEBUG)
logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))
M, N, K = 1024, 1024, 1024
task = autotvm.task.create(gemm, args=(M, N, K), target='cuda')
print(task.config_space)

measure_option = autotvm.measure_option(builder=autotvm.LocalBuilder(),
                                        runner=autotvm.LocalRunner(
                                            repeat=3,
                                            min_repeat_ms=100,
                                            timeout=4))

tuner = autotvm.tuner.XGBTuner(task)
tuner.tune(n_trial=200000,
           measure_option=measure_option,
           callbacks=[autotvm.callback.log_to_file('gemm.log')])

# inspect the best config
dispatch_context = autotvm.apply_history_best("gemm.log")
best_config = dispatch_context.query(task.target, task.workload)
print("\nBest config:")
print(best_config)

# apply history best from log file
with autotvm.apply_history_best('gemm.log'):
    with tvm.target.create("cuda"):
        s, arg_bufs = gemm(M, N, K)
        func = tvm.build(s, arg_bufs)

print(func.imported_modules[0].get_source())
# check correctness
a_np = np.random.uniform(size=(M, K)).astype(np.float32)
b_np = np.random.uniform(size=(K, N)).astype(np.float32)
c_np = np.matmul(a_np, b_np)
ctx = tvm.gpu()
a_tvm = tvm.nd.array(a_np, ctx=ctx)
b_tvm = tvm.nd.array(b_np, ctx=ctx)
c_tvm = tvm.nd.empty(c_np.shape, ctx=ctx)
func(a_tvm, b_tvm, c_tvm)

tvm.testing.assert_allclose(c_np, c_tvm.asnumpy(), rtol=1e-2)

# Evaluate running time. Here we choose a large repeat number (400) to reduce the noise
# and the overhead of kernel launch. You can also use nvprof to validate the result.
evaluator = func.time_evaluator(func.entry_name, ctx, number=400)
print('Time cost of this operator: %f' % evaluator(a_tvm, b_tvm, c_tvm).mean)