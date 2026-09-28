// Metal kernels for the fly whole-brain LIF simulator.
// Units: v/g in mV, dt in ms. Faithful port of the Brian2 model in
// fly-brain/code/run_brian2_cuda.py:
//   dv/dt = (v0 - v + g) / t_mbr   (unless refractory)
//   dg/dt = -g / tau               (unless refractory)
//   threshold v > v_th -> v = v_rst, g = 0, refractory rfc_len[i]
//   synapses: g_pre spike -> g_post += w after dly_steps
//   PoissonInput on stim neurons: v += w_poi at rate r (rfc forced to 0)
#include <metal_stdlib>
using namespace metal;

// integrate: one thread per neuron
kernel void integrate(
    device float *v            [[buffer(0)]],
    device float *g            [[buffer(1)]],
    device const int *rfc_until [[buffer(2)]],
    constant int &step          [[buffer(3)]],
    constant float4 &p          [[buffer(4)]],  // dt, v0, t_mbr, tau
    constant int &nn            [[buffer(5)]],
    uint i [[thread_position_in_grid]])
{
    if ((int)i >= nn) return;
    if (step >= rfc_until[i]) {
        float vv = v[i], gg = g[i];
        v[i] = vv + p.x * (p.y - vv + gg) / p.z;
        g[i] = gg - p.x * gg / p.w;
    }
}

// deliver_grab: single thread swaps the due-slot count into proc_cnt
kernel void deliver_grab(
    device atomic_int *ring_cnt  [[buffer(0)]],   // SLOTS counters
    device int *proc_cnt         [[buffer(1)]],   // SLOTS scratch
    constant int &slot           [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i == 0) {
        proc_cnt[slot] = atomic_exchange_explicit(&ring_cnt[slot], 0,
                                                  memory_order_relaxed);
    }
}

// deliver_apply: CAP threads; each processes at most one due spike
kernel void deliver_apply(
    device const int *proc_cnt   [[buffer(0)]],
    device const int *ring_sp    [[buffer(1)]],   // SLOTS * CAP entries
    device atomic_float *g       [[buffer(2)]],
    device const int *csr_start  [[buffer(3)]],
    device const int *csr_post   [[buffer(4)]],
    device const float *csr_w    [[buffer(5)]],
    constant int2 &slot_cap      [[buffer(6)]],   // slot, cap
    uint t [[thread_position_in_grid]])
{
    int slot = slot_cap.x, cap = slot_cap.y;
    if ((int)t >= cap) return;
    int cnt = proc_cnt[slot];
    if ((int)t >= cnt) return;
    int pre = ring_sp[slot * cap + t];
    int s = csr_start[pre], e = csr_start[pre + 1];
    for (int k = s; k < e; k++) {
        atomic_fetch_add_explicit(&g[csr_post[k]], csr_w[k],
                                  memory_order_relaxed);
    }
}

// poisson: one thread per neuron; kick v by w_poi with prob rate*dt
kernel void poisson(
    device float *v         [[buffer(0)]],
    device const float *rate [[buffer(1)]],
    constant float2 &p      [[buffer(2)]],       // dt_ms, w_poi
    constant int &step      [[buffer(3)]],
    constant int &nn        [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    if ((int)i >= nn) return;
    float r = rate[i];
    if (r <= 0.0f) return;
    uint x = (uint)i * 747796405u + (uint)step * 2891336453u;
    x = ((x >> ((x >> 28u) + 4u)) + x) * 277803737u;
    x = (x >> 22u) ^ x;
    float u = (float)x * (1.0f / 4294967296.0f);
    if (u < r * p.x) v[i] += p.y;
}

// threshold: one thread per neuron; fire, reset, schedule, log
kernel void threshold(
    device float *v            [[buffer(0)]],
    device float *g            [[buffer(1)]],
    device int *rfc_until      [[buffer(2)]],
    device const int *rfc_len  [[buffer(3)]],
    device atomic_int *ring_cnt [[buffer(4)]],
    device int *ring_sp        [[buffer(5)]],
    device atomic_uint *log_idx [[buffer(6)]],
    device uint2 *log_buf      [[buffer(7)]],    // (step, neuron)
    constant float &th         [[buffer(8)]],
    constant float &vr         [[buffer(9)]],
    constant int2 &slot_cap    [[buffer(10)]],   // (step + DLY) % SLOTS, cap
    constant int &step         [[buffer(11)]],
    constant int &nn           [[buffer(12)]],
    uint i [[thread_position_in_grid]])
{
    if ((int)i >= nn) return;
    if (step < rfc_until[i]) return;
    if (v[i] <= th) return;
    v[i] = vr;
    g[i] = 0.0f;
    rfc_until[i] = step + rfc_len[i];
    uint li = atomic_fetch_add_explicit(log_idx, 1u, memory_order_relaxed);
    if (li < 1u << 20) log_buf[li] = uint2((uint)step, (uint)i);
    uint si = atomic_fetch_add_explicit(&ring_cnt[slot_cap.x], 1,
                                        memory_order_relaxed);
    if (si < (uint)slot_cap.y) ring_sp[slot_cap.x * slot_cap.y + si] = (int)i;
}
