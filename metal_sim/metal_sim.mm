// metal_sim: orchestrates the Metal LIF whole-brain simulation.
// Usage: metal_sim <data_dir> [kernel.metal]
// Protocol (stdin -> stdout, one command per line):
//   run <steps>          advance simulation, replies "OK <total_steps>"
//   rate <group> <hz>    set Poisson rate for a group (0 = off)
//   spikes               dump new spikes as "S <step> <neuron>" lines, end "E"
//   stat                 print "STAT <sum_v> <sum_g> <n_above_th> <steps>"
//   quit
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <algorithm>

static const int SLOTS = 32;   // delay ring slots (must be > dly_steps = 18)
static const int CAP = 8192;   // max spikes per slot per step
static const uint32_t LOGBUF = 1u << 20;

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: metal_sim <data_dir> [kernel.metal]\n"); return 1; }
    std::string dir = argv[1];
    std::string ksrc = argc > 2 ? argv[2] : dir + "/../brain.metal";

    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { fprintf(stderr, "no Metal device\n"); return 1; }

        NSString *src = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:ksrc.c_str()]
                                                  encoding:NSUTF8StringEncoding error:nil];
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
        if (!lib) { fprintf(stderr, "kernel compile error: %s\n",
                            err.localizedDescription.UTF8String); return 1; }

        std::map<std::string, id<MTLComputePipelineState>> ps;
        for (NSString *fn in @[ @"integrate", @"deliver_grab", @"deliver_apply",
                                @"poisson", @"threshold" ]) {
            id<MTLFunction> f = [lib newFunctionWithName:fn];
            id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:f error:&err];
            if (!p) { fprintf(stderr, "pipeline error %s: %s\n", fn.UTF8String,
                              err.localizedDescription.UTF8String); return 1; }
            ps[fn.UTF8String] = p;
        }

        auto loadAll = [&](const std::string &name, size_t &count) -> std::vector<uint8_t> {
            FILE *fp = fopen((dir + "/" + name).c_str(), "rb");
            if (!fp) { fprintf(stderr, "missing %s\n", name.c_str()); exit(1); }
            fseek(fp, 0, SEEK_END); long sz = ftell(fp); fseek(fp, 0, SEEK_SET);
            count = (size_t)sz;
            std::vector<uint8_t> v(sz);
            if (sz && fread(v.data(), 1, sz, fp) != (size_t)sz) exit(1);
            fclose(fp);
            return v;
        };

        size_t n_start_bytes, n_post_bytes, n_w_bytes;
        auto csr_start_v = loadAll("csr_start.i32", n_start_bytes);
        auto csr_post_v = loadAll("csr_post.i32", n_post_bytes);
        auto csr_w_v = loadAll("csr_w.f32", n_w_bytes);
        const int n = (int)(n_start_bytes / 4) - 1;
        const size_t n_syn = n_w_bytes / 4;

        std::map<std::string, std::vector<uint8_t>> groups;
        for (const char *gn : { "sugar", "p9_left", "p9_right" }) {
            size_t c;
            groups[gn] = loadAll(std::string("group_") + gn + ".i32", c);
        }

        id<MTLBuffer> bV     = [dev newBufferWithLength:(size_t)n * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bG     = [dev newBufferWithLength:(size_t)n * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bRfcU  = [dev newBufferWithLength:(size_t)n * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bRfcL  = [dev newBufferWithLength:(size_t)n * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bRate  = [dev newBufferWithLength:(size_t)n * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bStart = [dev newBufferWithBytes:csr_start_v.data() length:n_start_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bPost  = [dev newBufferWithBytes:csr_post_v.data() length:n_post_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bW     = [dev newBufferWithBytes:csr_w_v.data() length:n_w_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bRingC = [dev newBufferWithLength:SLOTS * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bRingS = [dev newBufferWithLength:(size_t)SLOTS * CAP * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bProcC = [dev newBufferWithLength:SLOTS * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bLogI  = [dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bLog   = [dev newBufferWithLength:(size_t)LOGBUF * 8 options:MTLResourceStorageModeShared];

        float *V = (float *)bV.contents;
        float *G = (float *)bG.contents;
        int *rfcU = (int *)bRfcU.contents;
        int *rfcL = (int *)bRfcL.contents;
        float *rate = (float *)bRate.contents;
        for (int i = 0; i < n; i++) {
            V[i] = -52.0f; G[i] = 0.0f; rfcU[i] = 0; rfcL[i] = 22; rate[i] = 0.0f;
        }
        uint32_t *logIdx = (uint32_t *)bLogI.contents;
        *logIdx = 0;
        struct SU32 { uint32_t x, y; };
        SU32 *logBuf = (SU32 *)bLog.contents;

        const float dt = 0.1f, v0 = -52.0f, t_mbr = 20.0f, tau = 5.0f;
        const float th = -45.0f, vr = -52.0f;
        const float w_poi = 0.275f * 250.0f;

        int total_steps = 0;
        uint64_t consumed = 0;
        id<MTLCommandQueue> q = [dev newCommandQueue];
        char line[1 << 16];
        fprintf(stdout, "READY n=%d syn=%zu\n", n, n_syn);
        fflush(stdout);

        auto setGroupRate = [&](const std::string &gname, float hz) {
            auto it = groups.find(gname);
            if (it == groups.end()) { fprintf(stdout, "ERR group\n"); return; }
            int32_t cnt = ((const int32_t *)it->second.data())[0];
            const int32_t *idx = ((const int32_t *)it->second.data()) + 1;
            for (int32_t k = 0; k < cnt; k++) {
                rate[idx[k]] = hz * 0.001f;   // Hz -> events/ms (kernel: p = rate*dt_ms)
                rfcL[idx[k]] = 0;   // Brian2: neu[i].rfc = 0 for stim neurons
            }
            fprintf(stdout, "OK rate %s %g x%d\n", gname.c_str(), hz, cnt);
        };

        while (fgets(line, sizeof(line), stdin)) {
            char cmd[32] = {0}, name[64] = {0};
            long a = 0; double b = 0;
            if (sscanf(line, "%31s", cmd) != 1) continue;
            if (!strcmp(cmd, "quit")) break;

            if (!strcmp(cmd, "rate")) {
                if (sscanf(line, "%*s %63s %lf", name, &b) == 2) setGroupRate(name, (float)b);
                else fprintf(stdout, "ERR arg\n");
            } else if (!strcmp(cmd, "run")) {
                if (sscanf(line, "%*s %ld", &a) != 1) { fprintf(stdout, "ERR arg\n"); continue; }
                for (int done = 0; done < a; ) {
                    int batch = (int)std::min<long>(100, a - done);
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
                    for (int s = 0; s < batch; s++) {
                        int step = total_steps + done + s;
                        int slot = ((step % SLOTS) + SLOTS) % SLOTS;
                        int fut_slot = (((step + 18) % SLOTS) + SLOTS) % SLOTS;
                        int sc_now[2] = {slot, CAP};
                        int sc_fut[2] = {fut_slot, CAP};
                        float p4[4] = {dt, v0, t_mbr, tau};
                        float p2[2] = {dt, w_poi};
                        MTLSize tg256 = MTLSizeMake((n + 255) / 256, 1, 1);
                        MTLSize th256 = MTLSizeMake(256, 1, 1);
                        MTLSize tgCap = MTLSizeMake((CAP + 255) / 256, 1, 1);

                        [ce setComputePipelineState:ps["integrate"]];
                        [ce setBuffer:bV offset:0 atIndex:0];
                        [ce setBuffer:bG offset:0 atIndex:1];
                        [ce setBuffer:bRfcU offset:0 atIndex:2];
                        [ce setBytes:&step length:4 atIndex:3];
                        [ce setBytes:p4 length:16 atIndex:4];
                        [ce setBytes:&n length:4 atIndex:5];
                        [ce dispatchThreadgroups:tg256 threadsPerThreadgroup:th256];

                        [ce setComputePipelineState:ps["deliver_grab"]];
                        [ce setBuffer:bRingC offset:0 atIndex:0];
                        [ce setBuffer:bProcC offset:0 atIndex:1];
                        [ce setBytes:&slot length:4 atIndex:2];
                        [ce dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];

                        [ce setComputePipelineState:ps["deliver_apply"]];
                        [ce setBuffer:bProcC offset:0 atIndex:0];
                        [ce setBuffer:bRingS offset:0 atIndex:1];
                        [ce setBuffer:bG offset:0 atIndex:2];
                        [ce setBuffer:bStart offset:0 atIndex:3];
                        [ce setBuffer:bPost offset:0 atIndex:4];
                        [ce setBuffer:bW offset:0 atIndex:5];
                        [ce setBytes:sc_now length:8 atIndex:6];
                        [ce dispatchThreadgroups:tgCap threadsPerThreadgroup:th256];

                        [ce setComputePipelineState:ps["poisson"]];
                        [ce setBuffer:bV offset:0 atIndex:0];
                        [ce setBuffer:bRate offset:0 atIndex:1];
                        [ce setBytes:p2 length:8 atIndex:2];
                        [ce setBytes:&step length:4 atIndex:3];
                        [ce setBytes:&n length:4 atIndex:4];
                        [ce dispatchThreadgroups:tg256 threadsPerThreadgroup:th256];

                        [ce setComputePipelineState:ps["threshold"]];
                        [ce setBuffer:bV offset:0 atIndex:0];
                        [ce setBuffer:bG offset:0 atIndex:1];
                        [ce setBuffer:bRfcU offset:0 atIndex:2];
                        [ce setBuffer:bRfcL offset:0 atIndex:3];
                        [ce setBuffer:bRingC offset:0 atIndex:4];
                        [ce setBuffer:bRingS offset:0 atIndex:5];
                        [ce setBuffer:bLogI offset:0 atIndex:6];
                        [ce setBuffer:bLog offset:0 atIndex:7];
                        [ce setBytes:&th length:4 atIndex:8];
                        [ce setBytes:&vr length:4 atIndex:9];
                        [ce setBytes:sc_fut length:8 atIndex:10];
                        [ce setBytes:&step length:4 atIndex:11];
                        [ce setBytes:&n length:4 atIndex:12];
                        [ce dispatchThreadgroups:tg256 threadsPerThreadgroup:th256];
                    }
                    [ce endEncoding];
                    [cb commit];
                    [cb waitUntilCompleted];
                    done += batch;
                }
                total_steps += (int)a;
                fprintf(stdout, "OK %d\n", total_steps);
            } else if (!strcmp(cmd, "spikes")) {
                uint32_t produced = *logIdx;
                uint64_t upto = std::min<uint64_t>(produced, LOGBUF);
                for (uint64_t j = consumed; j < upto; j++)
                    fprintf(stdout, "S %u %u\n", logBuf[j].x, logBuf[j].y);
                consumed = upto;
                fprintf(stdout, "E %llu\n", (unsigned long long)consumed);
            } else if (!strcmp(cmd, "reset")) {
                for (int i = 0; i < n; i++) {
                    V[i] = -52.0f; G[i] = 0.0f; rfcU[i] = 0;
                    rfcL[i] = 22; rate[i] = 0.0f;
                }
                *logIdx = 0; consumed = 0;
                memset((int *)bRingC.contents, 0, SLOTS * 4);
                memset((int *)bProcC.contents, 0, SLOTS * 4);
                fprintf(stdout, "OK reset\n");
            } else if (!strcmp(cmd, "stat")) {
                double sv = 0, sg = 0; int nth = 0;
                for (int i = 0; i < n; i++) {
                    sv += V[i]; sg += G[i];
                    if (V[i] > -45.0f) nth++;
                }
                fprintf(stdout, "STAT %.3f %.3f %d %d\n", sv, sg, nth, total_steps);
            } else {
                fprintf(stdout, "ERR cmd\n");
            }
            fflush(stdout);
        }
    }
    return 0;
}
