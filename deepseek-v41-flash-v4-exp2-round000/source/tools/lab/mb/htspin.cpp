// schedule-call / total latency of one NEFF with the measuring thread pinned to CPU A and a spinner on CPU B
#include <nrt/nrt.h>
#include <nrt/nrt_async.h>
#include <nrt/nrt_experimental.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <thread>
#include <atomic>
#include <sched.h>
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
#define CK(x) do { NRT_STATUS s_=(x); if (s_) { fprintf(stderr, "%s -> %d\n", #x, (int)s_); exit(1);} } while(0)
static void pin(int c) { cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(c, &cs); sched_setaffinity(0, sizeof cs, &cs); }
static NRT_STATUS g_ret;
int main(int argc, char** argv) {
  CK(nrt_init(NRT_FRAMEWORK_TYPE_NO_FW, "", ""));
  std::ifstream f(argv[1], std::ios::binary); std::vector<char> b((std::istreambuf_iterator<char>(f)), {});
  nrt_model_t* m; CK(nrt_load(b.data(), b.size(), 0, 1, &m));
  nrt_tensor_info_array_t* info; CK(nrt_get_model_tensor_info(m, &info));
  nrt_tensor_set_t *in, *out; CK(nrt_allocate_tensor_set(&in)); CK(nrt_allocate_tensor_set(&out));
  nrt_tensor_t* xin = nullptr;
  for (uint64_t i = 0; i < info->tensor_count; i++) {
    auto& t = info->tensor_array[i];
    nrt_tensor_t* x; CK(nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, 0, t.size, t.name, &x));
    CK(nrt_add_tensor_to_tensor_set(t.usage == NRT_TENSOR_USAGE_INPUT ? in : out, t.name, x));
    if (!xin && t.usage == NRT_TENSOR_USAGE_INPUT && t.size >= 10240) xin = x;
  }
  int A = atoi(argv[2]);
  std::atomic<int> stop{0};
  std::vector<std::thread> sp;
  for (int k = 3; k < argc; k++) { int c = atoi(argv[k]); sp.emplace_back([&, c] { pin(c); while (!stop) __builtin_ia32_pause(); }); }
  pin(A);
  std::vector<double> ts, tw, twr;
  std::vector<char> buf(10240);
  for (int it = 0; it < 200; it++) {
    double te = now() + 1e-3; while (now() < te) {}
    double tw0 = now(); CK(nrt_tensor_write(xin, buf.data(), 0, 10240)); double tw1 = now();
    nrta_seq_t seq; double t0 = now();
    CK(nrta_execute_schedule(m, in, out, 0, &g_ret, &seq));
    double t1 = now(); bool d = false; while (!d) CK(nrta_is_completed(seq, &d));
    if (it >= 20) { ts.push_back((t1 - t0) * 1e6); tw.push_back((now() - t0) * 1e6); twr.push_back((tw1 - tw0) * 1e6); }
  }
  stop = 1; for (auto& t : sp) t.join();
  std::sort(ts.begin(), ts.end()); std::sort(tw.begin(), tw.end()); std::sort(twr.begin(), twr.end());
  printf("cpu %d spinners %d: write10K p50 %5.1f  sched p50 %5.1f  total p50 %6.1f\n", A, argc - 3, twr[90], ts[90], tw[90]);
  nrt_close();
}
