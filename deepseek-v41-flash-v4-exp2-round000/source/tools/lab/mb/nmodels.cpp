// schedule-call cost vs number of models loaded on the core / in the process
#include <nrt/nrt.h>
#include <nrt/nrt_experimental.h>
#include <nrt/nrt_async.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <fstream>
#include <iterator>
#include <algorithm>
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
#define CK(x) do { NRT_STATUS s_=(x); if (s_) { fprintf(stderr, "%s -> %d\n", #x, (int)s_); exit(1);} } while(0)
struct M { nrt_model_t* m; nrt_tensor_set_t *in, *out; };
static M load(std::vector<char>& b, int core) {
  M r; CK(nrt_load(b.data(), b.size(), core, 1, &r.m));
  nrt_tensor_info_array_t* info; CK(nrt_get_model_tensor_info(r.m, &info));
  CK(nrt_allocate_tensor_set(&r.in)); CK(nrt_allocate_tensor_set(&r.out));
  for (uint64_t i = 0; i < info->tensor_count; i++) {
    auto& t = info->tensor_array[i];
    nrt_tensor_t* x; CK(nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, core, t.size, t.name, &x));
    CK(nrt_add_tensor_to_tensor_set(t.usage == NRT_TENSOR_USAGE_INPUT ? r.in : r.out, t.name, x));
  }
  return r;
}
static NRT_STATUS g_ret;
static void meas(M& a, const char* tag) {
  std::vector<double> ts, tw;
  for (int it = 0; it < 150; it++) {
    double te = now() + 2e-3; while (now() < te) {}
    nrta_seq_t seq; double t0 = now();
    CK(nrta_execute_schedule(a.m, a.in, a.out, 0, &g_ret, &seq));
    double t1 = now(); bool d = false; while (!d) CK(nrta_is_completed(seq, &d));
    if (it >= 20) { ts.push_back((t1 - t0) * 1e6); tw.push_back((now() - t0) * 1e6); }
  }
  std::sort(ts.begin(), ts.end()); std::sort(tw.begin(), tw.end());
  printf("%-28s sched p50 %6.1f  total p50 %7.1f\n", tag, ts[ts.size()/2], tw[tw.size()/2]); fflush(stdout);
}
int main(int argc, char** argv) {
  CK(nrt_init(NRT_FRAMEWORK_TYPE_NO_FW, "", ""));
  std::ifstream f(argv[1], std::ios::binary); std::vector<char> A((std::istreambuf_iterator<char>(f)), {});
  std::ifstream g(argv[2], std::ios::binary); std::vector<char> Bf((std::istreambuf_iterator<char>(g)), {});
  M a = load(A, 0);
  meas(a, "1 model");
  std::vector<M> extra;
  for (int n : {20, 60, 150}) {
    while ((int)extra.size() < n) extra.push_back(load(Bf, 0));
    char t[64]; snprintf(t, 64, "+%d models same core", n); meas(a, t);
  }
  nrt_close();
}
