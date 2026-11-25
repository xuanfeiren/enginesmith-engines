// idle-penalty micro-benchmark: model A (big) on core c; vary the idle time before it and an optional warm-up
// launch of model W (tiny) or of A itself X us before.
#include <nrt/nrt.h>
#include <nrt/nrt_experimental.h>
#include <nrt/nrt_async.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <iterator>
#include <algorithm>
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
#define CK(x) do { NRT_STATUS s_=(x); if (s_) { fprintf(stderr, "%s -> %d\n", #x, (int)s_); exit(1);} } while(0)
struct M { nrt_model_t* m; nrt_tensor_set_t *in, *out; };
static M load(const char* path, int core) {
  std::ifstream f(path, std::ios::binary);
  std::vector<char> b((std::istreambuf_iterator<char>(f)), {});
  M r; CK(nrt_load(b.data(), b.size(), core, 1, &r.m));
  nrt_tensor_info_array_t* info; CK(nrt_get_model_tensor_info(r.m, &info));
  CK(nrt_allocate_tensor_set(&r.in)); CK(nrt_allocate_tensor_set(&r.out));
  for (uint64_t i = 0; i < info->tensor_count; i++) {
    auto& t = info->tensor_array[i];
    nrt_tensor_t* x; CK(nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, core, t.size, t.name, &x));
    CK(nrt_tensor_memset(x, 0, 0, t.size));
    if (t.usage == NRT_TENSOR_USAGE_INPUT) CK(nrt_add_tensor_to_tensor_set(r.in, t.name, x));
    else CK(nrt_add_tensor_to_tensor_set(r.out, t.name, x));
  }
  return r;
}
static NRT_STATUS g_ret[4096]; static int g_ri = 0;
static nrta_seq_t launch(M& m) { NRT_STATUS& ret = g_ret[(g_ri++) % 4096]; nrta_seq_t seq; while (true) { NRT_STATUS s = nrta_execute_schedule(m.m, m.in, m.out, 0, &ret, &seq); if (s == NRT_SUCCESS) break; if (s != NRT_QUEUE_FULL) { fprintf(stderr, "sched %d\n", s); exit(1);} } return seq; }
static void waitc(nrta_seq_t s) { bool d = false; while (!d) CK(nrta_is_completed(s, &d)); }
static void spin(double us) { double t = now() + us * 1e-6; while (now() < t) {} }
int main(int argc, char** argv) {
  int core = atoi(argv[1]);
  CK(nrt_init(NRT_FRAMEWORK_TYPE_NO_FW, "", ""));
  fprintf(stderr,"init ok\n"); M A = load(argv[2], core); fprintf(stderr,"A ok\n"); M W = load(argv[3], core); fprintf(stderr,"W ok\n");
  for (int i = 0; i < 30; i++) { waitc(launch(A)); waitc(launch(W)); }
  double idles[] = {0, 200, 1000, 5000, 40000};
  for (double idle : idles) {
    for (int mode = 0; mode < 5; mode++) {   // 0 none, 1 W 100us before, 2 W 400us before, 3 W 1000us before, 4: W back-to-back (queued right before)
      if (idle < 5000 && mode) continue;
      std::vector<double> tw;
      for (int it = 0; it < 60; it++) {
        double pre = mode == 1 ? 100 : mode == 2 ? 400 : mode == 3 ? 1000 : 0;
        spin(idle - pre);
        if (mode >= 1 && mode <= 3) { nrta_seq_t s = launch(W); spin(pre); (void)s; }
        if (mode == 4) launch(W);
        double t0 = now();
        waitc(launch(A));
        tw.push_back((now() - t0) * 1e6);
      }
      std::sort(tw.begin(), tw.end());
      printf("idle %6.0f us mode %d: A total p50 %7.1f p10 %7.1f p90 %7.1f\n", idle, mode, tw[30], tw[6], tw[54]);
      fflush(stdout);
    }
  }
  nrt_close();
}
