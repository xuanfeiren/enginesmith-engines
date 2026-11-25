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
int main(int argc, char** argv) {
  int core = atoi(argv[1]);
  CK(nrt_init(NRT_FRAMEWORK_TYPE_NO_FW, "", ""));
  for (int a = 2; a < argc; a++) {
    std::ifstream f(argv[a], std::ios::binary);
    std::vector<char> b((std::istreambuf_iterator<char>(f)), {});
    nrt_model_t* m; CK(nrt_load(b.data(), b.size(), core, 1, &m));
    nrt_tensor_info_array_t* info; CK(nrt_get_model_tensor_info(m, &info));
    nrt_tensor_set_t *in, *out; CK(nrt_allocate_tensor_set(&in)); CK(nrt_allocate_tensor_set(&out));
    int nin = 0, nout = 0;
    for (uint64_t i = 0; i < info->tensor_count; i++) {
      auto& t = info->tensor_array[i];
      nrt_tensor_t* x; CK(nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, core, t.size, t.name, &x));
      CK(nrt_tensor_memset(x, 0, 0, t.size));
      if (t.usage == NRT_TENSOR_USAGE_INPUT) { CK(nrt_add_tensor_to_tensor_set(in, t.name, x)); nin++; }
      else { CK(nrt_add_tensor_to_tensor_set(out, t.name, x)); nout++; }
    }
    std::vector<double> ts, tw;
    for (int it = 0; it < 300; it++) {
      NRT_STATUS ret; nrta_seq_t seq;
      double t0 = now();
      CK(nrta_execute_schedule(m, in, out, 0, &ret, &seq));
      double t1 = now();
      bool done = false;
      while (!done) CK(nrta_is_completed(seq, &done));
      double t2 = now();
      if (it >= 20) { ts.push_back((t1 - t0) * 1e6); tw.push_back((t2 - t0) * 1e6); }
    }
    std::sort(ts.begin(), ts.end()); std::sort(tw.begin(), tw.end());
    printf("%-40s in %3d out %2d  sched p50 %6.1f us  total p50 %7.1f us p10 %7.1f\n", argv[a], nin, nout, ts[ts.size()/2], tw[tw.size()/2], tw[tw.size()/10]);
    nrt_free_model_tensor_info(info);
  }
  nrt_close();
}
