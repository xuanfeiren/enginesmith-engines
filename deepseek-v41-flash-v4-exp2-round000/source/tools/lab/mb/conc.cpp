#include <nrt/nrt.h>
#include <nrt/nrt_experimental.h>
#include <nrt/nrt_async.h>
#include <sys/eventfd.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <thread>
#include <atomic>
#include <sched.h>
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
#define CK(x) do { NRT_STATUS s_=(x); if (s_) { fprintf(stderr, "%s -> %d\n", #x, (int)s_); exit(1);} } while(0)
struct M { nrt_tensor_t* xin = nullptr; nrt_tensor_t* yout = nullptr; nrt_model_t* m; nrt_tensor_set_t *in, *out; std::vector<nrt_tensor_set_t*> ins; };
static int KSETS = getenv("KSETS") ? atoi(getenv("KSETS")) : 1;
static M load(const char* path, int core) {
  std::ifstream f(path, std::ios::binary);
  std::vector<char> b((std::istreambuf_iterator<char>(f)), {});
  M r; CK(nrt_load(b.data(), b.size(), core, 1, &r.m));
  nrt_tensor_info_array_t* info; CK(nrt_get_model_tensor_info(r.m, &info));
  CK(nrt_allocate_tensor_set(&r.in)); CK(nrt_allocate_tensor_set(&r.out));
  for (int k = 0; k < KSETS; k++) { nrt_tensor_set_t* s; CK(nrt_allocate_tensor_set(&s)); r.ins.push_back(s); }
  for (uint64_t i = 0; i < info->tensor_count; i++) {
    auto& t = info->tensor_array[i];
    nrt_tensor_t* x; CK(nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, core, t.size, t.name, &x));
    CK(nrt_tensor_memset(x, 0, 0, t.size));
    CK(nrt_add_tensor_to_tensor_set(t.usage == NRT_TENSOR_USAGE_INPUT ? r.in : r.out, t.name, x));
    if (std::string(t.name) == "x") r.xin = x;
    if (t.usage != NRT_TENSOR_USAGE_INPUT) r.yout = x;
    if (t.usage == NRT_TENSOR_USAGE_INPUT)
      for (int k = 0; k < KSETS; k++) {
        nrt_tensor_t* y = x;
        if (k > 0 && t.size > 1000000) { CK(nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, core, t.size, t.name, &y)); CK(nrt_tensor_memset(y, 0, 0, t.size)); }
        CK(nrt_add_tensor_to_tensor_set(r.ins[k], t.name, y));
      }
  }
  // warm every set once
  for (int k = 0; k < KSETS; k++) { NRT_STATUS ret; nrta_seq_t seq; CK(nrta_execute_schedule(r.m, r.ins[k], r.out, 0, &ret, &seq)); bool d = false; while (!d) CK(nrta_is_completed(seq, &d)); }
  return r;
}
int main(int argc, char** argv) {
  // usage: conc NEFF mode ncores core0 stride   (mode 0 pause-poll, 1 tight poll, 2 eventfd, 3 usleep poll)
  const char* path = argv[1]; int mode = atoi(argv[2]); int n = atoi(argv[3]); int c0 = atoi(argv[4]); int st = atoi(argv[5]);
  CK(nrt_init(NRT_FRAMEWORK_TYPE_NO_FW, "", ""));
  std::vector<M> ms;
  for (int i = 0; i < n; i++) ms.push_back(load(path, c0 + i * st));
  const char* other = getenv("OTHER");
  std::vector<M> os;
  if (other) for (int i = 0; i < n; i++) os.push_back(load(other, c0 + i * st));
  std::atomic<int> go{0}, ready{0};
  std::atomic<int> bar{0};
  bool sync = getenv("SYNC") != nullptr;
  std::vector<double> maxv(n);
  const int IT0 = getenv("ITERS") ? atoi(getenv("ITERS")) : 400;
  std::vector<std::vector<double>> donet(IT0, std::vector<double>(n)), schedt(IT0, std::vector<double>(n));
  std::vector<double> tbar(IT0);
  std::vector<double> res(n), sch(n);
  std::vector<std::thread> th;
  const int IT = getenv("ITERS") ? atoi(getenv("ITERS")) : 400;
  for (int i = 0; i < n; i++) th.emplace_back([&, i] {
    int efd = eventfd(0, 0);
    if (getenv("PIN")) { cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(atoi(getenv("PIN")) + i * 2, &cs); sched_setaffinity(0, sizeof cs, &cs); }
    ready++;
    while (!go) {}
    double tot = 0, ts = 0;
    double mx = 0;
    for (int it = 0; it < IT; it++) {
      if (sync) {   // all threads start each iteration together
        int target = (it + 1) * n;
        bar.fetch_add(1);
        while (bar.load() < target) __builtin_ia32_pause();
        if (i == 0) tbar[it] = now();
      }
      if (getenv("GAP")) { int g = atoi(getenv("GAP")); if (getenv("GAPSPIN")) { double a = now(); while (now() - a < g * 1e-6) {} } else usleep(g); }
      if (other && getenv("WARMX")) {   // warm NEFF, then wait WARMX us after it completed
        NRT_STATUS r2; nrta_seq_t s2; CK(nrta_execute_schedule(os[i].m, os[i].in, os[i].out, 0, &r2, &s2));
        if (!getenv("NOWAITWARM")) { bool d = false; while (!d) CK(nrta_is_completed(s2, &d)); }
        double a = now(); while (now() - a < atof(getenv("WARMX")) * 1e-6) {}
      }
      double tstart = now();
      if (getenv("STAG")) { double w = i * atof(getenv("STAG")) * 1e-6; while (now() - tstart < w) {} }
      static thread_local NRT_STATUS ret; nrta_seq_t seq;
      if (other && !getenv("WARMX")) { NRT_STATUS r2; nrta_seq_t s2; CK(nrta_execute_schedule(os[i].m, os[i].in, os[i].out, 0, &r2, &s2)); bool d = false; while (!d) CK(nrta_is_completed(s2, &d)); }
      if (mode == 4) {
        if (i == 0) {
          std::vector<nrta_seq_t> sq(n); std::vector<NRT_STATUS> rr(n);
          double tq = now();
          for (int k = 0; k < n; k++) CK(nrta_execute_schedule(ms[k].m, ms[k].in, ms[k].out, 0, &rr[k], &sq[k]));
          std::vector<bool> dn(n, false); int left = n;
          while (left) for (int k = 0; k < n; k++) if (!dn[k]) { bool d; CK(nrta_is_completed(sq[k], &d)); if (d) { dn[k] = true; left--; donet[it][k] = now() - tq; } }
        }
        continue;
      }
      double t0 = now();
      static thread_local std::vector<char> buf(16384);
      if (getenv("WR") && ms[i].xin) CK(nrt_tensor_write(ms[i].xin, buf.data(), 0, 10240));
      if (mode == 5) { CK(nrt_execute(ms[i].m, ms[i].ins[(it * 7919) % KSETS], ms[i].out)); }
      else CK(nrta_execute_schedule(ms[i].m, ms[i].ins[(it * 7919) % KSETS], ms[i].out, 0, &ret, &seq));
      double t1 = now();
      if (mode == 5) {
      } else if (mode == 2) {
        CK(nrta_event_register_seq_id_completion(seq, efd));
        uint64_t v; if (read(efd, &v, 8) != 8) exit(2);
      } else {
        bool done = false;
        for (int k = 0;; k++) {
          CK(nrta_is_completed(seq, &done));
          if (done) break;
          if (mode == 0) { int sp = k < 64 ? 32 : 256; for (int q = 0; q < sp; q++) __builtin_ia32_pause(); }
          else if (mode == 3) usleep(3);
        }
      }
      if (getenv("WR") && ms[i].yout) CK(nrt_tensor_read(ms[i].yout, buf.data(), 0, 10240));
      double t2 = now();
      donet[it][i] = t2 - tstart; schedt[it][i] = t1 - tstart;
      if (it >= 50) { tot += t2 - t0; ts += t1 - t0; }
    }
    res[i] = tot / (IT - 50) * 1e6; sch[i] = ts / (IT - 50) * 1e6;
  });
  while (ready < n) {}
  go = 1;
  for (auto& t : th) t.join();
  if (sync) {
    std::vector<double> rk(n, 0), rs(n, 0);
    for (int it = 50; it < IT; it++) { auto v = donet[it]; std::sort(v.begin(), v.end()); auto w = schedt[it]; std::sort(w.begin(), w.end()); for (int k = 0; k < n; k++) { rk[k] += v[k]; rs[k] += w[k]; } }
    std::vector<double> bi(n, 0);
    for (int it = 50; it < IT; it++) for (int k = 0; k < n; k++) bi[k] += donet[it][k];
    printf("  done by thread:"); for (int k = 0; k < n; k++) printf(" %.0f", bi[k] / (IT - 50) * 1e6);
    printf("  done by rank:"); for (int k = 0; k < n; k++) printf(" %.0f", rk[k] / (IT - 50) * 1e6);
    printf("   sched-end by rank:"); for (int k = 0; k < n; k++) printf(" %.0f", rs[k] / (IT - 50) * 1e6); printf("\n");
  }
  double a = 0, b = 0; for (int i = 0; i < n; i++) { a += res[i]; b += sch[i]; }
  printf("mode %d n %d: iter %.1f us  sched %.1f us\n", mode, n, a / n, b / n);
  nrt_close();
}
