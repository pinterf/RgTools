// RgTools core test: the filter cores without a host (no AviSynth), at any bit depth.
//
// Every filter and mode on random planes (8 bit, 9..16 bit, 32 bit float; values weighted to 0 and the maximum),
// YUV 4:2:0, 4:1:1, 4:4:0, 4:1:0 (luma and chroma plane) and Gray, plane sizes from 1x1. Each SIMD level the CPU has is compared to C
// (bit-exact). The hash of all C outputs is printed: it must be the same on every platform (x86, aarch64).
// Exit code: 0 if all SIMD outputs equal the C output.

#include "rg_core.h"
#include "clense_core.h"
#include "removegrain_core.h"
#include "RemoveGrainT_core.h"
#include "repair_core.h"
#include "vertical_cleaner_core.h"

#include <cinttypes>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace {

struct Level { const char* name; int cpu_flags; };

std::vector<Level> simd_levels() {
  std::vector<Level> levels;
#if defined(INTEL_INTRINSICS) && (defined(__GNUC__) || defined(__clang__))
  __builtin_cpu_init();
  levels.push_back({ "SSE2", RG_CPU_SSE2 });
  if (__builtin_cpu_supports("sse4.1"))
    levels.push_back({ "SSE4.1", RG_CPU_SSE2 | RG_CPU_SSE41 });
  if (__builtin_cpu_supports("sse4.1") && __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
    levels.push_back({ "AVX2", RG_CPU_SSE2 | RG_CPU_SSE41 | RG_CPU_AVX2 });
#endif
  return levels;
}

// a plane in a padded, 64 byte aligned buffer (the SIMD kernels may read and write past rowsize, within the pitch)
struct Buffer {
  std::vector<uint8_t> mem;
  uint8_t* ptr = nullptr;
  int pitch = 0;
  int rowsize = 0;
  int height = 0;

  Buffer(int rowsize_, int height_) : rowsize(rowsize_), height(height_) {
    pitch = (rowsize + 64 + 63) / 64 * 64;
    mem.resize((size_t)pitch * height + 256);
    uintptr_t p = (uintptr_t)mem.data();
    ptr = mem.data() + ((64 - p % 64) % 64);
  }
  uint8_t* row(int y) { return ptr + (size_t)y * pitch; }
};

int pixelsize_of(int bits) { return bits == 8 ? 1 : bits == 32 ? 4 : 2; }

void fill(Buffer& b, int bits, bool chroma, std::mt19937& rng) {
  const int ps = pixelsize_of(bits);
  const int width = b.rowsize / ps;
  for (int y = 0; y < b.height; ++y) {
    uint8_t* r = b.row(y);
    for (int x = 0; x < width; ++x) {
      const unsigned kind = rng() % 8; // 0, 1: near 0; 2, 3: near the maximum; else anywhere
      if (bits == 32) {
        float v = kind < 2 ? (rng() % 4) / 1024.0f : kind < 4 ? 1.0f - (rng() % 4) / 1024.0f : (rng() % 65536) / 65535.0f;
        if (chroma)
          v -= 0.5f;
        reinterpret_cast<float*>(r)[x] = v;
      }
      else {
        const int max = (1 << bits) - 1;
        const int v = kind < 2 ? (int)(rng() % 4) : kind < 4 ? max - (int)(rng() % 4) : (int)(rng() % (max + 1));
        if (ps == 1)
          r[x] = (uint8_t)v;
        else
          reinterpret_cast<uint16_t*>(r)[x] = (uint16_t)v;
      }
    }
  }
}

uint64_t fnv1a(uint64_t h, const uint8_t* p, size_t n) {
  for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 0x100000001b3ULL; }
  return h;
}

RgFormat make_format(RgColorFamily family, int ssw, int ssh, int bits, int width, int height) {
  RgFormat f = {};
  f.family = family;
  f.bits_per_pixel = bits;
  f.pixelsize = pixelsize_of(bits);
  f.width = width;
  f.height = height;
  f.ssw = ssw;
  f.ssh = ssh;
  f.has_alpha = false;
  return f;
}

struct Stats { int configs = 0; int compared = 0; int mismatches = 0; int errors = 0; uint64_t c_hash = 0xcbf29ce484222325ULL; };

std::vector<Level> g_levels;

// Runs one filter configuration: inputs[0] is the source, the others the reference planes (per plane index).
// init(core, cpu_flags) -> error text; proc(core, plane_index, RgPlane, inputs) processes one plane.
template<class Core, class Init, class Proc>
void check(Stats& st, const std::string& cfg, const RgFormat& f, int num_inputs, Init init, Proc proc, std::mt19937& rng) {
  const int num_planes = f.family == RgColorFamily::Gray ? 1 : 2; // YUV: Y and U (V is the same as U)
  std::vector<std::vector<Buffer>> inputs(num_planes);
  for (int i = 0; i < num_planes; ++i) {
    const int w = i == 0 ? f.width : f.width >> f.ssw;
    const int h = i == 0 ? f.height : f.height >> f.ssh;
    for (int k = 0; k < num_inputs; ++k) {
      inputs[i].emplace_back(w * f.pixelsize, h);
      fill(inputs[i].back(), f.bits_per_pixel, i > 0 && f.family == RgColorFamily::YUV, rng);
    }
  }

  auto run = [&](int cpu_flags, std::vector<Buffer>& out) -> bool {
    Core core;
    const char* error = init(core, cpu_flags);
    if (error) {
      printf("ERROR %s: init: %s\n", cfg.c_str(), error);
      ++st.errors;
      return false;
    }
    for (int i = 0; i < num_planes; ++i) {
      Buffer& src = inputs[i][0];
      out.emplace_back(src.rowsize, src.height);
      memset(out.back().mem.data(), 0xCD, out.back().mem.size()); // mode -1, grey: the same unwritten content
      RgPlane p = { src.ptr, src.pitch, out.back().ptr, out.back().pitch, src.rowsize, src.height };
      proc(core, i, p, inputs[i]);
    }
    return true;
  };

  ++st.configs;
  std::vector<Buffer> ref;
  if (!run(0, ref))
    return;
  for (auto& b : ref)
    for (int y = 0; y < b.height; ++y)
      st.c_hash = fnv1a(st.c_hash, b.row(y), b.rowsize);

  for (const Level& level : g_levels) {
    std::vector<Buffer> out;
    if (!run(level.cpu_flags, out))
      return;
    ++st.compared;
    for (int i = 0; i < num_planes; ++i) {
      for (int y = 0; y < ref[i].height; ++y) {
        if (memcmp(ref[i].row(y), out[i].row(y), ref[i].rowsize) != 0) {
          if (st.mismatches < 20) {
            int x = 0;
            while (ref[i].row(y)[x] == out[i].row(y)[x]) ++x;
            printf("DIFF %s %s: plane %d row %d byte %d\n", cfg.c_str(), level.name, i, y, x);
          }
          ++st.mismatches;
          y = ref[i].height; i = num_planes; // one report per configuration and level
        }
      }
    }
  }
}

const int g_bits[] = { 8, 9, 10, 11, 12, 13, 14, 15, 16, 32 };
const int g_sizes[][2] = { { 1, 1 }, { 2, 3 }, { 4, 4 }, { 5, 7 }, { 17, 6 }, { 18, 5 }, { 33, 9 }, { 34, 8 }, { 66, 10 }, { 130, 9 } };
struct Family { const char* name; RgColorFamily family; int ssw, ssh; };
const Family g_families[] = {
  { "YUV420", RgColorFamily::YUV, 1, 1 }, { "YUV411", RgColorFamily::YUV, 2, 0 }, { "YUV440", RgColorFamily::YUV, 0, 1 },
  { "YUV410", RgColorFamily::YUV, 2, 2 }, { "Gray", RgColorFamily::Gray, 0, 0 } };

// the frame size is a multiple of the subsampling (as in the hosts)
template<class Body>
void for_formats(Body body) {
  for (const Family& fam : g_families)
    for (int bits : g_bits)
      for (auto& s : g_sizes) {
        const int w = (s[0] + (1 << fam.ssw) - 1) >> fam.ssw << fam.ssw;
        const int h = (s[1] + (1 << fam.ssh) - 1) >> fam.ssh << fam.ssh;
        body(make_format(fam.family, fam.ssw, fam.ssh, bits, w, h));
      }
}

const char* family_name(const RgFormat& f) {
  for (const Family& fam : g_families)
    if (fam.family == f.family && fam.ssw == f.ssw && fam.ssh == f.ssh)
      return fam.name;
  return "?";
}

std::string cfg_name(const char* filter, int mode, const RgFormat& f) {
  char buf[128];
  snprintf(buf, sizeof(buf), "%s mode %d, %s %d bit %dx%d", filter, mode, family_name(f), f.bits_per_pixel, f.width, f.height);
  return buf;
}

void report(const char* filter, const Stats& st) {
  printf("%-16s configs %5d  SIMD runs %6d  mismatches %3d  errors %d  C hash %016" PRIx64 "\n", filter, st.configs, st.compared,
    st.mismatches, st.errors, st.c_hash);
}

} // namespace

int main() {
  g_levels = simd_levels();
  printf("SIMD levels:");
  for (const Level& l : g_levels) printf(" %s", l.name);
  printf(g_levels.empty() ? " none (C only)\n" : "\n");

  std::mt19937 rng(20261029);
  int failed = 0;

  { // RemoveGrain
    Stats st;
    for (int mode = -1; mode <= 28; ++mode)
      for_formats([&](const RgFormat& f) {
        check<RemoveGrainCore>(st, cfg_name("RemoveGrain", mode, f), f, 1,
          [&](RemoveGrainCore& c, int cpu) { return c.init({ mode, mode, mode, 0 }, f, cpu); },
          [](RemoveGrainCore& c, int i, const RgPlane& p, std::vector<Buffer>&) { c.process_plane(i, p); }, rng);
      });
    report("RemoveGrain", st);
    failed += st.mismatches + st.errors;
  }
  { // Repair
    Stats st;
    for (int mode = -1; mode <= 28; ++mode) {
      if (mode == 25) continue;
      for_formats([&](const RgFormat& f) {
        check<RepairCore>(st, cfg_name("Repair", mode, f), f, 2,
          [&](RepairCore& c, int cpu) { return c.init({ mode, mode, mode, 0 }, f, f, cpu); },
          [](RepairCore& c, int i, const RgPlane& p, std::vector<Buffer>& in) { c.process_plane(i, p, in[1].ptr, in[1].pitch); }, rng);
      });
    }
    report("Repair", st);
    failed += st.mismatches + st.errors;
  }
  { // VerticalCleaner
    Stats st;
    for (int mode = -1; mode <= 2; ++mode)
      for_formats([&](const RgFormat& f) {
        check<VerticalCleanerCore>(st, cfg_name("VerticalCleaner", mode, f), f, 1,
          [&](VerticalCleanerCore& c, int cpu) { return c.init({ mode, mode, mode }, f, cpu); },
          [](VerticalCleanerCore& c, int i, const RgPlane& p, std::vector<Buffer>&) { c.process_plane(i, p); }, rng);
      });
    report("VerticalCleaner", st);
    failed += st.mismatches + st.errors;
  }
  { // Clense, ForwardClense, BackwardClense
    Stats st;
    const ClenseMode modes[] = { ClenseMode::BOTH, ClenseMode::FORWARD, ClenseMode::BACKWARD };
    for (int m = 0; m < 3; ++m)
      for (int grey = 0; grey <= 1; ++grey)
        for_formats([&](const RgFormat& f) {
          check<ClenseCore>(st, cfg_name(grey ? "Clense grey" : "Clense", m, f), f, 3,
            [&](ClenseCore& c, int cpu) { return c.init({ modes[m], grey != 0 }, f, cpu); },
            [](ClenseCore& c, int i, const RgPlane& p, std::vector<Buffer>& in) {
              c.process_plane(i, p, in[1].ptr, in[1].pitch, in[2].ptr, in[2].pitch); }, rng);
        });
    report("Clense", st);
    failed += st.mismatches + st.errors;
  }
  { // TemporalRepair
    Stats st;
    for (int mode = 0; mode <= 4; ++mode)
      for_formats([&](const RgFormat& f) {
        check<TemporalRepairCore>(st, cfg_name("TemporalRepair", mode, f), f, 4,
          [&](TemporalRepairCore& c, int cpu) { return c.init({ mode, false, -1 }, f, cpu); },
          [](TemporalRepairCore& c, int i, const RgPlane& p, std::vector<Buffer>& in) {
            c.process_plane(i, p, in[1].ptr, in[1].pitch, in[2].ptr, in[2].pitch, in[3].ptr, in[3].pitch); }, rng);
      });
    report("TemporalRepair", st);
    failed += st.mismatches + st.errors;
  }

  printf(failed ? "FAILED\n" : "OK\n");
  return failed ? 1 : 0;
}
