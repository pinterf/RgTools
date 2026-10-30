## RgTools

RgTools is a modern rewrite of RemoveGrain, Repair, BackwardClense, Clense, ForwardClense, VerticalCleaner and
TemporalRepair in a single plugin for AviSynth+ (and classic AviSynth 2.6). RgTools is mostly backward compatible
with the original plugins.

Some routines might be slightly less efficient than the original, some are faster. The output of a few RemoveGrain
modes is not exactly identical to the original due to minor rounding differences. Other functions should be
identical.

This plugin is written from scratch and licensed under the [MIT license][1]. Some modes of RemoveGrain and Repair
were taken from Firesledge's Dither package.

Source: [https://github.com/pinterf/RgTools][6] (originally by tp7: [https://github.com/tp7/RgTools][7]).

### Contents

- [Formats](#formats)
- [CPU support, the opt parameter](#cpu-support-the-opt-parameter)
- [Functions](#functions): [RemoveGrain](#removegrain), [Repair](#repair), [Clense, ForwardClense, BackwardClense](#clense-forwardclense-backwardclense), [VerticalCleaner](#verticalcleaner), [TemporalRepair](#temporalrepair)
- [Expr equivalents of RemoveGrain modes](#expr-equivalents-of-removegrain-modes)
- [Building](#building)
- [Change log](#change-log)

### Formats

* Planar formats only: Y (greyscale), YUV 4:2:0, 4:2:2, 4:4:4, and the AviSynth+ V13 formats YUV 4:1:1, 4:4:0,
  4:1:0, YA (Y with alpha); YUVA, planar RGB and RGBA.
* Bit depths: 8, 10, 12, 14, 16 bit integer and 32 bit float. 32 bit float chroma is processed in the -0.5..0.5
  range.
* The alpha plane is copied (TemporalRepair modes 0 and 4: from the original clip, see there).
* YUY2 and packed RGB (RGB24, RGB32, ...) give an error. Convert them first, e.g. `ConvertToYV16()`.
  The `planar` parameter of all filters is a dummy, it exists for compatibility reasons only (in the old
  RemoveGrain package it enabled the "planar YUY2" layout of Interleaved2Planar, which is not supported).
* Frames must be aligned: AviSynth+ frames always are. Classic AviSynth 2.6: do not use an unaligned crop
  (use `Crop(..., align=true)`).

### CPU support, the opt parameter

The optimizations are chosen at runtime from the CPU flags the host reports (`SetMaxCPU` in AviSynth+ applies,
`SetMaxCPU("none")` gives the C code):

* x86 / x64: SSE2, SSE4.1, AVX2 (RemoveGrain only). 10-16 bit needs SSE4.1, otherwise C is used.
* aarch64 (Linux, Windows on ARM64): NEON for all filters and bit depths.
* Very narrow planes (below about one vector + 1 pixel, e.g. the chroma of a small 4:2:0 clip) are processed with C.

All code paths give the same result (C, SSE2, SSE4.1, AVX2, NEON; tested bit-exact, also for 32 bit float).

The `opt` parameter forces a code path, mainly for testing. **Its values differ between the filters**: TemporalRepair
got `opt` in v0.98, RemoveGrain and Repair later (v0.99, v1.0) with a different numbering. Both are kept for
script compatibility.

| opt | RemoveGrain, Repair | TemporalRepair |
|---|---|---|
| default | 0 | -1 |
| auto (the best available) | 0 | -1 |
| C | **1** | **0** |
| SSE2 (x86) | 2 | 1 |
| SSE4.1 (x86) | 3 | 2 |
| AVX2 (x86; RemoveGrain only, Repair has none) | 4 | - |
| NEON (aarch64) | any value except 1 | any value except 0 |

Clense, ForwardClense, BackwardClense and VerticalCleaner have no `opt` parameter.

## Functions

### RemoveGrain
```
RemoveGrain(clip, int "mode", int "modeU", int "modeV", bool "planar", int "opt")
```
Purely spatial denoising function with 28 modes. Each pixel is processed in its 3x3 neighbourhood; the border
pixels (the first and last row and column) are copied. Additional info can be found in the [wiki][2].

* `mode` (default 1): the mode of the Y plane (planar RGB: all planes).
* `modeU` (default: `mode`), `modeV` (default: `modeU`): the modes of the U and V planes. Not allowed for planar RGB.
* `planar`: dummy, exists for compatibility reasons. `opt`: see [above](#cpu-support-the-opt-parameter).

Modes (descriptions mostly from the original author, Kassandro):

| mode | type | description |
|---|---|---|
| -1 | none | The plane is not processed at all: its content is undefined (random). Use it only when the plane is not needed later, e.g. `modeU=-1` and a greyscale filter afterwards. Faster than mode 0. |
| 0 | copy | The plane is copied. |
| 1 | median blur | Clipped to the range of its 8 neighbours. Same as Undot, but faster (single dots). |
| 2 | median blur | Clipped to the range of the second lowest and second highest neighbour (1x2 spots). |
| 3 | median blur | As 2, with the third lowest / highest (3 pixel clusters; some artifact risk). |
| 4 | median blur | As 2, with the fourth lowest / highest (up to 2x2 pixel clusters; artifact risk). |
| 5 | edge sensitive median blur | Only the 4 line pairs through the center are used. Strong edge protection. |
| 6 | edge sensitive median blur | As 5, fair edge protection. |
| 7 | edge sensitive median blur | As 5, mild edge protection. |
| 8 | edge sensitive median blur | As 5, faint edge protection. |
| 9 | edge sensitive median blur | As 5, barely any edge protection. Practically a spatial variant of trbarry's ST Median filter. |
| 10 | minimal sharpening | Replaced by its nearest neighbour ("very poor denoise sharpener"). |
| 11 | blur | 3x3 kernel (1 2 1, 2 4 2, 1 2 1)/16. Better than Blur(1.0). |
| 12 | blur | The same kernel computed with averages, faster; differs from mode 11 by at most 1. |
| 13 | smart bob (interlaced content) | Interpolates the top field (similar to Trbarry's weird bob / Tomsmocomp). |
| 14 | smart bob (interlaced content) | Interpolates the bottom field. |
| 15 | smart bob (interlaced content) | As 13, better quality, slightly slower. |
| 16 | smart bob (interlaced content) | As 14, better quality, slightly slower. |
| 17 | median blur | As 4, but with better edge protection (similar to the almost artifact free mode 2). Probably the best mode of all. |
| 18 | edge sensitive median blur | As 9, but with better edge protection (what 17 is to 4); far less denoising than 17. |
| 19 | blur | The average of the 8 neighbours (without the center). |
| 20 | blur | The uniform average of the 3x3 pixels. Better than 19 but slower; very similar to Blur(1.58). |
| 21 | median blur | Clipped to the range of the averages of the neighbour pairs. Best for cartoons. |
| 22 | median blur | As 21, much faster (fastest mode of all). |
| 23 | dehalo | Fixes small (one pixel wide) haloes. |
| 24 | dehalo | As 23, considerably more conservative, slightly slower. Preferred. |
| 25 | minimal sharpening | "Non destructive sharpen". |
| 26 | median blur | SmartRGC: based on mode 17, preserves corners but not thin lines. |
| 27 | median blur | SmartRGCL: as 26, preserves thin lines as well. |
| 28 | median blur | SmartRGCL2: a variant of 27. |

Mostly used as of 2026: in a collection of about 800 widely used AviSynth scripts (the usual script function
packages), the RemoveGrain calls with a literal mode use almost only these modes, in this order of frequency:

| mode | type | used in scripts |
|---|---|---|
| 11 | blur | 38 |
| 20 | blur | 35 |
| 4 | median blur | 34 |
| 12 | blur | 28 |
| 2 | median blur | 18 |
| 19 | blur | 15 |
| 1 | median blur | 11 |
| 17 | median blur | 9 |
| 5 | edge sensitive median blur | 7 |

Besides these, -1 (the plane is skipped) and 0 (copy) are common (12 and 7 scripts), to process only luma or only chroma. Modes 7, 21 and 22 occur
in a few scripts; the other modes were not found with a literal value (some scripts take the mode from a parameter).

Recommended modes: 12, 20 (gaussian-like blur); 17, 22 (median blur).

From the original documentation: "As far as compression (denoising) is concerned, my benchmarks give the following
mode ranking: 4, 17, 9, 8, 3, 7, 6, 2, 5, 1, but modes 4 and 17 really stand out. As far as artifacts are concerned,
we have unfortunately almost the reverse ranking: 1, 5, 2, 18, 6, 7, 8, 17, 3, 4, 9. Modes 1, 5, 2 are the risk free
modes, the modes 18, 6, 7, 8, 17 show low to moderate artifact risk (usually some softness) and modes 3, 4, 9 have
severe problems with thin lines. Mode 17 is my clear personal favorite."

| mode | sharp edges | corners | thin lines | line ends | thin curves | compression |
|---|---|---|---|---|---|---|
| 1 | 10 | 10 | 10 | 10 | 10 | 1 |
| 2 | 10 | 10 | 10 | 4 | 10 | 2 |
| 3 | 10 | 9 | 3 | 1 | 3 | 5 |
| 4 | 10 | 3 | 1 | 0 | 1 | 8 |
| 5 | 10 | 10 | 10 | 9 | 9 | 2 |
| 6 | 10 | 6 | 9 | 4 | 3 | 3 |
| 7 | 10 | 6 | 9 | 4 | 3 | 3 |
| 8 | 10 | 5 | 8 | 3 | 3 | 4 |
| 9 | 10 | 3 | 4 | 1 | 2 | 5 |
| 10 | 10 | 10 | 10 | 10 | 10 | 1 |
| 11 | 1 | 1 | 2 | 1 | 2 | 9 |
| 12 | 1 | 1 | 2 | 1 | 2 | 9 |
| 17 | 10 | 2 | 8 | 2 | 4 | 7 |
| 18 | 10 | 6 | 9 | 6 | 5 | 2 |
| 19 | 1 | 1 | 1 | 0 | 1 | 10 |
| 20 | 1 | 1 | 1 | 0 | 1 | 8 |
| 21 | 6 | 2 | 8 | 4 | 4 | 5 |
| 22 | 6 | 2 | 8 | 4 | 4 | 5 |
| 23 | 6 | 5 | 6 | 4 | 6 | 4 |
| 24 | 7 | 6 | 7 | 5 | 7 | 3 |
| 25 | 10 | 10 | 10 | 10 | 10 | -1 |

### Repair
```
Repair(clip, clip, int "mode", int "modeU", int "modeV", bool "planar", int "opt")
```
Repairs unwanted artifacts of (but not limited to) RemoveGrain or temporal filters like Clense: the first clip
(the filtered one) is limited by the 3x3 neighbourhood of the same pixel in the second clip (the original).
Modes -1..28, except 25. The border pixels are copied from the first clip.

* The two clips must have the same size, bit depth and color family (YUV / RGB); when a chroma mode is active,
  the same format (also the chroma subsampling and alpha).
* `mode` (default 1), `modeU`, `modeV`, `planar`, `opt`: as at RemoveGrain.

"c": the center pixel of the original, "the 3x3": the center and its 8 neighbours in the original.

| mode | |
|---|---|
| -1, 0 | As RemoveGrain (not processed / copy). |
| 1 | Clipped to the range of the 3x3. `RemoveGrain(input, mode=n)` = `Repair(input, input, n+1)` for n = 0..3. |
| 2, 3, 4 | Clipped to the range of the second (third, fourth) lowest and highest of the 3x3. |
| 5-9 | As RemoveGrain 5-9 with the center included in each line pair. |
| 10 | Replaced by the pixel of the 3x3 nearest to it. |
| 11 | Same as 1. |
| 12, 13, 14 | As RemoveGrain 2, 3, 4 on the 8 neighbours, the range extended to c. |
| 15, 16 | As RemoveGrain 5, 6: the line pair is chosen by c, the pixel clipped to that line extended to c. Mode 16 restores thin lines, with a good chance even their end points. |
| 17, 18 | As RemoveGrain 17, 18, the range extended to c. |
| 19 | Clipped to c +/- (the smallest difference between c and its neighbours). |
| 20 | As 19, with the second smallest difference. |
| 21 | Clipped to c +/- (the smallest distance of c to the range of a line pair). |
| 22, 23, 24 | As 19, 20, 21 the other way round: the differences are measured from the pixel of the first clip, and c is clipped. |
| 26, 27, 28 | As RemoveGrain 26, 27, 28, the range extended to c. |

The original author recommends modes 2, 5-9 for Repair after Clense, and mode 9 for taming sharpeners.

### Clense, ForwardClense, BackwardClense
```
Clense(clip, clip "previous", clip "next", bool "grey", bool "reduceflicker", bool "planar", int "cache")
ForwardClense(clip, bool "grey", bool "planar", int "cache")
BackwardClense(clip, bool "grey", bool "planar", int "cache")
```
Clense is a temporal median of three frames: each pixel is clipped to the range of the same pixel in the previous
and the next frame. Identical to `MedianBlurTemporal(0,0,0,1)` but a lot faster. It is a brutal cleaner and is
usually combined with Repair. Can be used as a building block for [many][3] [fancy][4] [medians][5].

ForwardClense uses the next two frames, BackwardClense the previous two (the range is extrapolated from them). They
are meant for clensing at scene changes: ForwardClense for the first frame of a new scene, BackwardClense for the
last frame of a scene.

* `previous`, `next` (Clense only): take the previous / the next frame from these clips instead of the input
  (same size and format as the input).
* `grey` (default false): only Y is processed, U and V are neither processed nor copied (random content, for
  speed; use a greyscale filter or MergeChroma later). The alpha plane is copied. Not allowed for planar RGB.
* `reduceflicker` (Clense only, default false): the previous frame is taken from the previous output frame,
  when the frames are requested sequentially. Then the filter reports MT_MULTI_INSTANCE for AviSynth+ MT
  (MT_NICE_FILTER otherwise, as all other filters).
* The edge frames are returned unchanged: Clense: the first and the last frame; ForwardClense: the last two;
  BackwardClense: the first two.
* `planar`, `cache`: dummy parameters, exist for compatibility reasons.

### VerticalCleaner
```
VerticalCleaner(clip, int "mode", int "modeU", int "modeV", bool "planar")
```
Very fast vertical median filter.

* mode (default 1), modeU, modeV: -1 (not processed, as at RemoveGrain), 0 (copy), 1: the median of the pixel and
  the pixels above and below, 2: relaxed vertical median (two rows above and below are used, preserves more
  detail). The border rows (1 at mode 1, 2 at mode 2) are copied. modeU / modeV are not allowed for planar RGB.
* `planar`: dummy, exists for compatibility reasons.

### TemporalRepair
```
TemporalRepair(clip, clip, int "mode", int "smooth", bool "grey", bool "planar", int "opt")
```
In the same way as Repair is derived from RemoveGrain, TemporalRepair is derived from Clense. While Repair is a
spatial filter most suitable for removing artifacts of temporal filters like Clense, TemporalRepair is a temporal
filter, primarily useful for restoring static (non moving) details of spatial filters like RemoveGrain. The first
clip is the filtered one, the second the original. Example from the original documentation:
```
function RemoveTemporalGrain(clip input, int _mode)
{
  rg = RemoveGrain(input, mode=_mode)
  return TemporalRepair(rg, input)
}
```

* `mode` (`smooth` is the same parameter, an older name; default 0):
  * 0: the pixel is clipped to the range of the original's previous, current and next frame.
  * 1, 2, 3: also the temporal changes of the 8 neighbours are taken into account, so less is restored than with
    mode 0. Mode 2 restores the least and is the safest for deinterlacing; mode 3 preserves the most detail.
    The border pixels are copied from the first clip.
  * 4: as 0, with a narrower range: the range of the original's previous and next frame mirrored at the original's
    current pixel (and limited to that range); where the current pixel is at an edge of the range, it is taken.
* `grey` (default false): only Y is processed (U and V: random content, for speed). The alpha plane is copied.
* `opt`: **different values than at RemoveGrain / Repair**: -1 = auto (default), 0 = C, 1 = SSE2, 2 = SSE4.1
  (see [the table](#cpu-support-the-opt-parameter)).
* The first and the last frame are returned unchanged (from the first clip). The length is that of the shorter clip.
* Alpha: modes 0 and 4 copy the alpha plane from the original (the second clip), modes 1-3 from the first clip.
* `planar`: dummy, exists for compatibility reasons.

## Expr equivalents of RemoveGrain modes

For some RemoveGrain modes an [Expr](http://avisynth.nl/index.php/Expr) expression with relative pixel addressing
(`x[dx,dy]`) gives the same result for the inner pixels (all bit depths unless noted, tested bit-exact). The border
pixels differ: RemoveGrain copies them, Expr takes the off-screen neighbours by its border mode (`:c` clone,
default, `:m`, `:r` mirror, `:k` constant, `:w` wrap; AviSynth+ 3.7.6). The other modes (sorting networks,
selections with tie order, cascaded rounding) have no practical Expr form.

Neighbours: `x[-1,-1] x[0,-1] x[1,-1]` / `x[-1,0] x x[1,0]` / `x[-1,1] x[0,1] x[1,1]`.

* Mode 1 (clipped to the range of the 8 neighbours):
  ```
  x[-1,-1] x[0,-1] min x[1,-1] min x[-1,0] min x[1,0] min x[-1,1] min x[0,1] min x[1,1] min L^ x[-1,-1] x[0,-1] max x[1,-1] max x[-1,0] max x[1,0] max x[-1,1] max x[0,1] max x[1,1] max U^ x L U clip
  ```
* Mode 11 (3x3 kernel), integer formats:
  ```
  x 4 * x[0,-1] x[-1,0] + x[1,0] + x[0,1] + 2 * + x[-1,-1] + x[1,-1] + x[-1,1] + x[1,1] + 8 + 16 / floor
  ```
  32 bit float:
  ```
  x 4 * x[0,-1] x[-1,0] + x[1,0] + x[0,1] + 2 * + x[-1,-1] + x[1,-1] + x[-1,1] + x[1,1] + 16 /
  ```
* Mode 17:
  ```
  x[-1,-1] x[1,1] min x[0,-1] x[0,1] min max x[1,-1] x[-1,1] min max x[-1,0] x[1,0] min max L^ x[-1,-1] x[1,1] max x[0,-1] x[0,1] max min x[1,-1] x[-1,1] max min x[-1,0] x[1,0] max min U^ x L U min L U max clip
  ```
* Mode 20 (3x3 average), 8-12 bit (at 14 and 16 bit RgTools divides by 9 with a fixed point multiplication that
  can give 1 more):
  ```
  x[-1,-1] x[0,-1] + x[1,-1] + x[-1,0] + x + x[1,0] + x[-1,1] + x[0,1] + x[1,1] + 4 + 9 / floor
  ```
  32 bit float (the summation order and the multiplication by 1/9 as in RgTools):
  ```
  x[-1,-1] x[0,-1] + x[1,-1] x[-1,0] + + x[1,0] x[-1,1] + x[0,1] x[1,1] + + + x + 0.11111111 *
  ```
* Modes 26, 27, 28: the pattern of mode 17 (lower = the maximum of the pair minimums, upper = the minimum of the pair
  maximums, x clipped between them) with these neighbour pairs:
  * 26: the 8 neighbouring pairs around the center: (-1,-1)(0,-1), (0,-1)(1,-1), (1,-1)(1,0), (1,0)(1,1),
    (0,1)(1,1), (-1,1)(0,1), (-1,0)(-1,1), (-1,-1)(-1,0)
  * 28: the pairs of 26 and the 4 lines through the center: (-1,-1)(1,1), (1,-1)(-1,1), (0,-1)(0,1), (-1,0)(1,0)
  * 27: (-1,-1)(1,1), (-1,-1)(0,-1), (0,1)(1,1), (0,-1)(0,1), (0,-1)(1,-1), (-1,1)(0,1), (1,-1)(-1,1), (1,-1)(1,0),
    (-1,0)(-1,1), (-1,0)(1,0), (1,0)(1,1), (-1,-1)(-1,0)

  Mode 26 written out:
  ```
  x[-1,-1] x[0,-1] min x[0,-1] x[1,-1] min max x[1,-1] x[1,0] min max x[1,0] x[1,1] min max x[0,1] x[1,1] min max x[-1,1] x[0,1] min max x[-1,0] x[-1,1] min max x[-1,-1] x[-1,0] min max L^ x[-1,-1] x[0,-1] max x[0,-1] x[1,-1] max min x[1,-1] x[1,0] max min x[1,0] x[1,1] max min x[0,1] x[1,1] max min x[-1,1] x[0,1] max min x[-1,0] x[-1,1] max min x[-1,-1] x[-1,0] max min U^ x L U min L U max clip
  ```

## Building

### Windows, Visual Studio (no CMake for MSVC, use the solutions)

* Visual Studio 2026: `RgTools.slnx`, configurations `Release` (MSVC v145) and `Rel_Clang` (ClangCL, the Clang
  toolset that comes with Visual Studio), Win32 and x64.
* Visual Studio 2022: `RgTools.sln`, configuration `Release_v141_xp` (Windows XP and Windows 7), Win32 and x64.
  Needs the "C++ Windows XP Support for VS 2017 (v141) tools" component.
* Output: `Build\<platform>\<configuration>\RgTools.dll`.

### Windows on ARM64

`build_arm64_mingw.bat`: cross-compiled on x64 Windows with [llvm-mingw](https://github.com/mstorsjo/llvm-mingw)
(CMake, Ninja; set `LLVM_MINGW_ROOT`). Output: `out\ARM64\llvm-mingw\RgTools.dll`.

### Release pack

`build_release.bat`: the ClangCL, v141_xp and ARM64 builds and the documentation into `build\release\<version>\`
and a .7z archive.

### Windows MinGW (MSYS2 GCC or Clang), Linux, macOS (CMake, Ninja or Makefiles)
```
cmake -B build -S . -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build      (Linux: installs to <libdir>/avisynth)
```
The target processor is detected: x86 enables `ENABLE_INTEL_SIMD`, aarch64 `ENABLE_NEON_SIMD`. With
`-DENABLE_INTEL_SIMD=OFF` a plain C build is made.

### Core test

`-DRGTOOLS_BUILD_TESTS=ON` builds `rgtools_core_test` (`tests/core_test.cpp`): the filter cores without a host, all
modes at 8, 9-16 and 32 bit, YUV 4:2:0 / 4:1:1 / 4:4:0 / 4:1:0 and greyscale, many plane sizes; every SIMD level of
the CPU is compared with C. It prints a hash of the C results, which must be the same on every platform. For an
AddressSanitizer run add `-DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"`.

## Change log

v1.3 (20261030)

Bug fixes
- RemoveGrain, Repair, VerticalCleaner: crash (memory corruption, also hangs) on planes of 1 row, and VerticalCleaner
  mode 2 on planes of 2-3 rows (e.g. the chroma of a 2 or 4 row high 4:2:0 clip). A plane without an inner row is
  copied now. Existed since 2013.
- RemoveGrain modes 13-16 (field interpolation), all code paths: with an odd plane height one row was not written
  (random content): the last row (modes 13, 15) or the last odd row (modes 14, 16), e.g. the 4:2:0 chroma of
  720x486. Existed since 2013.
- RemoveGrain, Repair (SSE2/SSE4.1/AVX2): wrong left border pixel, depending on memory content, when a plane is
  exactly one vector + 1 pixel wide (8 bit: 17 or 33, 10-16 bit: 9 or 17, float: 5 or 9 pixels; e.g. the 4:2:0
  chroma of a 34 or 66 pixel wide clip). Existed since 2013.
- RemoveGrain, Repair, TemporalRepair modes 1-3 (SSE2/SSE4.1, RemoveGrain also AVX2): with a plane width of a multiple
  of the vector size, the last vector of a row read one pixel past the end of the row, in the last row past the end of the frame
  (possible access violation). The output was not affected. Existed since 2013.
- TemporalRepair modes 1-3 (SSE2/SSE4.1): the right border column was overwritten when the row was a multiple of
  16 bytes (e.g. the 4:2:2 chroma of a 64 pixel wide clip). Existed since v0.98.
- Repair: wrong output with a reference clip of a different frame pitch (e.g. a cropped reference) on the C code
  path (CPUs without SSE2, opt=1, planes narrower than 17 pixels). Existed since 2013.
- Repair: the two clips must have the same bit depth and color family also when chroma is not processed (Y clips,
  modeU=modeV=-1); was accepted with wrong output and reads past the reference rows. Existed since v0.93.
- ForwardClense, BackwardClense, 32 bit float YUV: negative chroma (U, V) was clipped to 0. Existed since the 32 bit
  float support.
- Clense, ForwardClense, BackwardClense: with grey=true the alpha plane is copied (was random content). Existed
  since v0.93.
- VerticalCleaner: a mode below -1 is an error (was accepted: undefined behaviour, possible crash). Existed since
  2013.
- RemoveGrain mode 6, 32 bit float, SSE4.1 (opt=3, or auto on CPUs without AVX2): the costs were not clamped at
  1.0 as on the other code paths; the result could differ. Existed since the float support.
- RemoveGrain mode 25, 10-14 bit, C code path: the result could differ from SIMD for some pixel patterns.
  Existed since the 10-16 bit support.

Changes
- RemoveGrain mode 11 on SSE2/SSE4.1/AVX2 (the x86 default): the exact (1 2 1)/16 kernel as documented and as the C
  code, (4*c + 2*(a2+a4+a5+a7) + a1+a3+a6+a8 + 8) / 16. Before, it ran the mode 12 approximation (difference
  at most 1). Mode 12 stays the faster approximation.
- RemoveGrain C code path (opt=1, CPUs without SSE2, very narrow planes): modes 12, 15, 16, 19 and 20 give the same
  result as SIMD (they differed by up to 1, in float in the last bits). All code paths give the same result now.
- YUY2 and packed RGB give an error also with planar=true ("use ConvertToYV16"). Before, planar=true skipped the
  format check and a YUY2 clip was processed as one plane.
- The "Invalid memory alignment. Unaligned crop?" errors are removed: AviSynth+ frames are always aligned
  (classic AviSynth 2.6: use an aligned crop).

New
- aarch64 NEON for all filters and bit depths (Linux, Windows on ARM64), the same result as C. Used when the host
  reports NEON (AviSynth+: CPUF_ARM_NEON; SetMaxCPU("none") gives C). opt on aarch64: RemoveGrain, Repair: 1 = C,
  any other value NEON; TemporalRepair: 0 = C, any other value NEON.
- The AviSynth+ V13 formats: YUV 4:1:1 at all bit depths, 4:4:0, 4:1:0 (with or without alpha) and YA (Y with alpha;
  the alpha plane is copied).
- Documentation: all filters, modes and parameters in README.md; Expr equivalents of some RemoveGrain modes.

Performance
- The row copies of the plane borders use memcpy instead of the AviSynth BitBlt.
- AVX2 (RemoveGrain): no manual vzeroupper (the compilers insert it; with Clang it caused saving and restoring all
  vector registers in every plane function).

Build environment
- Visual Studio 2026 solution (RgTools.slnx: Release with the v145 toolset, Rel_Clang with ClangCL); RgTools.sln
  (Visual Studio 2022) only for the v141_xp build (Windows XP and 7). The old separate LLVM configurations are removed.
- CMake: rewritten (AviSynth+ style); detects x86 and aarch64 (NEON); a plain C build is possible
  (ENABLE_INTEL_SIMD=OFF). GCC, Clang, MSVC: no warnings.
- Windows on ARM64 DLL (llvm-mingw cross build, build_arm64_mingw.bat); release pack script (build_release.bat).
- AviSynth headers updated to the V14 interface.
- Core test program (tests/core_test.cpp, CMake option RGTOOLS_BUILD_TESTS): SIMD vs C at all bit depths, also
  under AddressSanitizer / UBSan and on aarch64 (qemu).

Source
- The filter cores are separated from the AviSynth interface (*_core.cpp; prepares other front ends, e.g.
  VapourSynth). The x86, aarch64 and C code are in separate files.
- One 10-16 bit code path instead of separate 10, 12, 14 and 16 bit templates (also usable for 9-15 bit).

Note: the v1.3 work (bug hunt, NEON port, core separation, build, tests, documentation) was done with AI
assistance (Claude Code, Anthropic), reviewed and tested by pinterf.

v1.2 (20210526)
- Fix: garbage in 10+ bit Backward/ForwardClense for non SSE4.1 processors (C code part)

v1.1 (20210104)
- Source syntax for GCC
- CMake build environment, builds on Linux, at least on my (pinterf) Ubuntu 19.10 WSL
  (INTEL_INTRINSICS handling not implemented in the source, so it compiles on Intel at the moment)
- Fix: artifacts in RemoveGrain mode 19 for 10-16 bits

v1.0 (20200427)
- RemoveGrain mode 26-28, (C, up to SSE4.1)
  by the original author:
  Mode 26 SmartRGC: 26 = medianblur. Based off mode 17, but preserves corners, but not thin lines.
  Mode 27 SmartRGCL: 27 = medianblur. Same as mode 26 but preserves thin lines.
  Mode 28 SmartRGCL2

  I swear that by seeing the code mode 28 is different from mode 27 but I cannot find a clip where 27 and 28 gave different results :)

- Repair mode 26-28 (25 is nonexisting, only exists in RemoveGrain)
- add "opt" debug parameter to Repair. 0:auto, 1:c 2:SSE2 3:SSE4.1 4:AVX2 (latter is not used here atm)

v0.99 (20200426)
- RemoveGrain mode 25 ("nondestructivesharpen") finally reverse engineered 
  and ported from the original RemoveGrain package. (no-no, Repair has still 24 modes)
  Plus: 10-32 bits, Plus: sse4.1, avx2
- RemoveGrain: Revised, more correct 32 bit float in mode 6,7,8,9,15,16,23,24 
  (proper clamping, chroma part specially treated)
- General: make C reference more similar to actual SIMD code
- fix: width<32 (or 16) with horizontal subsampling would use AVX2 (SSE2) on chroma
- (noticed: RemoveGrain: mode 11 and 12 SSE is simply avaraging
   instead of really do 1/4 1/2 1/4 kernel blur like C (todo).
   This is written in docs but even very old RemoveGrainSSE2 acts like this)
- Avisynth+ V8 header, frame property copy support
- opt debug parameter to RemoveGrain. 0:auto, 1:c 2:SSE2 3:SSE4.1 4:AVX2. Replaces bool "optAvx2"

v0.98 (20190814)
- Include "TemporalRepair" filter from old RemoveGrainT package (rewritten C and SIMD intrinsics from pure inline asm)
  Add Y8, YV16, YV24 besides YV12, drop YUY2 support.
  Add 10-32 bit support for Y, YUV and planar RGB formats
  Add int "opt" parameter (mainly for debug: 0=C 1=SSE2 2=SSE4.1) for testing specific code paths
- Codes for different processor targets (SSSE3 and SSE4.1) are now separated and are compiled using function attributes (clang, gcc).
- Other source changes for errorless gcc and clang build
- LLVM support, see howto in RgTools.txt
  Note: use at least LLVM 9.0 build 21 June 2019 due to a clang compiler bug (_mm_avg_epu8 related, fixed on April 14 2019) older versions are up-to 1/3 slower than the Microsoft build.
  See latest snapshot builds at https://llvm.org/builds/
- GCC 8.3 support, CMakeFiles.txt, see howto in RgTools.txt
- RemoveGrain/Repair different code paths for SSE2/SSE4.1/AVX2 instead of SSE2/SSE3/AVX2.
- Add documentation (from old docs, new part: gcc/clang howto)
- todo: Overview 32 bit float parts for some RemoveGrain modes (6, 7, 8, 9, 15, 16, 23, 24)

v0.97 (20180702)
- Remove some inherited clipping to 0..1 range for 32bit float.
  In general we do not clamp in 32 bit float colorspaces, especially that U/V range is -0.5..+0.5 from Avisynth+ r2728

v0.96 (20170609)
- RemoveGrain: AVX2. Available when Avisynth+ reports AVX2 usability
  Can be disabled with new parameter: optAvx2=false
- Clense, ForwardClense, BackwardClense: ignore planar colorspace checking when planar=true. Like in RemoveGrain and Repair.
- Fix: Mode 11 and 13 for 32 bit float colorspaces (which worked like mode 10 and 12)

v0.95
- Fix: RemoveGrain Mode 20: overflow at 14 and 16 bit depths in SSE4 (stripes)
- Repair: error on unaligned frames (unaligned crop) instead of access violation error

v0.94
- Clense: new parameter (from v0.9): bool reduceflicker (default false)
- Clense: dummy compatibility parameters: bool planar, int cache
- autoregister filter MT modes as NICE_FILTER for Avisynth+
  except for Clense: when reduceflicker is true, MULTI_INSTANCE MT mode is reported
- alignment check in Repair and RemoveGrain (anti-unaligned crop measures)  

v0.93: new pixel formats are supported
- 10, 12, 14, 16 bit and float 
- Planar RGB, RGBA and YUVA (alpha plane is copied)

v0.92:
- tp7, high quality port of original source full with inline asm. C, SSE2 intrinsics. Modes 1-24

  [1]: http://opensource.org/licenses/MIT
  [2]: https://github.com/tp7/RgTools/wiki/RemoveGrain
  [3]: http://mechaweaponsvidya.wordpress.com/2014/01/31/enter-title-here/
  [4]: http://mechaweaponsvidya.wordpress.com/2014/04/23/ricing-your-temporal-medians-for-maximum-speed/
  [5]: http://mechaweaponsvidya.wordpress.com/2014/05/14/clense-versus-mt_clamp/
  [6]: https://github.com/pinterf/RgTools
  [7]: https://github.com/tp7/RgTools
