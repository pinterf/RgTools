// Clense, ForwardClense, BackwardClense core: C kernels, kernel selection

#include "clense_core.h"
#include "common.h"

RG_FORCEINLINE uint8_t clense_process_pixel_c(uint8_t src, uint8_t ref1, uint8_t ref2, int /*max_pixel_value*/) {
    return std::min(std::max(src, std::min(ref1, ref2)), std::max(ref1, ref2));
}

RG_FORCEINLINE uint16_t clense_process_pixel_c_16(uint16_t src, uint16_t ref1, uint16_t ref2, int /*max_pixel_value*/) {
  return std::min(std::max(src, std::min(ref1, ref2)), std::max(ref1, ref2));
}

RG_FORCEINLINE float clense_process_pixel_c_32(float src, float ref1, float ref2, int /*max_pixel_value*/) {
  return std::min(std::max(src, std::min(ref1, ref2)), std::max(ref1, ref2));
}

RG_FORCEINLINE uint8_t sclense_process_pixel_c(uint8_t src, uint8_t ref1, uint8_t ref2, int /*max_pixel_value*/) {
  uint8_t minref = std::min(ref1, ref2);
  uint8_t maxref = std::max(ref1, ref2);

  return clip(src, (uint8_t)std::max(minref*2 - ref2, 0), (uint8_t)std::min(maxref*2-ref2, 255));
}

RG_FORCEINLINE uint16_t sclense_process_pixel_c_16(uint16_t src, uint16_t ref1, uint16_t ref2, int max_pixel_value) {
  uint16_t minref = std::min(ref1, ref2);
  uint16_t maxref = std::max(ref1, ref2);

  uint16_t mi = subs_16_c(minref, subs_16_c(ref2, minref)); // minref - (ref2 - minref)
  uint16_t ma = adds_16_c(subs_16_c(maxref, ref2), maxref, max_pixel_value); // (maxref - ref2) + maxref

  return clip_16(src, mi, ma);
}

template<bool chroma>
RG_FORCEINLINE float sclense_process_pixel_c_32(float src, float ref1, float ref2, int /*max_pixel_value*/) {
  float minref = std::min(ref1, ref2);
  float maxref = std::max(ref1, ref2);

  float mi = subs_32_c<chroma>(minref, subs_32_c<chroma>(ref2, minref)); // minref - (ref2 - minref)
  float ma = adds_32_c<chroma>(subs_32_c<chroma>(maxref, ref2), maxref); // (maxref - ref2) + maxref

  return clip_32(src, mi, ma);
}

template<typename pixel_t, CModeProcessor<pixel_t>/*(clense_process_pixel_c)*/ processor>
static void process_plane_c(uint8_t* pDst, const uint8_t *pSrc, const uint8_t* pRef1, const uint8_t* pRef2, int dstPitch, int srcPitch, int ref1Pitch, int ref2Pitch, int rowsize, int height, int bits_per_pixel) {
  const int width = rowsize / sizeof(pixel_t);
  const int max_pixel_value = sizeof(pixel_t) == 2 ? (1 << bits_per_pixel) - 1 : 0; // used by the 10-16 bit pixel functions
  for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            reinterpret_cast<pixel_t *>(pDst)[x] = processor(reinterpret_cast<const pixel_t *>(pSrc)[x], reinterpret_cast<const pixel_t *>(pRef1)[x], reinterpret_cast<const pixel_t *>(pRef2)[x], max_pixel_value);
        }
        pDst += dstPitch;
        pSrc += srcPitch;
        pRef1 += ref1Pitch;
        pRef2 += ref2Pitch;
    }
}

const char* ClenseCore::init(const ClenseParams& params, const RgFormat& format, int cpu_flags) {
    (void)cpu_flags; // C-only builds
    if (params.grey && format.family == RgColorFamily::RGB)
      return "Clense: cannot speficy grey for planar RGB colorspaces";

    const ClenseMode mode = params.mode;
    format_ = format;
    grey_ = params.grey;


    if (format.pixelsize == 1) {
      processor_ = (mode == ClenseMode::BOTH)
        ? process_plane_c<uint8_t, clense_process_pixel_c>
        : process_plane_c<uint8_t, sclense_process_pixel_c>;
    }
    else if (format.pixelsize == 2) {
      processor_ = (mode == ClenseMode::BOTH)
        ? process_plane_c<uint16_t, clense_process_pixel_c_16>
        : process_plane_c<uint16_t, sclense_process_pixel_c_16>;
    }
    else { // format.pixelsize == 4
      processor_ = (mode == ClenseMode::BOTH)
        ? process_plane_c<float, clense_process_pixel_c_32>
        : process_plane_c<float, sclense_process_pixel_c_32<false>>;
      processor_chroma_ = (mode == ClenseMode::BOTH)
        ? process_plane_c<float, clense_process_pixel_c_32>
        : process_plane_c<float, sclense_process_pixel_c_32<true>>;
    }

    // only float has a separate chroma variant (chroma range -0.5..0.5)
    if (format.pixelsize != 4)
      processor_chroma_ = processor_;

#ifdef INTEL_INTRINSICS
    // the narrowest plane (subsampled chroma) must be wider than one vector
    int worst_case_width = format.width;
    if (format.family == RgColorFamily::YUV)
      worst_case_width >>= format.ssw;
    const bool sse2 = worst_case_width > 16 && (cpu_flags & RG_CPU_SSE2);
    const bool sse4 = worst_case_width > 16 && (cpu_flags & RG_CPU_SSE41);

    // 8 bit and float: SSE2, 10-16 bit: SSE4.1
    if (format.pixelsize == 2 ? sse4 : sse2) {
      processor_ = clense_sse_processor(mode, format.pixelsize, false);
      processor_chroma_ = format.pixelsize == 4 ? clense_sse_processor(mode, format.pixelsize, true) : processor_;
    }
#endif
    return nullptr;
}

void ClenseCore::process_plane(int plane_index, const RgPlane& plane, const uint8_t* ref1, int ref1_pitch, const uint8_t* ref2, int ref2_pitch) const {
    const bool chroma = format_.family == RgColorFamily::YUV && plane_index > 0;
    if (chroma && grey_)
      return; // grey: U and V are not processed and not copied (random content, for speed; as documented for RemoveGrain)

    ClenseProcessor* processor = chroma ? processor_chroma_ : processor_;
    processor(plane.dst, plane.src, ref1, ref2, plane.dst_pitch, plane.src_pitch, ref1_pitch, ref2_pitch, plane.rowsize, plane.height, format_.bits_per_pixel);
}
