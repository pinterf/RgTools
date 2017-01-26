#include "rg_functions_c.h"
#include "rg_functions_sse.h"
#include "removegrain.h"


template<typename pixel_t, SseModeProcessor processor, SseModeProcessor processor_a>
static void process_plane_sse(IScriptEnvironment* env, const BYTE* pSrc8, BYTE* pDst8, int rowsize, int height, int srcPitch, int dstPitch) {
    env->BitBlt(pDst8, dstPitch, pSrc8, srcPitch, rowsize, 1);

    pixel_t *pDst = reinterpret_cast<pixel_t *>(pDst8);
    const pixel_t *pSrc = reinterpret_cast<const pixel_t *>(pSrc8);

    dstPitch /= sizeof(pixel_t);
    const int srcPitchOrig = srcPitch;
    srcPitch /= sizeof(pixel_t);

    const int width = rowsize / sizeof(pixel_t);
    const int pixels_at_at_time = 16 / sizeof(pixel_t);

    pSrc += srcPitch;
    pDst += dstPitch;
    int mod_width = width / pixels_at_at_time * pixels_at_at_time;

    for (int y = 1; y < height - 1; ++y) {
      pDst[0] = pSrc[0];

      // unaligned first 16 bytes, last pixel overlaps with the next aligned loop
      __m128i result = processor((uint8_t *)(pSrc + 1), srcPitchOrig);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(pDst + 1), result);

      // aligned
      for (int x = pixels_at_at_time; x < mod_width - 1; x += pixels_at_at_time) {
        __m128i result = processor_a((uint8_t *)(pSrc + x), srcPitchOrig);
        _mm_store_si128(reinterpret_cast<__m128i*>(pDst + x), result);
      }
      
      if (mod_width != width) {
        __m128i result = processor((uint8_t *)(pSrc + width - 1 - pixels_at_at_time), srcPitchOrig);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(pDst + width - 1 - pixels_at_at_time), result);
      }
      
      pDst[width - 1] = pSrc[width - 1];

      pSrc += srcPitch;
      pDst += dstPitch;
    }

    env->BitBlt((uint8_t *)(pDst), dstPitch*sizeof(pixel_t), (uint8_t *)(pSrc), srcPitch*sizeof(pixel_t), rowsize, 1);
}


template<typename pixel_t, SseModeProcessor processor, SseModeProcessor processor_a>
static void process_halfplane_sse(IScriptEnvironment* env, const BYTE* pSrc8, BYTE* pDst8, int rowsize, int height, int srcPitch, int dstPitch) {
  pixel_t *pDst = reinterpret_cast<pixel_t *>(pDst8);
  const pixel_t *pSrc = reinterpret_cast<const pixel_t *>(pSrc8);

  dstPitch /= sizeof(pixel_t);
  const int srcPitchOrig = srcPitch;
  srcPitch /= sizeof(pixel_t);

  const int width = rowsize / sizeof(pixel_t);
  const int pixels_at_at_time = 16 / sizeof(pixel_t);

  pSrc += srcPitch;
    pDst += dstPitch;
    int mod_width = width / pixels_at_at_time * pixels_at_at_time;

    for (int y = 1; y < height/2; ++y) {
        pDst[0] = (pSrc[srcPitch] + pSrc[-srcPitch] + (sizeof(pixel_t) == 4 ? 0 : 1)) / 2; // float: no +1 rounding

        // unaligned first 16 bytes, last pixel overlaps with the next aligned loop
        __m128i result = processor((uint8_t *)(pSrc + 1), srcPitchOrig);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(pDst + 1), result);

        // aligned
        for (int x = pixels_at_at_time; x < mod_width - 1; x += pixels_at_at_time) {
          __m128i result = processor_a((uint8_t *)(pSrc + x), srcPitchOrig);
          _mm_store_si128(reinterpret_cast<__m128i*>(pDst + x), result);
        }

        if (mod_width != width) {
          __m128i result = processor((uint8_t *)(pSrc+width-1-pixels_at_at_time), srcPitchOrig);
          _mm_storeu_si128(reinterpret_cast<__m128i*>(pDst+width-1-pixels_at_at_time), result);
        }

        pDst[width-1] = (pSrc[width-1 + srcPitch] + pSrc[width-1 - srcPitch] + (sizeof(pixel_t) == 4 ? 0 : 1)) / 2; // float: no +1 rounding
        pSrc += srcPitch;
        pDst += dstPitch;

        env->BitBlt((uint8_t *)(pDst), dstPitch*sizeof(pixel_t), (uint8_t *)(pSrc), srcPitch*sizeof(pixel_t), rowsize, 1); //other field

        pSrc += srcPitch;
        pDst += dstPitch;
    }
}

template<typename pixel_t, SseModeProcessor processor, SseModeProcessor processor_a>
static void process_even_rows_sse(IScriptEnvironment* env, const BYTE* pSrc, BYTE* pDst, int rowsize, int height, int srcPitch, int dstPitch) {
    env->BitBlt(pDst, dstPitch, pSrc, srcPitch, rowsize, 2); //copy first two lines

    process_halfplane_sse<pixel_t, processor, processor_a>(env, pSrc+srcPitch, pDst+dstPitch, rowsize, height, srcPitch, dstPitch);
}

template<typename pixel_t, SseModeProcessor processor, SseModeProcessor processor_a>
static void process_odd_rows_sse(IScriptEnvironment* env, const BYTE* pSrc, BYTE* pDst, int rowsize, int height, int srcPitch, int dstPitch) {
    env->BitBlt(pDst, dstPitch, pSrc, srcPitch, rowsize, 1); //top border

    process_halfplane_sse<pixel_t, processor, processor_a>(env, pSrc, pDst, rowsize, height, srcPitch, dstPitch);

    env->BitBlt(pDst+dstPitch*(height-1), dstPitch, pSrc+srcPitch*(height-1), srcPitch, rowsize, 1); //bottom border
}

template<typename pixel_t, CModeProcessor<pixel_t> processor>
static void process_plane_c(IScriptEnvironment* env, const BYTE* pSrc8, BYTE* pDst8, int rowsize, int height, int srcPitch, int dstPitch) {
    env->BitBlt(pDst8, dstPitch, pSrc8, srcPitch, rowsize, 1);

    pixel_t *pDst = reinterpret_cast<pixel_t *>(pDst8);
    const pixel_t *pSrc = reinterpret_cast<const pixel_t *>(pSrc8);

    dstPitch /= sizeof(pixel_t);
    const int srcPitchOrig = srcPitch;
    srcPitch /= sizeof(pixel_t);

    const int width = rowsize / sizeof(pixel_t);

    pSrc += srcPitch;
    pDst += dstPitch;
    for (int y = 1; y < height-1; ++y) {
        pDst[0] = pSrc[0];
        for (int x = 1; x < width-1; x+=1) {
            pixel_t result = processor((uint8_t *)(pSrc + x), srcPitchOrig);
            pDst[x] = result;
        }
        pDst[width-1] = pSrc[width-1];

        pSrc += srcPitch;
        pDst += dstPitch;
    }

    env->BitBlt((uint8_t *)pDst, dstPitch*sizeof(pixel_t), (uint8_t *)pSrc, srcPitch*sizeof(pixel_t), rowsize, 1);
}

template<typename pixel_t, CModeProcessor<pixel_t> processor>
static void process_halfplane_c(IScriptEnvironment* env, const BYTE* pSrc8, BYTE* pDst8, int rowsize, int height, int srcPitch, int dstPitch) {
    pixel_t *pDst = reinterpret_cast<pixel_t *>(pDst8);
    const pixel_t *pSrc = reinterpret_cast<const pixel_t *>(pSrc8);

    dstPitch /= sizeof(pixel_t);
    const int srcPitchOrig = srcPitch;
    srcPitch /= sizeof(pixel_t);

    const int width = rowsize / sizeof(pixel_t);

    pSrc += srcPitch;
    pDst += dstPitch;
    for (int y = 1; y < height/2; ++y) {
        pDst[0] = (pSrc[srcPitch] + pSrc[-srcPitch] + (sizeof(pixel_t)==4 ? 0 : 1)) / 2; // float: no round
        for (int x = 1; x < width-1; x+=1) {
            pixel_t result = processor((uint8_t *)(pSrc + x), srcPitchOrig);
            pDst[x] = result;
        }
        pDst[width-1] = (pSrc[width-1 + srcPitch] + pSrc[width-1 - srcPitch] + (sizeof(pixel_t)==4 ? 0 : 1)) / 2; // float: no +1 rounding
        pSrc += srcPitch;
        pDst += dstPitch;

        env->BitBlt((uint8_t *)pDst, dstPitch*sizeof(pixel_t), (uint8_t *)pSrc, srcPitch*sizeof(pixel_t), rowsize, 1); //other field

        pSrc += srcPitch;
        pDst += dstPitch;
    }
}

template<typename pixel_t, CModeProcessor<pixel_t> processor>
static void process_even_rows_c(IScriptEnvironment* env, const BYTE* pSrc, BYTE* pDst, int rowsize, int height, int srcPitch, int dstPitch) {
    env->BitBlt(pDst, dstPitch, pSrc, srcPitch, rowsize, 2); //copy first two lines

    process_halfplane_c<pixel_t, processor>(env, pSrc+srcPitch, pDst+dstPitch, rowsize, height, srcPitch, dstPitch);
}

template<typename pixel_t, CModeProcessor<pixel_t> processor>
static void process_odd_rows_c(IScriptEnvironment* env, const BYTE* pSrc, BYTE* pDst, int rowsize, int height, int srcPitch, int dstPitch) {
    env->BitBlt(pDst, dstPitch, pSrc, srcPitch, rowsize, 1); //top border

    process_halfplane_c<pixel_t, processor>(env, pSrc, pDst, rowsize, height, srcPitch, dstPitch);

    env->BitBlt(pDst+dstPitch*(height-1), dstPitch, pSrc+srcPitch*(height-1), srcPitch, rowsize, 1); //bottom border
}

static void doNothing(IScriptEnvironment* env, const BYTE* pSrc, BYTE* pDst, int rowsize, int height, int srcPitch, int dstPitch) {

}

static void copyPlane(IScriptEnvironment* env, const BYTE* pSrc, BYTE* pDst, int rowsize, int height, int srcPitch, int dstPitch) {
    env->BitBlt(pDst, dstPitch, pSrc, srcPitch, rowsize, height);
}

/*
#ifdef	SHARPEN
static void	(*cleaning_methods[MAXMODE + 1])(BYTE *dp, int dpitch, const BYTE *sp, int spitch, int hblocks, int remainder, int incpitch, int height, int strength)
= { copy_plane, SSE_RemoveGrain1, SSE_RemoveGrain2, SSE_RemoveGrain3, SSE_RemoveGrain4, SSE_Repair15, SSE_Repair16, SSE_Repair17, SSE_Repair18a, diag9, copy_plane
, copy_plane, copy_plane, copy_plane, copy_plane, copy_plane, SmartRG, SSE_Repair18, copy_plane, copy_plane, copy_plane
, SmartAvgRGs, SmartAvgRGf
};
#else
static void	(*cleaning_methods[MAXMODE + 1])(BYTE *dp, int dpitch, const BYTE *sp, int spitch, int hblocks, int remainder, int incpitch, int height)
#ifdef	MODIFYPLUGIN (Repair)
= { do_nothing, SSE_RemoveGrain1, SSE_RemoveGrain2, SSE_RemoveGrain3, SSE_RemoveGrain4, diag5, diag6, diag7, diag8, diag9
, SSE_RemoveGrain10, SSE_RemoveGrain1, SSE_Repair12, SSE_Repair13, SSE_Repair14, SSE_Repair15, SSE_Repair16, SmartRG, SSE_Repair18
, New_Repair1, New_Repair2, New_Repair3, New_Repair1b, New_Repair2b, New_Repair3b};
#elif	defined(BLUR)
= { copy_plane, SSE_RemoveGrain1, SSE_RemoveGrain2, SSE_RemoveGrain3, SSE_RemoveGrain4, copy_plane, copy_plane, copy_plane, copy_plane, diag9, copy_plane
, copy_plane, copy_plane, copy_plane, copy_plane, copy_plane, copy_plane, copy_plane, copy_plane, copy_plane, copy_plane
, SmartAvgRGs, SmartAvgRGf
};
#else (RemoveGrain)
= { copy_plane, SSE_RemoveGrain1, SSE_RemoveGrain2, SSE_RemoveGrain3, SSE_RemoveGrain4, diag5, diag6, diag7, diag8, diag9
, SSE_RemoveGrain10, SSE_RemoveGrain11, SSE_RemoveGrain12, bob_top, bob_bottom, smartbob_top, smartbob_bottom, SmartRG, SSE_Repair18, SSE_RemoveGrain19, SSE_RemoveGrain20
, SmartAvgRGs, SmartAvgRGf, SSE_RemoveGrain23, SSE_RemoveGrain24, nondestructivesharpen, SmartRGC, SmartRGCL, SmartRGCL2, SmartRG18, SoftRG18};
#endif

PF extract:
seem to be new 
25 - nondestructivesharpen
26 - SmartRGC
27 - SmartRGCL
28 - SmartRGCL2
29 - SmartRG18
30 - SoftRG18
#endif	// SHARPEN

from 
http://web.archive.org/web/20130615165406/http://doom10.org/index.php?topic=2185.0

removegrain modes:

-1= bypass (output=0) faster than mode=0 (copy)

0 = copy

1 = medianblur. Same as Undot, but faster. (single dots)
2 = medianblur. Round up to the second closest minimum luma value in a 3x3 window matrix, if this second lowest value is lower than X pixel value, then leave unchanged. (1x2 spots)
3 = medianblur. Sames as mode 2 but rounded up to third  minimum value (but artifact risky). (3pixel-clusters)
4 = medianblur. Sames as mode 2 but rounded up to fourth minimum value (but artifact risky). (up to 2x2-pixel-clusters)

5 = medianblur. Edge sensitive. Only line pairs are used. Strong  edge protection.
6 = medianblur. Edge sensitive. Only line pairs are used. Fairly  edge protection.
7 = medianblur. Edge sensitive. Only line pairs are used. Mild    edge protection.
8 = medianblur. Edge sensitive. Only line pairs are used. Faint   edge protection.
9 = medianblur. Edge sensitive. Only line pairs are used. Barely  edge protection. Practically a spatial variant of trbarry's ST Median filter.

10 = Minimal sharpening. Replaces center pixel by its nearest neighbour. "Very poor denoise sharpener"

11 = Blur. 3x3 kernel convolution blur. Better than its counterpart internal Blur(1) (and faster)
12 = Blur. Same as 11 but fastest and only <= 1% less precise (still better than Blur(1))

13 = Smart bob (for interlaced content). Interpolates the top field.    Similar to Trbarry's weird bob (Tomsmocomp).
14 = Smart bob (for interlaced content). Interpolates the bottom field. Similar to Trbarry's weird bob (Tomsmocomp).
15 = Smart bob (for interlaced content). Same as mode 13 but more quality and slightly slower.
16 = Smart bob (for interlaced content). Same as mode 14 but more quality and slightly slower.

17 = medianblur. Same as mode 4 but better edge protection (similar to near artifact free mode 2). Probably best mode of all.
18 = medianblur. Same as mode 9 but better edge protection (Same as what mode 17 was to mode 4, but in this case to mode 9, and far less denoising than mode 17)

19 = Blur.         Average of its 8 neighbours.
20 = Blur. Uniform average of its 8 neighbours. Better than 19 but slower. Very similar to blur(1.58) but faster.

21 = medianblur. Clipping is done with respect to averages of neighbours. Best for cartoons.
22 = medianblur. Same as mode 21 but much faster (fastest mode of all)

23 = Dehalo. Fixes small (as one pixel wide) haloes.
24 = Dehalo. Same as 23 but considerably more conservative and slightly slower. Preferred.

25 = Minimal sharpening.

26 = medianblur. Based off mode 17, but preserves corners, but not thin lines.
27 = medianblur. Same as mode 26 but preserves thin lines.


Recommended modes: 12,20(Gaussian Blur) 17,22 (Median Blur)

Ranking:
Denoising/Compression: 4,17,9,8,3,7,6,2,5,1
Artifact free:         1,5,2,18,6,7,8,17,3,4,9

"As far as compression (Denoising) is concerned, my benchmarks so far give the following mode ranking: 4,17,9,8,3,7,6,2,5,1, but modes 4 and 17 really stand out.
As far as artifacts are concerned, we have unfortunately almost the reverse mode ranking: 1,5,2,18,6,7,8,17,3,4,9. Modes= 1,5,2 are the risk free modes,
the modes 18,6,7,8,17 show low to moderate artifact risk (usually some softness) and modes 3,4,9 have severe problems with thin lines. Mode 9 has
less artifacts than mode 4, but when they occur  they look a lot more ugly. Mode 17 is my clear personal favorite.
As far as compression is concerned it is fairly close to the leader, mode 4, and I have only seen some softness but hardly any visable artifacts."


mode 	sharp edges 	corners   thin lines  line ends  thin curves  compression
1   	10 	          10 	      10 	        10 	        10 	        1
2   	10 	          10 	      10 	        4 	        10 	        2
3   	10 	          9 	      3 	        1 	        3 	        5
4   	10 	          3 	      1           0 	        1 	        8
5   	10 	          10 	      10 	        9 	        9 	        2
6   	10 	          6 	      9 	        4         	3 	        3
7   	10   	        6 	      9 	        4 	        3 	        3
8   	10 	          5     	  8 	        3 	        3 	        4
9   	10 	          3     	  4 	        1        	  2 	        5
10 	  10 	          10    	  10          10   	      10 	        1
11   	1 	          1     	  2 	        1 	        2 	        9
12 	  1 	          1     	  2 	        1 	        2 	        9
17   	10 	          2     	  8 	        2 	        4 	        7
18   	10 	          6     	  9 	        6 	        5 	        2
19   	1 	          1     	  1 	        0 	        1 	        10
20   	1 	          1     	  1 	        0 	        1 	        8
21   	6 	          2     	  8 	        4 	        4 	        5
22   	6 	          2     	  8 	        4 	        4 	        5
23   	6 	          5     	  6 	        4 	        6 	        4
24   	7 	          6     	  7 	        5 	        7 	        3
25   	10 	          10    	  10 	        10 	        10 	        -1
*/

PlaneProcessor* sse2_functions[] = {
    doNothing,
    copyPlane,
    process_plane_sse<uint8_t, rg_mode1_sse<false, SSE2>, rg_mode1_sse<true, SSE2>>,
    process_plane_sse<uint8_t, rg_mode2_sse<false, SSE2>, rg_mode2_sse<true, SSE2>>,
    process_plane_sse<uint8_t, rg_mode3_sse<false, SSE2>, rg_mode3_sse<true, SSE2>>,
    process_plane_sse<uint8_t, rg_mode4_sse<false, SSE2>, rg_mode4_sse<true, SSE2>>,
    process_plane_sse<uint8_t, rg_mode5_sse<false, SSE2>, rg_mode5_sse<true, SSE2>>,
    process_plane_sse<uint8_t, rg_mode6_sse<false, SSE2>, rg_mode6_sse<true, SSE2>>,
    process_plane_sse<uint8_t, rg_mode7_sse<false, SSE2>, rg_mode7_sse<true, SSE2>>,
    process_plane_sse<uint8_t, rg_mode8_sse<false, SSE2>, rg_mode8_sse<true, SSE2>>,
    process_plane_sse<uint8_t, rg_mode9_sse<false, SSE2>, rg_mode9_sse<true, SSE2>>,
    process_plane_sse<uint8_t, rg_mode10_sse<false, SSE2>, rg_mode10_sse<true, SSE2>>,
    process_plane_sse<uint8_t, rg_mode11_sse<false, SSE2>, rg_mode11_sse<true, SSE2>>,
    process_plane_sse<uint8_t, rg_mode12_sse<false, SSE2>, rg_mode12_sse<true, SSE2>>,
    process_even_rows_sse<uint8_t, rg_mode13_and14_sse<false, SSE2>, rg_mode13_and14_sse<true, SSE2>>,
    process_odd_rows_sse<uint8_t, rg_mode13_and14_sse<false, SSE2>, rg_mode13_and14_sse<true, SSE2>>,
    process_even_rows_sse<uint8_t, rg_mode15_and16_sse<false, SSE2>, rg_mode15_and16_sse<true, SSE2>>,
    process_odd_rows_sse<uint8_t, rg_mode15_and16_sse<false, SSE2>, rg_mode15_and16_sse<true, SSE2>>,
    process_plane_sse<uint8_t, rg_mode17_sse<false, SSE2>, rg_mode17_sse<true, SSE2>>,
    process_plane_sse<uint8_t, rg_mode18_sse<false, SSE2>, rg_mode18_sse<true, SSE2>>,
    process_plane_sse<uint8_t, rg_mode19_sse<false, SSE2>, rg_mode19_sse<true, SSE2>>,
    process_plane_sse<uint8_t, rg_mode20_sse<false, SSE2>, rg_mode20_sse<true, SSE2>>,
    process_plane_sse<uint8_t, rg_mode21_sse<false, SSE2>, rg_mode21_sse<true, SSE2>>,
    process_plane_sse<uint8_t, rg_mode22_sse<false, SSE2>, rg_mode22_sse<true, SSE2>>,
    process_plane_sse<uint8_t, rg_mode23_sse<false, SSE2>, rg_mode23_sse<true, SSE2>>,
    process_plane_sse<uint8_t, rg_mode24_sse<false, SSE2>, rg_mode24_sse<true, SSE2>>,
};

PlaneProcessor* sse3_functions[] = {
    doNothing,
    copyPlane,
    process_plane_sse<uint8_t, rg_mode1_sse<false, SSE3>, rg_mode1_sse<true, SSE3>>,
    process_plane_sse<uint8_t, rg_mode2_sse<false, SSE3>, rg_mode2_sse<true, SSE3>>,
    process_plane_sse<uint8_t, rg_mode3_sse<false, SSE3>, rg_mode3_sse<true, SSE3>>,
    process_plane_sse<uint8_t, rg_mode4_sse<false, SSE3>, rg_mode4_sse<true, SSE3>>,
    process_plane_sse<uint8_t, rg_mode5_sse<false, SSE3>, rg_mode5_sse<true, SSE3>>,
    process_plane_sse<uint8_t, rg_mode6_sse<false, SSE3>, rg_mode6_sse<true, SSE3>>,
    process_plane_sse<uint8_t, rg_mode7_sse<false, SSE3>, rg_mode7_sse<true, SSE3>>,
    process_plane_sse<uint8_t, rg_mode8_sse<false, SSE3>, rg_mode8_sse<true, SSE3>>,
    process_plane_sse<uint8_t, rg_mode9_sse<false, SSE3>, rg_mode9_sse<true, SSE3>>,
    process_plane_sse<uint8_t, rg_mode10_sse<false, SSE3>, rg_mode10_sse<true, SSE3>>,
    process_plane_sse<uint8_t, rg_mode11_sse<false, SSE3>, rg_mode11_sse<true, SSE3>>,
    process_plane_sse<uint8_t, rg_mode12_sse<false, SSE3>, rg_mode12_sse<true, SSE3>>,
    process_even_rows_sse<uint8_t, rg_mode13_and14_sse<false, SSE3>, rg_mode13_and14_sse<true, SSE3>>,
    process_odd_rows_sse<uint8_t, rg_mode13_and14_sse<false, SSE3>, rg_mode13_and14_sse<true, SSE3>>,
    process_even_rows_sse<uint8_t, rg_mode15_and16_sse<false, SSE3>, rg_mode15_and16_sse<true, SSE3>>,
    process_odd_rows_sse<uint8_t, rg_mode15_and16_sse<false, SSE3>, rg_mode15_and16_sse<true, SSE3>>,
    process_plane_sse<uint8_t, rg_mode17_sse<false, SSE3>, rg_mode17_sse<true, SSE3>>,
    process_plane_sse<uint8_t, rg_mode18_sse<false, SSE3>, rg_mode18_sse<true, SSE3>>,
    process_plane_sse<uint8_t, rg_mode19_sse<false, SSE3>, rg_mode19_sse<true, SSE3>>,
    process_plane_sse<uint8_t, rg_mode20_sse<false, SSE3>, rg_mode20_sse<true, SSE3>>,
    process_plane_sse<uint8_t, rg_mode21_sse<false, SSE3>, rg_mode21_sse<true, SSE3>>,
    process_plane_sse<uint8_t, rg_mode22_sse<false, SSE3>, rg_mode22_sse<true, SSE3>>,
    process_plane_sse<uint8_t, rg_mode23_sse<false, SSE3>, rg_mode23_sse<true, SSE3>>,
    process_plane_sse<uint8_t, rg_mode24_sse<false, SSE3>, rg_mode24_sse<true, SSE3>>,
};

PlaneProcessor* sse4_functions_16_10[] = {
  doNothing,
  copyPlane,
  process_plane_sse<uint16_t, rg_mode1_sse_16<false>, rg_mode1_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode2_sse_16<false>, rg_mode2_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode3_sse_16<false>, rg_mode3_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode4_sse_16<false>, rg_mode4_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode5_sse_16<false>, rg_mode5_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode6_sse_16<10, false>, rg_mode6_sse_16<10, false>>,
  process_plane_sse<uint16_t, rg_mode7_sse_16<false>, rg_mode7_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode8_sse_16<10, false>, rg_mode8_sse_16<10, true>>,
  process_plane_sse<uint16_t, rg_mode9_sse_16<false>, rg_mode9_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode10_sse_16<false>, rg_mode10_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode11_sse_16<false>, rg_mode11_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode12_sse_16<false>, rg_mode12_sse_16<true>>,
  process_even_rows_sse<uint16_t, rg_mode13_and14_sse_16<false>, rg_mode13_and14_sse_16<true>>,
  process_odd_rows_sse<uint16_t, rg_mode13_and14_sse_16<false>, rg_mode13_and14_sse_16<true>>,
  process_even_rows_sse<uint16_t, rg_mode15_and16_sse_16<false>, rg_mode15_and16_sse_16<true>>,
  process_odd_rows_sse<uint16_t, rg_mode15_and16_sse_16<false>, rg_mode15_and16_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode17_sse_16<false>, rg_mode17_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode18_sse_16<false>, rg_mode18_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode19_sse_16<false>, rg_mode19_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode20_sse_16<false>, rg_mode20_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode21_sse_16<false>, rg_mode21_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode22_sse_16<false>, rg_mode22_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode23_sse_16<false>, rg_mode23_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode24_sse_16<false>, rg_mode24_sse_16<true>>,
};

PlaneProcessor* sse4_functions_16_12[] = {
  doNothing,
  copyPlane,
  process_plane_sse<uint16_t, rg_mode1_sse_16<false>, rg_mode1_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode2_sse_16<false>, rg_mode2_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode3_sse_16<false>, rg_mode3_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode4_sse_16<false>, rg_mode4_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode5_sse_16<false>, rg_mode5_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode6_sse_16<12, false>, rg_mode6_sse_16<12, false>>,
  process_plane_sse<uint16_t, rg_mode7_sse_16<false>, rg_mode7_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode8_sse_16<12, false>, rg_mode8_sse_16<12, true>>,
  process_plane_sse<uint16_t, rg_mode9_sse_16<false>, rg_mode9_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode10_sse_16<false>, rg_mode10_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode11_sse_16<false>, rg_mode11_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode12_sse_16<false>, rg_mode12_sse_16<true>>,
  process_even_rows_sse<uint16_t, rg_mode13_and14_sse_16<false>, rg_mode13_and14_sse_16<true>>,
  process_odd_rows_sse<uint16_t, rg_mode13_and14_sse_16<false>, rg_mode13_and14_sse_16<true>>,
  process_even_rows_sse<uint16_t, rg_mode15_and16_sse_16<false>, rg_mode15_and16_sse_16<true>>,
  process_odd_rows_sse<uint16_t, rg_mode15_and16_sse_16<false>, rg_mode15_and16_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode17_sse_16<false>, rg_mode17_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode18_sse_16<false>, rg_mode18_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode19_sse_16<false>, rg_mode19_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode20_sse_16<false>, rg_mode20_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode21_sse_16<false>, rg_mode21_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode22_sse_16<false>, rg_mode22_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode23_sse_16<false>, rg_mode23_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode24_sse_16<false>, rg_mode24_sse_16<true>>,
};

PlaneProcessor* sse4_functions_16_14[] = {
  doNothing,
  copyPlane,
  process_plane_sse<uint16_t, rg_mode1_sse_16<false>, rg_mode1_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode2_sse_16<false>, rg_mode2_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode3_sse_16<false>, rg_mode3_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode4_sse_16<false>, rg_mode4_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode5_sse_16<false>, rg_mode5_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode6_sse_16<14, false>, rg_mode6_sse_16<14, true>>,
  process_plane_sse<uint16_t, rg_mode7_sse_16<false>, rg_mode7_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode8_sse_16<14, false>, rg_mode8_sse_16<14, true>>,
  process_plane_sse<uint16_t, rg_mode9_sse_16<false>, rg_mode9_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode10_sse_16<false>, rg_mode10_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode11_sse_16<false>, rg_mode11_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode12_sse_16<false>, rg_mode12_sse_16<true>>,
  process_even_rows_sse<uint16_t, rg_mode13_and14_sse_16<false>, rg_mode13_and14_sse_16<true>>,
  process_odd_rows_sse<uint16_t, rg_mode13_and14_sse_16<false>, rg_mode13_and14_sse_16<true>>,
  process_even_rows_sse<uint16_t, rg_mode15_and16_sse_16<false>, rg_mode15_and16_sse_16<true>>,
  process_odd_rows_sse<uint16_t, rg_mode15_and16_sse_16<false>, rg_mode15_and16_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode17_sse_16<false>, rg_mode17_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode18_sse_16<false>, rg_mode18_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode19_sse_16<false>, rg_mode19_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode20_sse_16<false>, rg_mode20_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode21_sse_16<false>, rg_mode21_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode22_sse_16<false>, rg_mode22_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode23_sse_16<false>, rg_mode23_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode24_sse_16<false>, rg_mode24_sse_16<true>>,
};

PlaneProcessor* sse4_functions_16_16[] = {
  doNothing,
  copyPlane,
  process_plane_sse<uint16_t, rg_mode1_sse_16<false>, rg_mode1_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode2_sse_16<false>, rg_mode2_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode3_sse_16<false>, rg_mode3_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode4_sse_16<false>, rg_mode4_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode5_sse_16<false>, rg_mode5_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode6_sse_16<16, false>, rg_mode6_sse_16<16, true>>,
  process_plane_sse<uint16_t, rg_mode7_sse_16<false>, rg_mode7_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode8_sse_16<16, false>, rg_mode8_sse_16<16, true>>,
  process_plane_sse<uint16_t, rg_mode9_sse_16<false>, rg_mode9_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode10_sse_16<false>, rg_mode10_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode11_sse_16<false>, rg_mode11_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode12_sse_16<false>, rg_mode12_sse_16<true>>,
  process_even_rows_sse<uint16_t, rg_mode13_and14_sse_16<false>, rg_mode13_and14_sse_16<true>>,
  process_odd_rows_sse<uint16_t, rg_mode13_and14_sse_16<false>, rg_mode13_and14_sse_16<true>>,
  process_even_rows_sse<uint16_t, rg_mode15_and16_sse_16<false>, rg_mode15_and16_sse_16<true>>,
  process_odd_rows_sse<uint16_t, rg_mode15_and16_sse_16<false>, rg_mode15_and16_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode17_sse_16<false>, rg_mode17_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode18_sse_16<false>, rg_mode18_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode19_sse_16<false>, rg_mode19_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode20_sse_16<false>, rg_mode20_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode21_sse_16<false>, rg_mode21_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode22_sse_16<false>, rg_mode22_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode23_sse_16<false>, rg_mode23_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode24_sse_16<false>, rg_mode24_sse_16<true>>,
};



PlaneProcessor* sse4_functions_32[] = {
  doNothing,
  copyPlane,
  process_plane_sse<float, rg_mode1_sse_32<false>, rg_mode1_sse_32<true>>,
  process_plane_sse<float, rg_mode2_sse_32<false>, rg_mode2_sse_32<true>>,
  process_plane_sse<float, rg_mode3_sse_32<false>, rg_mode3_sse_32<true>>,
  process_plane_sse<float, rg_mode4_sse_32<false>, rg_mode4_sse_32<true>>,
  process_plane_sse<float, rg_mode5_sse_32<false>, rg_mode5_sse_32<true>>,
  process_plane_sse<float, rg_mode6_sse_32<false>, rg_mode6_sse_32<true>>,
  process_plane_sse<float, rg_mode7_sse_32<false>, rg_mode7_sse_32<true>>,
  process_plane_sse<float, rg_mode8_sse_32<false>, rg_mode8_sse_32<true>>,
  process_plane_sse<float, rg_mode9_sse_32<false>, rg_mode9_sse_32<true>>,
  process_plane_sse<float, rg_mode10_sse_32<false>, rg_mode10_sse_32<true>>,
  process_plane_sse<float, rg_mode11_sse_32<false>, rg_mode10_sse_32<true>>,
  process_plane_sse<float, rg_mode12_sse_32<false>, rg_mode12_sse_32<true>>,
  process_even_rows_sse<float, rg_mode13_and14_sse_32<false>, rg_mode12_sse_32<true>>,
  process_odd_rows_sse<float, rg_mode13_and14_sse_32<false>, rg_mode13_and14_sse_32<true>>,
  process_even_rows_sse<float, rg_mode15_and16_sse_32<false>, rg_mode15_and16_sse_32<true>>,
  process_odd_rows_sse<float, rg_mode15_and16_sse_32<false>, rg_mode15_and16_sse_32<true>>,
  process_plane_sse<float, rg_mode17_sse_32<false>, rg_mode17_sse_32<true>>,
  process_plane_sse<float, rg_mode18_sse_32<false>, rg_mode18_sse_32<true>>,
  process_plane_sse<float, rg_mode19_sse_32<false>, rg_mode19_sse_32<true>>,
  process_plane_sse<float, rg_mode20_sse_32<false>, rg_mode20_sse_32<true>>,
  process_plane_sse<float, rg_mode21_sse_32<false>, rg_mode21_sse_32<true>>,
  process_plane_sse<float, rg_mode22_sse_32<false>, rg_mode22_sse_32<true>>,
  process_plane_sse<float, rg_mode23_sse_32<false>, rg_mode23_sse_32<true>>,
  process_plane_sse<float, rg_mode24_sse_32<false>, rg_mode24_sse_32<true>>,
};


PlaneProcessor* c_functions[] = {
    doNothing,
    copyPlane,
    process_plane_c<uint8_t, rg_mode1_cpp>,
    process_plane_c<uint8_t, rg_mode2_cpp>,
    process_plane_c<uint8_t, rg_mode3_cpp>,
    process_plane_c<uint8_t, rg_mode4_cpp>,
    process_plane_c<uint8_t, rg_mode5_cpp>,
    process_plane_c<uint8_t, rg_mode6_cpp>,
    process_plane_c<uint8_t, rg_mode7_cpp>,
    process_plane_c<uint8_t, rg_mode8_cpp>,
    process_plane_c<uint8_t, rg_mode9_cpp>,
    process_plane_c<uint8_t, rg_mode10_cpp>,
    process_plane_c<uint8_t, rg_mode11_cpp>,
    process_plane_c<uint8_t, rg_mode12_cpp>,
    process_even_rows_c<uint8_t, rg_mode13_and14_cpp>,
    process_odd_rows_c<uint8_t, rg_mode13_and14_cpp>,
    process_even_rows_c<uint8_t, rg_mode15_and16_cpp>,
    process_odd_rows_c<uint8_t, rg_mode15_and16_cpp>,
    process_plane_c<uint8_t, rg_mode17_cpp>,
    process_plane_c<uint8_t, rg_mode18_cpp>,
    process_plane_c<uint8_t, rg_mode19_cpp>,
    process_plane_c<uint8_t, rg_mode20_cpp>,
    process_plane_c<uint8_t, rg_mode21_cpp>,
    process_plane_c<uint8_t, rg_mode22_cpp>,
    process_plane_c<uint8_t, rg_mode23_cpp>,
    process_plane_c<uint8_t, rg_mode24_cpp>
};

PlaneProcessor* c_functions_10[] = {
  doNothing,
  copyPlane,
  process_plane_c<uint16_t, rg_mode1_cpp_16>,
  process_plane_c<uint16_t, rg_mode2_cpp_16>,
  process_plane_c<uint16_t, rg_mode3_cpp_16>,
  process_plane_c<uint16_t, rg_mode4_cpp_16>,
  process_plane_c<uint16_t, rg_mode5_cpp_16>,
  process_plane_c<uint16_t, rg_mode6_cpp_16<10>>,
  process_plane_c<uint16_t, rg_mode7_cpp_16>,
  process_plane_c<uint16_t, rg_mode8_cpp_16<10>>,
  process_plane_c<uint16_t, rg_mode9_cpp_16>,
  process_plane_c<uint16_t, rg_mode10_cpp_16>,
  process_plane_c<uint16_t, rg_mode11_cpp_16>,
  process_plane_c<uint16_t, rg_mode12_cpp_16>,
  process_even_rows_c<uint16_t, rg_mode13_and14_cpp_16>,
  process_odd_rows_c<uint16_t, rg_mode13_and14_cpp_16>,
  process_even_rows_c<uint16_t, rg_mode15_and16_cpp_16>,
  process_odd_rows_c<uint16_t, rg_mode15_and16_cpp_16>,
  process_plane_c<uint16_t, rg_mode17_cpp_16>,
  process_plane_c<uint16_t, rg_mode18_cpp_16>,
  process_plane_c<uint16_t, rg_mode19_cpp_16>,
  process_plane_c<uint16_t, rg_mode20_cpp_16>,
  process_plane_c<uint16_t, rg_mode21_cpp_16>,
  process_plane_c<uint16_t, rg_mode22_cpp_16>,
  process_plane_c<uint16_t, rg_mode23_cpp_16>,
  process_plane_c<uint16_t, rg_mode24_cpp_16>
};

PlaneProcessor* c_functions_12[] = {
  doNothing,
  copyPlane,
  process_plane_c<uint16_t, rg_mode1_cpp_16>,
  process_plane_c<uint16_t, rg_mode2_cpp_16>,
  process_plane_c<uint16_t, rg_mode3_cpp_16>,
  process_plane_c<uint16_t, rg_mode4_cpp_16>,
  process_plane_c<uint16_t, rg_mode5_cpp_16>,
  process_plane_c<uint16_t, rg_mode6_cpp_16<12>>,
  process_plane_c<uint16_t, rg_mode7_cpp_16>,
  process_plane_c<uint16_t, rg_mode8_cpp_16<12>>,
  process_plane_c<uint16_t, rg_mode9_cpp_16>,
  process_plane_c<uint16_t, rg_mode10_cpp_16>,
  process_plane_c<uint16_t, rg_mode11_cpp_16>,
  process_plane_c<uint16_t, rg_mode12_cpp_16>,
  process_even_rows_c<uint16_t, rg_mode13_and14_cpp_16>,
  process_odd_rows_c<uint16_t, rg_mode13_and14_cpp_16>,
  process_even_rows_c<uint16_t, rg_mode15_and16_cpp_16>,
  process_odd_rows_c<uint16_t, rg_mode15_and16_cpp_16>,
  process_plane_c<uint16_t, rg_mode17_cpp_16>,
  process_plane_c<uint16_t, rg_mode18_cpp_16>,
  process_plane_c<uint16_t, rg_mode19_cpp_16>,
  process_plane_c<uint16_t, rg_mode20_cpp_16>,
  process_plane_c<uint16_t, rg_mode21_cpp_16>,
  process_plane_c<uint16_t, rg_mode22_cpp_16>,
  process_plane_c<uint16_t, rg_mode23_cpp_16>,
  process_plane_c<uint16_t, rg_mode24_cpp_16>
};

PlaneProcessor* c_functions_14[] = {
  doNothing,
  copyPlane,
  process_plane_c<uint16_t, rg_mode1_cpp_16>,
  process_plane_c<uint16_t, rg_mode2_cpp_16>,
  process_plane_c<uint16_t, rg_mode3_cpp_16>,
  process_plane_c<uint16_t, rg_mode4_cpp_16>,
  process_plane_c<uint16_t, rg_mode5_cpp_16>,
  process_plane_c<uint16_t, rg_mode6_cpp_16<14>>,
  process_plane_c<uint16_t, rg_mode7_cpp_16>,
  process_plane_c<uint16_t, rg_mode8_cpp_16<14>>,
  process_plane_c<uint16_t, rg_mode9_cpp_16>,
  process_plane_c<uint16_t, rg_mode10_cpp_16>,
  process_plane_c<uint16_t, rg_mode11_cpp_16>,
  process_plane_c<uint16_t, rg_mode12_cpp_16>,
  process_even_rows_c<uint16_t, rg_mode13_and14_cpp_16>,
  process_odd_rows_c<uint16_t, rg_mode13_and14_cpp_16>,
  process_even_rows_c<uint16_t, rg_mode15_and16_cpp_16>,
  process_odd_rows_c<uint16_t, rg_mode15_and16_cpp_16>,
  process_plane_c<uint16_t, rg_mode17_cpp_16>,
  process_plane_c<uint16_t, rg_mode18_cpp_16>,
  process_plane_c<uint16_t, rg_mode19_cpp_16>,
  process_plane_c<uint16_t, rg_mode20_cpp_16>,
  process_plane_c<uint16_t, rg_mode21_cpp_16>,
  process_plane_c<uint16_t, rg_mode22_cpp_16>,
  process_plane_c<uint16_t, rg_mode23_cpp_16>,
  process_plane_c<uint16_t, rg_mode24_cpp_16>
};


PlaneProcessor* c_functions_16[] = {
  doNothing,
  copyPlane,
  process_plane_c<uint16_t, rg_mode1_cpp_16>,
  process_plane_c<uint16_t, rg_mode2_cpp_16>,
  process_plane_c<uint16_t, rg_mode3_cpp_16>,
  process_plane_c<uint16_t, rg_mode4_cpp_16>,
  process_plane_c<uint16_t, rg_mode5_cpp_16>,
  process_plane_c<uint16_t, rg_mode6_cpp_16<16>>,
  process_plane_c<uint16_t, rg_mode7_cpp_16>,
  process_plane_c<uint16_t, rg_mode8_cpp_16<16>>,
  process_plane_c<uint16_t, rg_mode9_cpp_16>,
  process_plane_c<uint16_t, rg_mode10_cpp_16>,
  process_plane_c<uint16_t, rg_mode11_cpp_16>,
  process_plane_c<uint16_t, rg_mode12_cpp_16>,
  process_even_rows_c<uint16_t, rg_mode13_and14_cpp_16>,
  process_odd_rows_c<uint16_t, rg_mode13_and14_cpp_16>,
  process_even_rows_c<uint16_t, rg_mode15_and16_cpp_16>,
  process_odd_rows_c<uint16_t, rg_mode15_and16_cpp_16>,
  process_plane_c<uint16_t, rg_mode17_cpp_16>,
  process_plane_c<uint16_t, rg_mode18_cpp_16>,
  process_plane_c<uint16_t, rg_mode19_cpp_16>,
  process_plane_c<uint16_t, rg_mode20_cpp_16>,
  process_plane_c<uint16_t, rg_mode21_cpp_16>,
  process_plane_c<uint16_t, rg_mode22_cpp_16>,
  process_plane_c<uint16_t, rg_mode23_cpp_16>,
  process_plane_c<uint16_t, rg_mode24_cpp_16>
};

PlaneProcessor* c_functions_32[] = {
  doNothing,
  copyPlane,
  process_plane_c<float, rg_mode1_cpp_32>,
  process_plane_c<float, rg_mode2_cpp_32>,
  process_plane_c<float, rg_mode3_cpp_32>,
  process_plane_c<float, rg_mode4_cpp_32>,
  process_plane_c<float, rg_mode5_cpp_32>,
  process_plane_c<float, rg_mode6_cpp_32>,
  process_plane_c<float, rg_mode7_cpp_32>,
  process_plane_c<float, rg_mode8_cpp_32>,
  process_plane_c<float, rg_mode9_cpp_32>,
  process_plane_c<float, rg_mode10_cpp_32>,
  process_plane_c<float, rg_mode11_cpp_32>,
  process_plane_c<float, rg_mode12_cpp_32>,
  process_even_rows_c<float, rg_mode13_and14_cpp_32>,
  process_odd_rows_c<float, rg_mode13_and14_cpp_32>,
  process_even_rows_c<float, rg_mode15_and16_cpp_32>,
  process_odd_rows_c<float, rg_mode15_and16_cpp_32>,
  process_plane_c<float, rg_mode17_cpp_32>,
  process_plane_c<float, rg_mode18_cpp_32>,
  process_plane_c<float, rg_mode19_cpp_32>,
  process_plane_c<float, rg_mode20_cpp_32>,
  process_plane_c<float, rg_mode21_cpp_32>,
  process_plane_c<float, rg_mode22_cpp_32>,
  process_plane_c<float, rg_mode23_cpp_32>,
  process_plane_c<float, rg_mode24_cpp_32>
};


RemoveGrain::RemoveGrain(PClip child, int mode, int modeU, int modeV, bool skip_cs_check, IScriptEnvironment* env)
    : GenericVideoFilter(child), mode_(mode), modeU_(modeU), modeV_(modeV), functions(nullptr) {
    if (!(vi.IsPlanar() || skip_cs_check)) {
        env->ThrowError("RemoveGrain works only with planar colorspaces");
    }

    if (mode <= UNDEFINED_MODE || mode_ > 24 || modeU_ > 24 || modeV_ > 24) {
        env->ThrowError("RemoveGrain mode should be between -1 and 24!");
    }

    bool isPlanarRGB = vi.IsPlanarRGB() || vi.IsPlanarRGBA();
    if (isPlanarRGB && ((modeU_ > UNDEFINED_MODE) || (modeV_ > UNDEFINED_MODE))) {
      env->ThrowError("RemoveGrain: cannot specify U or V mode for planar RGB!");
    }

    //now change undefined mode value and EVERYTHING WILL BREAK
    if (modeU_ <= UNDEFINED_MODE) { 
        modeU_ = mode_;
    }
    if (modeV_ <= UNDEFINED_MODE) {
        modeV_ = modeU_;
    }

    pixelsize = vi.ComponentSize();
    bits_per_pixel = vi.BitsPerComponent();

    if (pixelsize == 1) {
      functions = (env->GetCPUFlags() & CPUF_SSE3) ? sse3_functions
        : (env->GetCPUFlags() & CPUF_SSE2) ? sse2_functions
        : c_functions;

      if (vi.width < 17) { //not enough for XMM
        functions = c_functions;
      }
    }
    else if (pixelsize == 2) {
      if ((env->GetCPUFlags() & CPUF_SSE4) && vi.width >= (16/sizeof(uint16_t) + 1)) {
        // mode 6 and 8 bitdepth clamp specific
        switch (bits_per_pixel) {
        case 10: functions = sse4_functions_16_10; break;
        case 12: functions = sse4_functions_16_12; break;
        case 14: functions = sse4_functions_16_14; break;
        case 16: functions = sse4_functions_16_16; break;
        default: env->ThrowError("Illegal bit-depth: %d!", bits_per_pixel);
        }
      }
      else {
        switch (bits_per_pixel) {
        case 10: functions = c_functions_10; break;
        case 12: functions = c_functions_12; break;
        case 14: functions = c_functions_14; break;
        case 16: functions = c_functions_16; break;
        default: env->ThrowError("Illegal bit-depth: %d!", bits_per_pixel);
        }
      }
    }
    else {// if (pixelsize == 4) 
      if ((env->GetCPUFlags() & CPUF_SSE4) && vi.width >= (16/sizeof(float) + 1))
        functions = sse4_functions_32;
      else
        functions = c_functions_32;
    }
}


PVideoFrame RemoveGrain::GetFrame(int n, IScriptEnvironment* env) {
    auto srcFrame = child->GetFrame(n, env);
    auto dstFrame = env->NewVideoFrame(vi);
    
    int planes_y[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
    int planes_r[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
    int *planes = (vi.IsYUV() || vi.IsYUVA()) ? planes_y : planes_r;

    if (vi.IsPlanarRGB() || vi.IsPlanarRGBA()) {
      for (int p = 0; p < 3; ++p) {
        const int plane = planes[p];

        if (!is_16byte_aligned(srcFrame->GetReadPtr(plane)))
          env->ThrowError("RemoveGrain: Unaligned frame!");

        functions[mode_ + 1](env, srcFrame->GetReadPtr(plane), dstFrame->GetWritePtr(plane), srcFrame->GetRowSize(plane),
          srcFrame->GetHeight(plane), srcFrame->GetPitch(plane), dstFrame->GetPitch(plane));
      }
    } else {
      if(!is_16byte_aligned(srcFrame->GetReadPtr(PLANAR_Y)))
        env->ThrowError("RemoveGrain: Unaligned frame!");

      functions[mode_+1](env, srcFrame->GetReadPtr(PLANAR_Y), dstFrame->GetWritePtr(PLANAR_Y), srcFrame->GetRowSize(PLANAR_Y), 
        srcFrame->GetHeight(PLANAR_Y), srcFrame->GetPitch(PLANAR_Y), dstFrame->GetPitch(PLANAR_Y));

      if (vi.IsPlanar() && !vi.IsY()) {
        if(!is_16byte_aligned(srcFrame->GetReadPtr(PLANAR_U)))
          env->ThrowError("RemoveGrain: Unaligned frame!");

        functions[modeU_ + 1](env, srcFrame->GetReadPtr(PLANAR_U), dstFrame->GetWritePtr(PLANAR_U), srcFrame->GetRowSize(PLANAR_U),
          srcFrame->GetHeight(PLANAR_U), srcFrame->GetPitch(PLANAR_U), dstFrame->GetPitch(PLANAR_U));

        functions[modeV_ + 1](env, srcFrame->GetReadPtr(PLANAR_V), dstFrame->GetWritePtr(PLANAR_V), srcFrame->GetRowSize(PLANAR_V),
          srcFrame->GetHeight(PLANAR_V), srcFrame->GetPitch(PLANAR_V), dstFrame->GetPitch(PLANAR_V));
      }
    }
    if (vi.IsYUVA() || vi.IsPlanarRGBA())
    { // copy alpha
      env->BitBlt(dstFrame->GetWritePtr(PLANAR_A), dstFrame->GetPitch(PLANAR_A), srcFrame->GetReadPtr(PLANAR_A), srcFrame->GetPitch(PLANAR_A), srcFrame->GetRowSize(PLANAR_A_ALIGNED), srcFrame->GetHeight(PLANAR_A));
    }
    return dstFrame;
}


AVSValue __cdecl Create_RemoveGrain(AVSValue args, void*, IScriptEnvironment* env) {
    enum { CLIP, MODE, MODEU, MODEV, PLANAR };
    return new RemoveGrain(args[CLIP].AsClip(), args[MODE].AsInt(1), args[MODEU].AsInt(RemoveGrain::UNDEFINED_MODE), args[MODEV].AsInt(RemoveGrain::UNDEFINED_MODE), args[PLANAR].AsBool(false), env);
}
