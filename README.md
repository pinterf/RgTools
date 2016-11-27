## RgTools

RgTools is a modern rewrite of RemoveGrain, Repair, BackwardClense, Clense, ForwardClense and VerticalCleaner in a single plugin. RgTools is mostly backward compatible to the original plugins.

Some routines might be slightly less efficient than original, some are faster. Output of a few RemoveGrain modes is not exactly identical to the original due to some minor rounding differences which you shouldn't care about. Other functions should be identical.

This plugin is written from scratch and licensed under the [MIT license][1]. Some modes of RemoveGrain and Repair were taken from the Firesledge's Dither package.

From v0.93 new pixel formats are supported
- 10, 12, 14, 16 bit and float 
- Planar RGB, RGBA and YUVA (alpha plane is copied)


### Functions
```
RemoveGrain(clip c, int "mode", int "modeU", int "modeV", bool "planar")
```
Purely spatial denoising function, includes 24 different modes. Additional info can be found in the [wiki][2].

```
Repair(clip c, int "mode", int "modeU", int "modeV", bool "planar")
```
Repairs unwanted artifacts from (but not limited to) RemoveGrain, includes 24 modes.

```
Clense(clip c, clip "previous", clip "next", bool "grey")
```
Temporal median of three frames. Identical to `MedianBlurTemporal(0,0,0,1)` but a lot faster. Can be used as a building block for [many][3] [fancy][4] [medians][5].

```
ForwardClense(clip c, bool "grey")
```
Modified version of Clense that works on current and next frames.

```
BackwardClense(clip c, bool "grey")
```
Modified version of Clense that works on current and previous frames.

```
VerticalCleaner(clip c, int "mode", int "modeU", int "modeV", bool "planar")
```
Very fast vertical median filter. Has only two modes.


  [1]: http://opensource.org/licenses/MIT
  [2]: https://github.com/tp7/RgTools/wiki/RemoveGrain
  [3]: http://mechaweaponsvidya.wordpress.com/2014/01/31/enter-title-here/
  [4]: http://mechaweaponsvidya.wordpress.com/2014/04/23/ricing-your-temporal-medians-for-maximum-speed/
  [5]: http://mechaweaponsvidya.wordpress.com/2014/05/14/clense-versus-mt_clamp/
