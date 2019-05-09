#ifdef	SHARPEN
void	SmartRG(BYTE *dp, int dpitch, const BYTE *_sp, int spitch, int hblocks, int remainder, int incpitch, int height, int strength)
#else
void	SmartRG(BYTE *dp, int dpitch, const BYTE *_sp, int spitch, int hblocks, int remainder, int incpitch, int height)
#endif
{
#if defined(CVERSION) && !defined(MODIFYPLUGIN) && !defined(SHLUR)
		_sp -= spitch;
		int width = (hblocks + 2) * SSE_INCREMENT + remainder;
		int	spitch2 = spitch - width;
		dpitch -= width;
		do
		{
			int	w = width;
			dp[0] = _sp[spitch];
			do
			{
				int Max = min(_sp[0], _sp[2 * spitch  + 2]);
				int Min = max(_sp[0], _sp[2 * spitch  + 2]);
				int max1 = min(_sp[1], _sp[2*spitch  + 1]);
				Max = max(Max, max1);
				int min1 = max(_sp[1], _sp[2*spitch  + 1]);
				Min = min(Min, min1);
				max1 = min(_sp[2], _sp[2*spitch]);
				Max = max(Max, max1);
				min1 = max(_sp[2], _sp[2*spitch]);
				Min = min(Min, min1);
				max1 = min(_sp[spitch], _sp[spitch + 2]);
				Max = max(Max, max1);
				min1 = max(_sp[spitch], _sp[spitch + 2]);
				Min = min(Min, min1);
				if( Min > Max )
				{
					max1 = Max;
					Max = Min;
					Min = max1;
				}
				*++dp = max(min(_sp[spitch + 1], Max), Min);
				++_sp;
			} while( --w );
			dp[1] = _sp[spitch + 1];
			dp += dpitch;
			_sp += spitch2; 
		} while( --height );
#else // defined(CVERSION) && !defined(MODIFYPLUGIN) && !defined(SHLUR)
__asm	mov			eax,				hblocks
__asm	mov			ebx,				spitch
#ifdef	MODIFYPLUGIN
__asm	mov			ecx,				eax
#endif
__asm	mov			edx,				remainder
#if		SSE_INCREMENT == 16
__asm	add			eax,				eax
#endif
__asm	mov			esi,				_sp
#ifdef	MODIFYPLUGIN
__asm	lea			eax,				[eax * 8 + edx]
#else
__asm	lea			eax,				[eax * 8 + edx + SSE_INCREMENT + 1]
#endif
__asm	sub			esi,				ebx
__asm	sub			dpitch,				eax
__asm	neg			eax
__asm	mov			edi,				dp
#ifdef	MODIFYPLUGIN
__asm	inc			edi
__asm	lea			eax,				[ebx + eax]	
#else
__asm	lea			eax,				[ebx + eax + 1]	
#ifdef	SHARPEN
__asm	mov			spitch,				eax
__asm	mov			eax,				strength
#endif
__asm	align		16
__asm	column_loop:
__asm	SSE3_MOVE	SSE0,				[esi]
__asm	SSE3_MOVE	SSE7,				[esi + 2*ebx + 2]
__asm	SSE_RMOVE	SSE1,				SSE0
__asm	pminub		SSE0,				SSE7
__asm	pmaxub		SSE1,				SSE7
__asm	SSE3_MOVE	SSE4,				[esi + 2]
__asm	SSE3_MOVE	SSE7,				[esi + 2*ebx]
__asm	SSE_RMOVE	SSE5,				SSE4
__asm	SSE3_MOVE	SSE2,				[esi + 1]
__asm	pminub		SSE4,				SSE7
__asm	pmaxub		SSE5,				SSE7
__asm	pmaxub		SSE0,				SSE4
__asm	SSE3_MOVE	SSE7,				[esi + 2*ebx + 1]
__asm	SSE_RMOVE	SSE3,				SSE2
__asm	pminub		SSE1,				SSE5

__asm	pminub		SSE2,				SSE7
__asm	SSE3_MOVE	SSE4,				[esi + ebx]
__asm	pmaxub		SSE3,				SSE7
__asm	movd		[edi],				SSE4
__asm	pmaxub		SSE0,				SSE2
__asm	SSE3_MOVE	SSE7,				[esi + ebx + 2]
__asm	SSE_RMOVE	SSE5,				SSE4
__asm	pminub		SSE1,				SSE3

__asm	pminub		SSE4,				SSE7
__asm	pmaxub		SSE5,				SSE7
__asm	pmaxub		SSE0,				SSE4
__asm	pminub		SSE1,				SSE5
__asm	SSE_RMOVE	SSE2,				SSE0
#if	(ISSE > 1) || defined(SHLUR)
__asm	SSE3_MOVE	SSE4,				[esi + ebx + 1]				
#endif
__asm	pminub		SSE0,				SSE1
__asm	pmaxub		SSE2,				SSE1
#ifdef	SHLUR
		sharpen(SSE4, SSE0, SSE2, rshift[eax], shift_mask[eax], SSE7, SSE3)
__asm	SSE_MOVE	[edi + 1],			SSE4
#else
#if	ISSE > 1
__asm	pmaxub		SSE0,				SSE4
#else
__asm	pmaxub		SSE0,				[esi + ebx + 1]	
#endif
__asm	pminub		SSE0,				SSE2
__asm	SSE_MOVE	[edi + 1],			SSE0
#endif	// SHLUR
// now the pixels in the middle
__asm	add			esi,				SSE_INCREMENT
__asm	add			edi,				SSE_INCREMENT + 1
__asm	mov			ecx,				hblocks
#endif // MODIFYPLUGIN
__asm	align		16
__asm	middle_loop:
__asm	SSE3_MOVE	SSE0,				[esi]
__asm	SSE3_MOVE	SSE7,				[esi + 2*ebx + 2]
__asm	SSE_RMOVE	SSE1,				SSE0
__asm	SSE3_MOVE	SSE4,				[esi + 2]
__asm	pminub		SSE0,				SSE7
__asm	SSE3_MOVE	SSE6,				[esi + 2*ebx]
__asm	SSE_RMOVE	SSE5,				SSE4
__asm	pmaxub		SSE1,				SSE7
__asm	pminub		SSE4,				SSE6
__asm	SSE3_MOVE	SSE2,				[esi + 1]
__asm	pmaxub		SSE5,				SSE6
__asm	pmaxub		SSE0,				SSE4
__asm	SSE3_MOVE	SSE7,				[esi + 2*ebx + 1]
__asm	SSE_RMOVE	SSE3,				SSE2
__asm	pminub		SSE1,				SSE5

__asm	pminub		SSE2,				SSE7
__asm	SSE3_MOVE	SSE4,				[esi + ebx]
__asm	pmaxub		SSE3,				SSE7
__asm	pmaxub		SSE0,				SSE2
__asm	SSE3_MOVE	SSE7,				[esi + ebx + 2]
__asm	SSE_RMOVE	SSE5,				SSE4
__asm	pminub		SSE1,				SSE3

__asm	pminub		SSE4,				SSE7
__asm	pmaxub		SSE5,				SSE7
__asm	pmaxub		SSE0,				SSE4
__asm	pminub		SSE1,				SSE5
__asm	SSE_RMOVE	SSE2,				SSE0

#if		(ISSE > 1) || defined(SHLUR)
#ifdef	MODIFYPLUGIN
__asm	SSE3_MOVE	SSE4,				[edi]
#else
__asm	SSE3_MOVE	SSE4,				[esi + ebx + 1]	
#endif
#endif
#if		MODIFYPLUGIN > 0
__asm	SSE3_MOVE	SSE5,				[esi + ebx + 1]		
#endif

__asm	pminub		SSE0,				SSE1
__asm	pmaxub		SSE2,				SSE1
#if		MODIFYPLUGIN > 0
__asm	pminub		SSE0,				SSE5
__asm	pmaxub		SSE2,				SSE5
#endif

#ifdef	SHLUR
		sharpen(SSE4, SSE0, SSE2, rshift[eax], shift_mask[eax], SSE7, SSE3)
__asm	add			esi,				SSE_INCREMENT
__asm	SSE_MOVE	[edi],				SSE4
#else
#if	ISSE > 1
__asm	pmaxub		SSE0,				SSE4
#else
#ifdef	MODIFYPLUGIN
__asm	pmaxub		SSE0,				[edi]
#else
__asm	pmaxub		SSE0,				[esi + ebx + 1]	
#endif
#endif
__asm	pminub		SSE0,				SSE2
__asm	add			esi,				SSE_INCREMENT
__asm	SSE_MOVE	[edi],				SSE0
#endif	// SHLUR
__asm	add			edi,				SSE_INCREMENT
__asm	dec			ecx
__asm	jnz			middle_loop
// the last pixels
__asm	add			esi,				edx
__asm	SSE3_MOVE	SSE0,				[esi]
__asm	add			edi,				edx
__asm	SSE3_MOVE	SSE7,				[esi + 2*ebx + 2]
__asm	SSE_RMOVE	SSE1,				SSE0
__asm	SSE3_MOVE	SSE4,				[esi + 2]
__asm	pminub		SSE0,				SSE7
__asm	pmaxub		SSE1,				SSE7
__asm	SSE3_MOVE	SSE6,				[esi + 2*ebx]
__asm	SSE_RMOVE	SSE5,				SSE4
__asm	SSE3_MOVE	SSE2,				[esi + 1]
__asm	pminub		SSE4,				SSE6
__asm	pmaxub		SSE5,				SSE6
__asm	pmaxub		SSE0,				SSE4
__asm	SSE3_MOVE	SSE7,				[esi + 2*ebx + 1]
__asm	SSE_RMOVE	SSE3,				SSE2
__asm	pminub		SSE1,				SSE5

__asm	pminub		SSE2,				SSE7
__asm	SSE3_MOVE	SSE4,				[esi + ebx]
__asm	pmaxub		SSE3,				SSE7
__asm	pmaxub		SSE0,				SSE2
__asm	SSE3_MOVE	SSE7,				[esi + ebx + 2]
__asm	SSE_RMOVE	SSE5,				SSE4
#ifndef	MODIFYPLUGIN
__asm	SSE_MOVE	[edi + 1],			SSE7
#endif
__asm	pminub		SSE1,				SSE3

__asm	pminub		SSE4,				SSE7
__asm	pmaxub		SSE5,				SSE7
__asm	pmaxub		SSE0,				SSE4
__asm	pminub		SSE1,				SSE5
__asm	SSE_RMOVE	SSE2,				SSE0

#if		(ISSE > 1) || defined(SHLUR)
#ifdef	MODIFYPLUGIN
__asm	SSE3_MOVE	SSE4,				[edi]
#else
__asm	SSE3_MOVE	SSE4,				[esi + ebx + 1]	
#endif
#endif
#if		MODIFYPLUGIN > 0
__asm	SSE3_MOVE	SSE5,				[esi + ebx + 1]		
#endif
__asm	pminub		SSE0,				SSE1
__asm	pmaxub		SSE2,				SSE1
#if		MODIFYPLUGIN > 0
__asm	pminub		SSE0,				SSE5
__asm	pmaxub		SSE2,				SSE5
#endif

#ifdef	SHLUR
		sharpen(SSE4, SSE0, SSE2, rshift[eax], shift_mask[eax], SSE7, SSE3)
#ifdef	SHARPEN
__asm	add			esi,				spitch
#else
__asm	add			esi,				eax
#endif
__asm	SSE_MOVE	[edi],				SSE4
#else
#if	ISSE > 1
__asm	pmaxub		SSE0,				SSE4
#else
#ifdef	MODIFYPLUGIN
__asm	pmaxub		SSE0,				[edi]
#else
__asm	pmaxub		SSE0,				[esi + ebx + 1]	
#endif
#endif
__asm	pminub		SSE0,				SSE2
__asm	add			esi,				eax
__asm	SSE_MOVE	[edi],				SSE0
#endif	// SHLUR
__asm	add			edi,				dpitch
__asm	dec			height
#ifdef	MODIFYPLUGIN
__asm	mov			ecx,				hblocks
__asm	jnz			middle_loop
#else
__asm	jnz			column_loop
#endif
#endif // defined(CVERSION) && !defined(MODIFYPLUGIN) && !defined(SHLUR)
}
