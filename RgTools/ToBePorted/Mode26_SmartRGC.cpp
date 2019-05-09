#ifdef	SHARPEN
void	SmartRGC(BYTE *dp, int dpitch, const BYTE *_sp, int spitch, int hblocks, int remainder, int incpitch, int height, int strength)
#else
void	SmartRGC(BYTE *dp, int dpitch, const BYTE *_sp, int spitch, int hblocks, int remainder, int incpitch, int height)
#endif
{
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
__asm	SSE3_MOVE	SSE7,				[esi]
__asm	SSE3_MOVE	SSE6,				[esi + 1]
__asm	SSE_RMOVE	SSE0,				SSE7
__asm	SSE_RMOVE	SSE1,				SSE6
__asm	SSE3_MOVE	SSE5,				[esi + 2]
__asm	pminub		SSE0,				SSE6
__asm	SSE_RMOVE	SSE2,				SSE5
__asm	pmaxub		SSE1,				SSE7
__asm	pminub		SSE2,				SSE6
__asm	SSE3_MOVE	SSE4,				[esi + ebx + 2]
__asm	pmaxub		SSE6,				SSE5
__asm	pmaxub		SSE0,				SSE2
__asm	SSE_RMOVE	SSE3,				SSE4
__asm	pminub		SSE1,				SSE6
__asm	pminub		SSE3,				SSE5
__asm	SSE3_MOVE	SSE6,				[esi + 2*ebx + 2]
__asm	pmaxub		SSE5,				SSE4
__asm	pmaxub		SSE0,				SSE3
__asm	SSE_RMOVE	SSE2,				SSE6
__asm	pminub		SSE1,				SSE5
__asm	pminub		SSE2,				SSE4
__asm	SSE3_MOVE	SSE5,				[esi + 2*ebx + 1]
__asm	pmaxub		SSE4,				SSE6
__asm	pmaxub		SSE0,				SSE2
__asm	SSE_RMOVE	SSE3,				SSE5
__asm	pminub		SSE1,				SSE4
__asm	pminub		SSE3,				SSE6
__asm	SSE3_MOVE	SSE4,				[esi + 2*ebx]
__asm	pmaxub		SSE6,				SSE5
__asm	pmaxub		SSE0,				SSE3
__asm	SSE_RMOVE	SSE2,				SSE4
__asm	pminub		SSE1,				SSE6
__asm	pminub		SSE2,				SSE5
__asm	SSE3_MOVE	SSE6,				[esi + ebx]
__asm	pmaxub		SSE5,				SSE4
__asm	pmaxub		SSE0,				SSE2
__asm	SSE_RMOVE	SSE3,				SSE6
__asm	pminub		SSE1,				SSE5
__asm	movd		[edi],				SSE6
__asm	pminub		SSE3,				SSE4
__asm	SSE_RMOVE	SSE2,				SSE7
__asm	pmaxub		SSE4,				SSE6
__asm	pmaxub		SSE0,				SSE3
__asm	pminub		SSE1,				SSE4
__asm	pminub		SSE2,				SSE6
__asm	pmaxub		SSE7,				SSE6
__asm	pmaxub		SSE0,				SSE2
__asm	pminub		SSE1,				SSE7

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
__asm	SSE3_MOVE	SSE7,				[esi]
__asm	SSE3_MOVE	SSE6,				[esi + 1]
__asm	SSE_RMOVE	SSE0,				SSE7
__asm	SSE_RMOVE	SSE1,				SSE6
__asm	SSE3_MOVE	SSE5,				[esi + 2]
__asm	pminub		SSE0,				SSE6
__asm	SSE_RMOVE	SSE2,				SSE5
__asm	pmaxub		SSE1,				SSE7
__asm	pminub		SSE2,				SSE6
__asm	SSE3_MOVE	SSE4,				[esi + ebx + 2]
__asm	pmaxub		SSE6,				SSE5
__asm	pmaxub		SSE0,				SSE2
__asm	SSE_RMOVE	SSE3,				SSE4
__asm	pminub		SSE1,				SSE6
__asm	pminub		SSE3,				SSE5
__asm	SSE3_MOVE	SSE6,				[esi + 2*ebx + 2]
__asm	pmaxub		SSE5,				SSE4
__asm	pmaxub		SSE0,				SSE3
__asm	SSE_RMOVE	SSE2,				SSE6
__asm	pminub		SSE1,				SSE5
__asm	pminub		SSE2,				SSE4
__asm	SSE3_MOVE	SSE5,				[esi + 2*ebx + 1]
__asm	pmaxub		SSE4,				SSE6
__asm	pmaxub		SSE0,				SSE2
__asm	SSE_RMOVE	SSE3,				SSE5
__asm	pminub		SSE1,				SSE4
__asm	pminub		SSE3,				SSE6
__asm	SSE3_MOVE	SSE4,				[esi + 2*ebx]
__asm	pmaxub		SSE6,				SSE5
__asm	pmaxub		SSE0,				SSE3
__asm	SSE_RMOVE	SSE2,				SSE4
__asm	pminub		SSE1,				SSE6
__asm	pminub		SSE2,				SSE5
__asm	SSE3_MOVE	SSE6,				[esi + ebx]
__asm	pmaxub		SSE5,				SSE4
__asm	pmaxub		SSE0,				SSE2
__asm	SSE_RMOVE	SSE3,				SSE6
__asm	pminub		SSE1,				SSE5
__asm	pminub		SSE3,				SSE4
__asm	SSE_RMOVE	SSE2,				SSE7
__asm	pmaxub		SSE4,				SSE6
__asm	pmaxub		SSE0,				SSE3
__asm	pminub		SSE1,				SSE4
__asm	pminub		SSE2,				SSE6
__asm	pmaxub		SSE7,				SSE6
__asm	pmaxub		SSE0,				SSE2
__asm	pminub		SSE1,				SSE7

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
__asm	SSE3_MOVE	SSE7,				[esi]
__asm	add			edi,				edx
__asm	SSE3_MOVE	SSE6,				[esi + 1]
__asm	SSE_RMOVE	SSE0,				SSE7
__asm	SSE_RMOVE	SSE1,				SSE6
__asm	SSE3_MOVE	SSE5,				[esi + 2]
__asm	pminub		SSE0,				SSE6
__asm	SSE_RMOVE	SSE2,				SSE5
__asm	pmaxub		SSE1,				SSE7
__asm	pminub		SSE2,				SSE6
__asm	SSE3_MOVE	SSE4,				[esi + ebx + 2]
__asm	pmaxub		SSE6,				SSE5
__asm	pmaxub		SSE0,				SSE2
__asm	SSE_RMOVE	SSE3,				SSE4
__asm	pminub		SSE1,				SSE6
__asm	SSE_MOVE	[edi + 1],			SSE4
__asm	pminub		SSE3,				SSE5
__asm	SSE3_MOVE	SSE6,				[esi + 2*ebx + 2]
__asm	pmaxub		SSE5,				SSE4
__asm	pmaxub		SSE0,				SSE3
__asm	SSE_RMOVE	SSE2,				SSE6
__asm	pminub		SSE1,				SSE5
__asm	pminub		SSE2,				SSE4
__asm	SSE3_MOVE	SSE5,				[esi + 2*ebx + 1]
__asm	pmaxub		SSE4,				SSE6
__asm	pmaxub		SSE0,				SSE2
__asm	SSE_RMOVE	SSE3,				SSE5
__asm	pminub		SSE1,				SSE4
__asm	pminub		SSE3,				SSE6
__asm	SSE3_MOVE	SSE4,				[esi + 2*ebx]
__asm	pmaxub		SSE6,				SSE5
__asm	pmaxub		SSE0,				SSE3
__asm	SSE_RMOVE	SSE2,				SSE4
__asm	pminub		SSE1,				SSE6
__asm	pminub		SSE2,				SSE5
__asm	SSE3_MOVE	SSE6,				[esi + ebx]
__asm	pmaxub		SSE5,				SSE4
__asm	pmaxub		SSE0,				SSE2
__asm	SSE_RMOVE	SSE3,				SSE6
__asm	pminub		SSE1,				SSE5
__asm	pminub		SSE3,				SSE4
__asm	SSE_RMOVE	SSE2,				SSE7
__asm	pmaxub		SSE4,				SSE6
__asm	pmaxub		SSE0,				SSE3
__asm	pminub		SSE1,				SSE4
__asm	pminub		SSE2,				SSE6
__asm	pmaxub		SSE7,				SSE6
__asm	pmaxub		SSE0,				SSE2
__asm	pminub		SSE1,				SSE7

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
}
