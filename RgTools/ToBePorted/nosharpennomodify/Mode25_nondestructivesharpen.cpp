#if	!(defined(MODIFYPLUGIN) || defined(SHLUR))
#define neighbourdiff(minus, plus, center1, center2, neighbour, nullreg)	\
__asm	SSE_RMOVE	center1,			center2		\
__asm	psubusb		center2,			neighbour	\
__asm	psubusb		neighbour,			center1		\
__asm	SSE_RMOVE	minus,				center2		\
__asm	SSE_RMOVE	plus,				neighbour	\
__asm	pcmpeqb		center2,			nullreg		\
__asm	pcmpeqb		neighbour,			nullreg		\
__asm	por			minus,				center2		\
__asm	pand		center2,			neighbour	\
__asm	por			plus,				neighbour	\
__asm	psubusb		minus,				center2		\
__asm	psubusb		plus,				center2

#define neighbourdiff_w(minus, plus, center1, center2, dest, neighbour, nullreg, mwrite)	\
__asm	SSE_RMOVE	center1,			center2		\
__asm	mwrite		dest,				neighbour	\
__asm	psubusb		center2,			neighbour	\
__asm	psubusb		neighbour,			center1		\
__asm	SSE_RMOVE	minus,				center2		\
__asm	SSE_RMOVE	plus,				neighbour	\
__asm	pcmpeqb		center2,			nullreg		\
__asm	pcmpeqb		neighbour,			nullreg		\
__asm	por			minus,				center2		\
__asm	pand		center2,			neighbour	\
__asm	por			plus,				neighbour	\
__asm	psubusb		minus,				center2		\
__asm	psubusb		plus,				center2

#define	SHIFT_MASK1	127

static	const __declspec(align(SSE_INCREMENT)) BYTE	shift_mask[SSE_INCREMENT] =
	{
		SHIFT_MASK1, SHIFT_MASK1, SHIFT_MASK1, SHIFT_MASK1, SHIFT_MASK1, SHIFT_MASK1, SHIFT_MASK1, SHIFT_MASK1		
#if	SSE_INCREMENT == 16
		, SHIFT_MASK1, SHIFT_MASK1, SHIFT_MASK1, SHIFT_MASK1, SHIFT_MASK1, SHIFT_MASK1, SHIFT_MASK1, SHIFT_MASK1
#endif
	};

#define	sharpen(center, minus, plus, reg1, reg2)\
__asm	SSE_RMOVE	reg1,				minus			\
__asm	SSE_RMOVE	reg2,				plus			\
__asm	psubusb		reg1,				plus			\
__asm	psubusb		reg2,				minus			\
__asm	psrlw		plus,				1				\
__asm	psrlw		minus,				1				\
__asm	pand		plus,				shift_mask		\
__asm	pand		minus,				shift_mask		\
__asm	pminub		plus,				reg1			\
__asm	pminub		minus,				reg2			\
__asm	paddusb		center,				plus			\
__asm	psubusb		center,				minus

void	nondestructivesharpen(BYTE *dp, int dpitch, const BYTE *_sp, int spitch, int hblocks, int remainder, int incpitch, int height)
{
__asm	mov			eax,				hblocks
__asm	mov			ebx,				spitch
__asm	mov			edx,				remainder
__asm	pxor		SSE0,				SSE0
#if		SSE_INCREMENT == 16
__asm	add			eax,				eax
#endif
__asm	mov			esi,				_sp
__asm	lea			eax,				[eax * 8 + edx + SSE_INCREMENT + 1]
__asm	sub			esi,				ebx
__asm	sub			dpitch,				eax
__asm	neg			eax
__asm	mov			edi,				dp
__asm	lea			eax,				[ebx + eax + 1]	
__asm	align		16
__asm	column_loop:
__asm	SSE3_MOVE	SSE1,				[esi + ebx + 1]
__asm	SSE3_MOVE	SSE3,				[esi + ebx]	
		neighbourdiff_w(SSE4, SSE5, SSE2, SSE1, [edi], SSE3, SSE0, movd)

__asm	SSE3_MOVE	SSE3,				[esi + ebx + 2]	
		neighbourdiff(SSE6, SSE7, SSE1, SSE2, SSE3, SSE0)
__asm	pminub		SSE4,				SSE6
__asm	pminub		SSE5,				SSE7

__asm	SSE3_MOVE	SSE3,				[esi]	
		neighbourdiff(SSE6, SSE7, SSE2, SSE1, SSE3, SSE0)
__asm	pminub		SSE4,				SSE6
__asm	pminub		SSE5,				SSE7

__asm	SSE3_MOVE	SSE3,				[esi + 1]	
		neighbourdiff(SSE6, SSE7, SSE1, SSE2, SSE3, SSE0)
__asm	pminub		SSE4,				SSE6
__asm	pminub		SSE5,				SSE7

__asm	SSE3_MOVE	SSE3,				[esi + 2]	
		neighbourdiff(SSE6, SSE7, SSE2, SSE1, SSE3, SSE0)
__asm	pminub		SSE4,				SSE6
__asm	pminub		SSE5,				SSE7

__asm	SSE3_MOVE	SSE3,				[esi + 2*ebx]	
		neighbourdiff(SSE6, SSE7, SSE1, SSE2, SSE3, SSE0)
__asm	pminub		SSE4,				SSE6
__asm	pminub		SSE5,				SSE7

__asm	SSE3_MOVE	SSE3,				[esi + 2*ebx + 1]	
		neighbourdiff(SSE6, SSE7, SSE2, SSE1, SSE3, SSE0)
__asm	pminub		SSE4,				SSE6
__asm	pminub		SSE5,				SSE7

__asm	SSE3_MOVE	SSE3,				[esi + 2*ebx + 2]	
		neighbourdiff(SSE6, SSE7, SSE1, SSE2, SSE3, SSE0)
__asm	pminub		SSE4,				SSE6
__asm	pminub		SSE5,				SSE7
		sharpen(SSE1, SSE4, SSE5, SSE6, SSE7)
__asm	SSE_MOVE	[edi + 1],			SSE1
// now the pixels in the middle
__asm	add			esi,				SSE_INCREMENT
__asm	add			edi,				SSE_INCREMENT + 1
__asm	mov			ecx,				hblocks
__asm	align		16
__asm	middle_loop:
__asm	SSE3_MOVE	SSE1,				[esi + ebx + 1]
__asm	SSE3_MOVE	SSE3,				[esi + ebx]	
		neighbourdiff(SSE4, SSE5, SSE2, SSE1, SSE3, SSE0)

__asm	SSE3_MOVE	SSE3,				[esi + ebx + 2]	
		neighbourdiff(SSE6, SSE7, SSE1, SSE2, SSE3, SSE0)
__asm	pminub		SSE4,				SSE6
__asm	pminub		SSE5,				SSE7

__asm	SSE3_MOVE	SSE3,				[esi]	
		neighbourdiff(SSE6, SSE7, SSE2, SSE1, SSE3, SSE0)
__asm	pminub		SSE4,				SSE6
__asm	pminub		SSE5,				SSE7

__asm	SSE3_MOVE	SSE3,				[esi + 1]	
		neighbourdiff(SSE6, SSE7, SSE1, SSE2, SSE3, SSE0)
__asm	pminub		SSE4,				SSE6
__asm	pminub		SSE5,				SSE7

__asm	SSE3_MOVE	SSE3,				[esi + 2]	
		neighbourdiff(SSE6, SSE7, SSE2, SSE1, SSE3, SSE0)
__asm	pminub		SSE4,				SSE6
__asm	pminub		SSE5,				SSE7

__asm	SSE3_MOVE	SSE3,				[esi + 2*ebx]	
		neighbourdiff(SSE6, SSE7, SSE1, SSE2, SSE3, SSE0)
__asm	pminub		SSE4,				SSE6
__asm	pminub		SSE5,				SSE7

__asm	SSE3_MOVE	SSE3,				[esi + 2*ebx + 1]	
		neighbourdiff(SSE6, SSE7, SSE2, SSE1, SSE3, SSE0)
__asm	pminub		SSE4,				SSE6
__asm	pminub		SSE5,				SSE7

__asm	SSE3_MOVE	SSE3,				[esi + 2*ebx + 2]	
		neighbourdiff(SSE6, SSE7, SSE1, SSE2, SSE3, SSE0)
__asm	pminub		SSE4,				SSE6
__asm	pminub		SSE5,				SSE7
__asm	add			esi,				SSE_INCREMENT
		sharpen(SSE1, SSE4, SSE5, SSE6, SSE7)
__asm	SSE_MOVE	[edi],				SSE1
__asm	add			edi,				SSE_INCREMENT
__asm	dec			ecx
__asm	jnz			middle_loop
// the last pixels
__asm	add			esi,				edx
__asm	add			edi,				edx
__asm	SSE3_MOVE	SSE1,				[esi + ebx + 1]
__asm	SSE3_MOVE	SSE3,				[esi + ebx]	
		neighbourdiff(SSE4, SSE5, SSE2, SSE1, SSE3, SSE0)

__asm	SSE3_MOVE	SSE3,				[esi + ebx + 2]	
		neighbourdiff_w(SSE6, SSE7, SSE1, SSE2, [edi + 1], SSE3, SSE0, SSE_MOVE)
__asm	pminub		SSE4,				SSE6
__asm	pminub		SSE5,				SSE7

__asm	SSE3_MOVE	SSE3,				[esi]	
		neighbourdiff(SSE6, SSE7, SSE2, SSE1, SSE3, SSE0)
__asm	pminub		SSE4,				SSE6
__asm	pminub		SSE5,				SSE7

__asm	SSE3_MOVE	SSE3,				[esi + 1]	
		neighbourdiff(SSE6, SSE7, SSE1, SSE2, SSE3, SSE0)
__asm	pminub		SSE4,				SSE6
__asm	pminub		SSE5,				SSE7

__asm	SSE3_MOVE	SSE3,				[esi + 2]	
		neighbourdiff(SSE6, SSE7, SSE2, SSE1, SSE3, SSE0)
__asm	pminub		SSE4,				SSE6
__asm	pminub		SSE5,				SSE7

__asm	SSE3_MOVE	SSE3,				[esi + 2*ebx]	
		neighbourdiff(SSE6, SSE7, SSE1, SSE2, SSE3, SSE0)
__asm	pminub		SSE4,				SSE6
__asm	pminub		SSE5,				SSE7

__asm	SSE3_MOVE	SSE3,				[esi + 2*ebx + 1]	
		neighbourdiff(SSE6, SSE7, SSE2, SSE1, SSE3, SSE0)
__asm	pminub		SSE4,				SSE6
__asm	pminub		SSE5,				SSE7

__asm	SSE3_MOVE	SSE3,				[esi + 2*ebx + 2]	
		neighbourdiff(SSE6, SSE7, SSE1, SSE2, SSE3, SSE0)
__asm	pminub		SSE4,				SSE6
__asm	pminub		SSE5,				SSE7
__asm	add			esi,				eax
		sharpen(SSE1, SSE4, SSE5, SSE6, SSE7)
__asm	SSE_MOVE	[edi],				SSE1
__asm	add			edi,				dpitch
__asm	dec			height
__asm	jnz			column_loop
}
#endif	!(defined(MODIFYPLUGIN) || defined(SHLUR))
