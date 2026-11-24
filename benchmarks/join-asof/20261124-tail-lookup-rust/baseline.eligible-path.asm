  64093d:	mov    r12,QWORD PTR [r13+0x4e0]
  640944:	test   r12,r12
  640947:	je     642ad0
  64094d:	shl    rdx,0x4
  640951:	mov    rcx,QWORD PTR [rsp+0x10]
  640956:	mov    r11,QWORD PTR [rcx+rdx*1-0x10]
  64095b:	mov    edi,DWORD PTR [rcx+rdx*1-0x8]
  64095f:	mov    rdx,QWORD PTR [r13+0x4e8]
  640966:	mov    rsi,rdx
  640969:	mov    rbx,r12
  64096c:	movzx  r14d,WORD PTR [rbx+0xbba]
  640974:	mov    ecx,r14d
  640977:	shl    ecx,0x8
  64097a:	lea    rcx,[rcx+r14*8]
  64097e:	mov    r8,0xffffffffffffffff
  640985:	xor    r9d,r9d
  640988:	nop    DWORD PTR [rax+rax*1+0x0]
  640990:	cmp    rcx,r9
  640993:	je     6409c0
  640995:	cmp    r11,QWORD PTR [rbx+r8*8+0x10]
  64099a:	seta   bpl
  64099e:	sbb    bpl,0x0
  6409a2:	add    r9,0x108
  6409a9:	inc    r8
  6409ac:	cmp    bpl,0x1
  6409b0:	je     640990
  6409b2:	movzx  ecx,bpl
  6409b6:	test   ecx,ecx
  6409b8:	jne    6409c3
  6409ba:	jmp    6409d7
  6409bc:	nop    DWORD PTR [rax+0x0]
  6409c0:	mov    r8,r14
  6409c3:	sub    rsi,0x1
  6409c7:	jb     642ad0
  6409cd:	mov    rbx,QWORD PTR [rbx+r8*8+0xbc0]
  6409d5:	jmp    64096c
  6409d7:	mov    rsi,QWORD PTR [rbx+r9*1]
  6409db:	cmp    rsi,rdi
  6409de:	jbe    642b04
  6409e4:	mov    rcx,QWORD PTR [rbx+r9*1-0x8]
  6409e9:	mov    rsi,QWORD PTR [rsp+0x8]
  6409ee:	cmp    QWORD PTR [rcx+rdi*8],rsi
  6409f2:	jle    640c05

  640c05:	movzx  esi,WORD PTR [r12+0xbba]
  640c0e:	mov    eax,esi
  640c10:	shl    eax,0x8
  640c13:	lea    rcx,[rax+rsi*8]
  640c17:	mov    rax,0xffffffffffffffff
  640c1e:	xor    r8d,r8d
  640c21:	data16 data16 data16 data16 data16 cs nop WORD PTR [rax+rax*1+0x0]
  640c30:	cmp    rcx,r8
  640c33:	je     640bf0
  640c35:	cmp    r11,QWORD PTR [r12+rax*8+0x10]
  640c3a:	seta   r9b
  640c3e:	sbb    r9b,0x0
  640c42:	add    r8,0x108
  640c49:	inc    rax
  640c4c:	cmp    r9b,0x1
  640c50:	je     640c30
  640c52:	movzx  ecx,r9b
  640c56:	test   ecx,ecx
  640c58:	jne    640bf3
  640c5a:	mov    eax,edi
  640c5c:	mov    rsi,QWORD PTR [r12+r8*1]
  640c60:	cmp    rsi,rax
  640c63:	jbe    642b01
  640c69:	mov    rcx,QWORD PTR [r12+r8*1-0x8]
  640c6e:	mov    rax,QWORD PTR [rcx+rax*8]
  640c72:	mov    rcx,rax
  640c75:	sar    rcx,0x3f
  640c79:	mov    rsi,QWORD PTR [rsp+0x8]
  640c7e:	mov    rdx,rsi
  640c81:	sar    rdx,0x3f
  640c85:	sub    rsi,QWORD PTR [r13+0x2f8]
  640c8c:	sbb    rdx,0x0
  640c90:	xor    r14d,r14d
  640c93:	cmp    rax,rsi
  640c96:	sbb    rcx,rdx
  640c99:	mov    r15d,0x0
  640c9f:	mov    r12,QWORD PTR [rsp+0xe8]
  640ca7:	jl     640ce0
