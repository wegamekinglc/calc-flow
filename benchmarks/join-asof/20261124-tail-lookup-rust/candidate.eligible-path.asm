  6408e7:	mov    rcx,QWORD PTR [r12+0x4e0]
  6408ef:	test   rcx,rcx
  6408f2:	je     642970
  6408f8:	mov    r14,rcx
  6408fb:	shl    rdx,0x4
  6408ff:	mov    r11,QWORD PTR [r10+rdx*1-0x10]
  640904:	mov    edi,DWORD PTR [r10+rdx*1-0x8]
  640909:	mov    r12,QWORD PTR [rsp+0x20]
  64090e:	mov    rsi,QWORD PTR [r12+0x4e8]
  640916:	mov    rdx,rcx
  640919:	movzx  ebx,WORD PTR [rdx+0xbba]
  640920:	mov    ecx,ebx
  640922:	shl    ecx,0x8
  640925:	lea    rcx,[rcx+rbx*8]
  640929:	mov    r8,0xffffffffffffffff
  640930:	xor    r9d,r9d
  640933:	data16 data16 data16 cs nop WORD PTR [rax+rax*1+0x0]
  640940:	cmp    rcx,r9
  640943:	je     640970
  640945:	cmp    r11,QWORD PTR [rdx+r8*8+0x10]
  64094a:	seta   bpl
  64094e:	sbb    bpl,0x0
  640952:	add    r9,0x108
  640959:	inc    r8
  64095c:	cmp    bpl,0x1
  640960:	je     640940
  640962:	movzx  ecx,bpl
  640966:	test   ecx,ecx
  640968:	jne    640973
  64096a:	jmp    64098c
  64096c:	nop    DWORD PTR [rax+0x0]
  640970:	mov    r8,rbx
  640973:	sub    rsi,0x1
  640977:	jb     642970
  64097d:	mov    rdx,QWORD PTR [rdx+r8*8+0xbc0]
  640985:	mov    r12,QWORD PTR [rsp+0x20]
  64098a:	jmp    640919
  64098c:	mov    rsi,QWORD PTR [rdx+r9*1]
  640990:	cmp    rsi,rdi
  640993:	mov    r13,QWORD PTR [rsp+0x298]
  64099b:	jbe    642998
  6409a1:	mov    rcx,QWORD PTR [rdx+r9*1-0x8]
  6409a6:	mov    rcx,QWORD PTR [rcx+rdi*8]
  6409aa:	cmp    rcx,QWORD PTR [rsp+0x10]
  6409af:	jle    640c53

  640c53:	mov    rax,rcx
  640c56:	sar    rax,0x3f
  640c5a:	mov    rsi,QWORD PTR [rsp+0x10]
  640c5f:	mov    rdx,rsi
  640c62:	sar    rdx,0x3f
  640c66:	sub    rsi,QWORD PTR [r12+0x2f8]
  640c6e:	sbb    rdx,0x0
  640c72:	xor    r14d,r14d
  640c75:	cmp    rcx,rsi
  640c78:	sbb    rax,rdx
  640c7b:	mov    r15d,0x0
  640c81:	jl     640ca0
