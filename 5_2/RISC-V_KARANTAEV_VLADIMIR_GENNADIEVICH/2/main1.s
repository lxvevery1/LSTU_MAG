.file      "main.c"
.option    pic
.attribute arch, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_zicsr2p0_zifencei2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zca1p0_zcd1p0"
.attribute unaligned_access, 0
.attribute stack_align, 16
.text
.section   .rodata.str1.8, "aMS", @progbits, 1
.align     3

.LC0:
	.string "FizzBuzz"
	.align  3

.LC1:
	.string "Fizz"
	.align  3

.LC2:
	.string "Buzz"
	.align  3

.LC3:
	.string "%d\n"
	.text
	.align  1
	.globl  main
	.type   main, @function

main:
.LFB11:
	.cfi_startproc
	addi sp, sp, -80
	.cfi_def_cfa_offset 80
	sd   ra, 72(sp)
	sd   s0, 64(sp)
	sd   s1, 56(sp)
	sd   s2, 48(sp)
	sd   s3, 40(sp)
	sd   s4, 32(sp)
	sd   s5, 24(sp)
	sd   s6, 16(sp)
	sd   s7, 8(sp)
	.cfi_offset 1, -8
	.cfi_offset 8, -16
	.cfi_offset 9, -24
	.cfi_offset 18, -32
	.cfi_offset 19, -40
	.cfi_offset 20, -48
	.cfi_offset 21, -56
	.cfi_offset 22, -64
	.cfi_offset 23, -72
	li   s0, 1
	li   s1, -2004316160
	addi s1, s1, -1911   // это для загруки 64
	li   s2, 1431654400
	addi s2, s2, 1366
	li   s3, 1717985280
	addi s3, s3, 1639
	lla  s6, .LC3
	lla  s5, .LC2
	lla  s7, .LC1
	lla  s4, .LC0
	j    .L6

.L10:
	mv   a0, s4
	call puts@plt

.L3:
	addiw s0, s0, 1
	li    a5, 101
	beq   s0, a5, .L9

.L6:
	mul   a5, s0, s1
	srli  a5, a5, 32
	addw  a5, s0, a5
	sraiw a5, a5, 3
	sraiw a4, s0, 31
	subw  a5, a5, a4
	slliw a4, a5, 4
	subw  a5, a4, a5
	subw  a5, s0, a5
	beq   a5, zero, .L10
	mul   a5, s0, s2
	srli  a5, a5, 32
	sraiw a4, s0, 31
	subw  a5, a5, a4
	slliw a4, a5, 1
	addw  a5, a4, a5
	subw  a5, s0, a5
	beq   a5, zero, .L11
	mul   a5, s0, s3
	srai  a5, a5, 33
	sraiw a4, s0, 31
	subw  a5, a5, a4
	slliw a4, a5, 2
	addw  a5, a4, a5
	subw  a5, s0, a5
	bne   a5, zero, .L5
	mv    a0, s5
	call  puts@plt
	j     .L3

.L11:
	mv   a0, s7
	call puts@plt
	j    .L3

.L5:
	mv   a1, s0
	mv   a0, s6
	call printf@plt
	j    .L3

.L9:
	li   a0, 0
	ld   ra, 72(sp)
	.cfi_restore 1
	ld   s0, 64(sp)
	.cfi_restore 8
	ld   s1, 56(sp)
	.cfi_restore 9
	ld   s2, 48(sp)
	.cfi_restore 18
	ld   s3, 40(sp)
	.cfi_restore 19
	ld   s4, 32(sp)
	.cfi_restore 20
	ld   s5, 24(sp)
	.cfi_restore 21
	ld   s6, 16(sp)
	.cfi_restore 22
	ld   s7, 8(sp)
	.cfi_restore 23
	addi sp, sp, 80
	.cfi_def_cfa_offset 0
	jr   ra
	.cfi_endproc

.LFE11:
	.size    main, .-main
	.ident   "GCC: (GNU) 15.1.0"
	.section .note.GNU-stack, "", @progbits
