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
	.string  "%d\n"
	.section .text.startup, "ax", @progbits
	.align   1
	.globl   main
	.type    main, @function

main:
.LFB11:
	.cfi_startproc
	addi  sp, sp, -64
	.cfi_def_cfa_offset 64
	sd    s1, 40(sp)
	sd    s2, 32(sp)
	sd    s3, 24(sp)
	sd    s4, 16(sp)
	sd    s5, 8(sp)
	sd    s6, 0(sp)
	.cfi_offset 9, -24
	.cfi_offset 18, -32
	.cfi_offset 19, -40
	.cfi_offset 20, -48
	.cfi_offset 21, -56
	.cfi_offset 22, -64
	li    s2, -286330880
	li    s1, 286330880
	li    s4, -1431654400
	li    s3, 1431654400
	li    s6, -858992640
	li    s5, 858992640
	sd    s0, 48(sp)
	sd    ra, 56(sp)
	.cfi_offset 8, -16
	.cfi_offset 1, -8
	addiw s2, s2, -273
	addi  s1, s1, 273
	addiw s4, s4, -1365
	addi  s3, s3, 1365
	addiw s6, s6, -819
	addi  s5, s5, 819
	li    s0, 1
	j     .L6

.L2:
	mulw a5, s0, s4
	bleu a5, s3, .L10
	mulw a5, s0, s6
	lla  a0, .LC2
	mv   a1, s0
	bgtu a5, s5, .L5
	call puts@plt

.L3:
	addiw s0, s0, 1
	li    a5, 101
	beq   s0, a5, .L11

.L6:
	mulw  a5, s0, s2
	bgtu  a5, s1, .L2
	lla   a0, .LC0
	call  puts@plt
	addiw s0, s0, 1
	li    a5, 101
	bne   s0, a5, .L6

.L11:
	ld   ra, 56(sp)
	.cfi_remember_state
	.cfi_restore 1
	ld   s0, 48(sp)
	.cfi_restore 8
	ld   s1, 40(sp)
	.cfi_restore 9
	ld   s2, 32(sp)
	.cfi_restore 18
	ld   s3, 24(sp)
	.cfi_restore 19
	ld   s4, 16(sp)
	.cfi_restore 20
	ld   s5, 8(sp)
	.cfi_restore 21
	ld   s6, 0(sp)
	.cfi_restore 22
	li   a0, 0
	addi sp, sp, 64
	.cfi_def_cfa_offset 0
	jr   ra

.L5:
	.cfi_restore_state
	lla  a0, .LC3
	call printf@plt
	j    .L3

.L10:
	lla  a0, .LC1
	call puts@plt
	j    .L3
	.cfi_endproc

.LFE11:
	.size    main, .-main
	.ident   "GCC: (GNU) 15.1.0"
	.section .note.GNU-stack, "", @progbits
