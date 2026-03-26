.file      "main0.S"
.option    pic
.attribute arch, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_zicsr2p0_zifencei2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zca1p0_zcd1p0"
.attribute unaligned_access, 0
.attribute stack_align, 16
.text
.section   .rodata
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
.LFB0:
	.cfi_startproc
	addi sp, sp, -32
	.cfi_def_cfa_offset 32
	sd   ra, 24(sp)
	sd   s0, 16(sp)
	.cfi_offset 1, -8
	.cfi_offset 8, -16
	addi s0, sp, 32
	.cfi_def_cfa 8, 0
	li   a5, 1
	sw   a5, -20(s0)
	j    .L2

.L7:
	lw     a5, -20(s0)
	sext.w a3, a5
	li     a4, -2004316160
	addi   a4, a4, -1911
	mul    a4, a3, a4
	srli   a4, a4, 32
	addw   a4, a5, a4
	sraiw  a4, a4, 3
	mv     a3, a4
	sraiw  a4, a5, 31
	subw   a4, a3, a4
	mv     a3, a4
	mv     a4, a3
	slliw  a4, a4, 4
	subw   a4, a4, a3
	subw   a5, a5, a4
	sext.w a5, a5
	bne    a5, zero, .L3
	lla    a0, .LC0
	call   puts@plt
	j      .L4

.L3:
	lw     a5, -20(s0)
	sext.w a3, a5
	li     a4, 1431654400
	addi   a4, a4, 1366
	mul    a4, a3, a4
	srli   a4, a4, 32
	sraiw  a3, a5, 31
	subw   a4, a4, a3
	mv     a3, a4
	mv     a4, a3
	slliw  a4, a4, 1
	addw   a4, a4, a3
	subw   a5, a5, a4
	sext.w a5, a5
	bne    a5, zero, .L5
	lla    a0, .LC1
	call   puts@plt
	j      .L4

.L5:
	lw     a5, -20(s0)
	sext.w a3, a5
	li     a4, 1717985280
	addi   a4, a4, 1639
	mul    a4, a3, a4
	srli   a4, a4, 32
	sraiw  a4, a4, 1
	mv     a3, a4
	sraiw  a4, a5, 31
	subw   a4, a3, a4
	mv     a3, a4
	mv     a4, a3
	slliw  a4, a4, 2
	addw   a4, a4, a3
	subw   a5, a5, a4
	sext.w a5, a5
	bne    a5, zero, .L6
	lla    a0, .LC2
	call   puts@plt
	j      .L4

.L6:
	lw   a5, -20(s0)
	mv   a1, a5
	lla  a0, .LC3
	call printf@plt

.L4:
	lw    a5, -20(s0)
	addiw a5, a5, 1
	sw    a5, -20(s0)

.L2:
	lw     a5, -20(s0)
	sext.w a4, a5
	li     a5, 100
	ble    a4, a5, .L7
	li     a5, 0
	mv     a0, a5
	ld     ra, 24(sp)
	.cfi_restore 1
	ld     s0, 16(sp)
	.cfi_restore 8
	.cfi_def_cfa 2, 32
	addi   sp, sp, 32
	.cfi_def_cfa_offset 0
	jr     ra
	.cfi_endproc

.LFE0:
	.size    main, .-main
	.ident   "GCC: (GNU) 15.1.0"
	.section .note.GNU-stack, "", @progbits
