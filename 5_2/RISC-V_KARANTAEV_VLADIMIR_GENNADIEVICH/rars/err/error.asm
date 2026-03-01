.text
.globl   _start

_start:
	addi sp, sp, -16    # выделить 16 байт на стеке
	li   t0, 0x11223344    # данные
	li   t1, 4
	
	sw   t0, 0(sp)
	
	li   a7, 93    # exit
	li   a0, 0
	ecall
