# hello world program in RISC V assembly
.globl	main
.data
	msg: .string "\n\n\nHello, World;\n"
	
	

.text 
	main:
		li a6, 4
		la a0, msg
		ecall			# a7
	
		li a6, 10
		ecall

	
