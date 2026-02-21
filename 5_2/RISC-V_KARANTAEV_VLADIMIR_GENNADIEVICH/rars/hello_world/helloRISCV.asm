# calculator
.globl	main
.data
	# 0 - +
	# 1 - -
	# 2 - *
	# 3 - /
	oper_sum: .byte 0
	
	oper: 	.byte 1
	a:   	.word 3
	b:	.word 1
	err_msg: .string "\nError: invalid oper!"
	max_oper:.byte 3
	
	# a + b = 3 + 2

.text 

	main:
		# записываем операцию в регистр
		la t0, oper
		lb s2, 0(t0)
	
		# a
		la t0, a 
		lw s3, 0(t0)
	
		# b
		la t0, b
		lw s4, 0(t0)
		
		# валидация
		# if (s2 >= 0 && s2 <= 3)
		blt s2, zero, invalid
		
		# проверка на больше
		la t0, max_oper
		lb s5, 0(t0)
		
		bgt s2, s5, invalid
		
		la t0, oper_sum
		lb s5, 0(t0)
		
		beq s2, s5, sum
		
		addi s5, s5, 1
		beq s2, s5, sub
		
		addi s5, s5, 1
		beq s2, s5, mul
		
		addi s5, s5, 1
		beq s2, s5, div
		
		j end
		

		li a7, 10
		ecall

	
	
	sum:    
		# s11 - это результат
		add s11, s3, s4
		j end
		
	sub:    
		# s11 - это результат
		sub s11, s3, s4
		j end
		
	mul:    
		# s11 - это результат
		mul s11, s3, s4
		j end
	
	div:    
		beq s4, zero, invalid
			
		# s11 - это результат
		div s11, s3, s4
		j end
		
				
	invalid: 
		li a7, 4
		la a0, err_msg
		
		ecall
	
	end:
		# конец