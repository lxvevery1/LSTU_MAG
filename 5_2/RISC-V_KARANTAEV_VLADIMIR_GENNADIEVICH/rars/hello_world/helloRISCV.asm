# calculator
# oper задвать через ecall (input)

# идентификация ошибок
# степень, квадратный корень, факториал
# вывод результата
# циклы - нужно ли сделать операцию другую еще раз

# типы инструкций i s u (как узнать в рарс)
# где регистр источник, где регистр назначения

.data

# 0 - +
# 1 - -
# 2 - *
# 3 - /
# 4 - ^
# 5 - sqrt - без b
# 6 - !  3! = 3 * 2 * 1 - без b
# 9 - закончить
oper_sum: .byte 0

# oper: 	.byte 6 # s2
# a:   	.word 3 # s3
# b:	.word 0 # s4
err_msg:  .string "\nError: invalid index oper (0 <= id <= 5)!"
err_msg_div_0:  .string "\nError: division by 0!"
msg_enter_a:  .string "\nEnter a:"
msg_enter_b:  .string "\nEnter b:"
msg_enter_oper: .string "\nEnter operation (0 - '+', 1 - '-', 2 - '*', 3 - '/', 4 - '^', 5 - 'sqrt', 6 - '!', 9 - exit):"

max_oper: .byte 6
bez_b:   .byte 4

exit:
	.byte 9

# a + b = 3 + 2

.text

	main:
		# ввод oper
		li a7, 4
		la a0, msg_enter_oper
		ecall
		
		li a7, 5
		ecall
		mv s2, a0
		
		# валидация
		# проверка на выход
		la  t0, exit
		lb  s9, 0(t0)
		beq s2, s9, end
		
		# if (s2 >= 0 && s2 <= 3)
		blt s2, zero, invalid
		
		# проверка на больше
		la t0, max_oper
		lb s9, 0(t0)
		
		bgt s2, s9, invalid
		
		# ввод a
		li a7, 4
		la a0, msg_enter_a
		ecall
		
		li a7, 5
		ecall
		mv s3, a0
		
		la t0, bez_b
		lb s8, 0(t0)
		
		bge s2, s8, skip_b
		
		# ввод b
		li a7, 4
		la a0, msg_enter_b
		ecall
		
		li a7, 5
		ecall
		mv s4, a0

	skip_b:

		# зануляем результат
		li s11, 0
		li t3, 2
		
		la t0, oper_sum
		lb s5, 0(t0)
		
		beq s2, s5, sum
		
		addi s5, s5, 1
		beq  s2, s5, sub
		
		addi s5, s5, 1
		beq  s2, s5, mul
		
		addi s5, s5, 1
		beq  s2, s5, div
		
		# s11 - перед началом нового цикла ставим 0
		addi s5, s5, 1
		addi s11, s11, 1
		beq  s2, s5, grade
		
		addi s5, s5, 1
		beq  s2, s5, sqrt
		
		addi s5, s5, 1
		beq  s2, s5, factorial
		
		j end
		
		li a7, 10
		ecall

	sum:
		# s11 - это результат
		add s11, s3, s4
		j   print_result

	sub:
		# s11 - это результат
		sub s11, s3, s4
		j   print_result

	mul:
		# s11 - это результат
		mul s11, s3, s4
		j   print_result

	div:
		beq s4, zero, invalid_div_zero

		# s11 - это результат
		div s11, s3, s4
		j   print_result

	grade:
		# s11 - это результат
		# s6 - счетчик
		# s4 - b - значение степени
		beq s6, s4, print_result
		
		mul s11, s3, s11
		
		addi s6, s6, 1
		j    grade

	sqrt:
		fcvt.s.w f0, s3

		# s11 - это результат
		fsqrt.s fa0, f0
		
		# 2 - float
		li a7, 2
		
		# свойский вывод
		ecall
		j main

	factorial:
		# if s3 == 0
		beq s3, zero, print_result
		mul s11, s3, s11
		
		#
		addi s3, s3, -1
		
		j factorial

	invalid:
		li a7, 4
		la a0, err_msg

		ecall
		j main

	invalid_div_zero:
		li a7, 4
		la a0, err_msg_div_0
	
		ecall
		j main

	print_result:
		li a7, 1
		mv a0, s11
	
		ecall
		j main

	end:
		# конец
