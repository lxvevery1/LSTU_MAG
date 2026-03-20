.data

fizz:
	.string "Fizz"

buzz:
	.string "Buzz"

full:
	.string "FizzBuzz"

line:
	.string "\n"

	.text
	.globl main

main:
	li s1, 100

	addi s0, s0, 1
	bgt  s0, s1, end

# Блок проверки делимости
li   t0, 15
rem  t1, s0, t0
beqz t1, printfull

li   t0, 3
rem  t1, s0, t0
beqz t1, printfizz

li   t0, 5
rem  t1, s0, t0
beqz t1, printbuzz

# Печать числа
li a7, 1
mv a0, s0
ecall
li a7, 4
la a0, line
ecall
j  main

printfull:
	li a7, 4
	la a0, full
	ecall
	li a7, 4
	la a0, line
	ecall
	j  main

printfizz:
	li a7, 4
	la a0, fizz
	ecall
	li a7, 4
	la a0, line
	ecall
	j  main

printbuzz:
	li a7, 4
	la a0, buzz
	ecall
	li a7, 4
	la a0, line
	ecall
	j  main

end:
	li a7, 10
	ecall
