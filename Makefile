CC = gcc
CFLAGS = -ansi -Wall -Wextra -Werror -pedantic-errors -lm

symnmf: symnmf.c symnmf.h
	$(CC) -o symnmf symnmf.c $(CFLAGS)