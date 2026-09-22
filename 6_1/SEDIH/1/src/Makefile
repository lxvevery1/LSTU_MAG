CC = gcc
CFLAGS = -Wall -Wextra
LDLIBS = -lm
TARGET = tff
SRCS = main.c xsrc.c xtgt.c encoder.c decoder.c output.c
OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDLIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
