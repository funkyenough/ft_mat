NAME = ft_mat

CC = cc
CFLAGS = -g -fsanitize=address

SRCS = ft_mat.c
OBJS = $(SRCS:.c=.o)

all: $(NAME)
	./$(NAME)

$(NAME): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -lm -o $(NAME)

%.o: %.c ft_mat.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS)

fclean: clean
	rm -f $(NAME)

re: fclean all

.PHONY: all clean fclean re
