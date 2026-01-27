#ifndef FT_MAT_H
# define FT_MAT_H

# include <math.h>
# include <stdio.h>
# include <stdlib.h>
# include <string.h>
# include <time.h>

# define RAND_MAX 0x7fffffff
# define MIN_COEF 0.000000000000001
# define COLOR_RESET "\033[0m"
# define COLOR_ZERO "\033[90m"
# define COLOR_ONE "\033[32m"

typedef struct mat_s
{
	unsigned int	num_rows;
	unsigned int	num_cols;
	double			**data;
	int				is_square;
}					mat;

typedef struct mat_lup_s
{
	mat				*L;
	mat				*U;
	mat				*P;
	unsigned int	num_perm;
}					mat_lup;

mat					*mat_new(unsigned int num_rows, unsigned int num_cols);
void				mat_free(mat *matrix);
int					mat_eq(mat *m1, mat *m2, double tolerance);

// UTILS
double				_rand_internal(double min, double max);
int					_mat_pivot_idx(mat *m, unsigned int row, unsigned int col);
int					_mat_abs_max(mat *m, unsigned int k);

void				mat_printf(mat *m, const char *d_fmt);
void				mat_print(mat *m);
void				mat_print_name(char *name, mat *m);
void				mat_print_eq(char *a, char *b, mat *ma, mat *mb,
						double tolerance);

#endif