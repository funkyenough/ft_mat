#ifndef FT_MAT_H
# define FT_MAT_H

# include <math.h>
# include <stdio.h>
# include <stdlib.h>
# include <string.h>

# define RAND_MAX 0x7fffffff
# define MIN_COEF 0.000000000000001

typedef struct mat_s
{
	unsigned int	num_rows;
	unsigned int	num_cols;
	double			**data;
	int				is_square;
}					mat;

mat					*mat_new(unsigned int num_rows, unsigned int num_cols);
void				mat_free(mat *matrix);

// UTILS
double				rand_internal(double min, double max);
#endif