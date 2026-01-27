#include "ft_mat.h"

double	_rand_internal(double min, double max)
{
	double	d;

	d = (double)rand() / ((double)RAND_MAX + 1);
	return (min + d * (max - min));
}

int	_mat_pivot_idx(mat *m, unsigned int row, unsigned int col)
{
	for (unsigned int i = row; i < m->num_rows; i++)
	{
		if (fabs(m->data[i][col]) > MIN_COEF)
			return (i);
	}
	return (-1);
}

// Andrei seems to have made some mistakes here?
int	_mat_abs_max(mat *m, unsigned int k)
{
	double	max;
	int		maxi;

	max = fabs(m->data[k][k]);
	maxi = k;
	for (int i = k + 1; i < m->num_rows; i++)
	{
		if (fabs(m->data[i][k]) > max)
		{
			max = fabs(m->data[i][k]);
			maxi = i;
		}
	}
	return (maxi);
}

// Highlight 1.0 values
void	mat_printf(mat *m, const char *d_fmt)
{
	double	val;

	fprintf(stdout, "\n");
	for (unsigned int i = 0; i < m->num_rows; i++)
	{
		for (unsigned int j = 0; j < m->num_cols; j++)
		{
			val = fabs(m->data[i][j]);
			if (fabs(val) < MIN_COEF)
				fprintf(stdout, COLOR_ZERO);
			else if (fabs(val - 1.0) < MIN_COEF)
				fprintf(stdout, COLOR_ONE);
			fprintf(stdout, d_fmt, val < MIN_COEF ? 0.0 : m->data[i][j]);
			fprintf(stdout, COLOR_RESET);
		}
		fprintf(stdout, "\n");
	}
	fprintf(stdout, "\n");
}

void	mat_print(mat *m)
{
	mat_printf(m, "%.1lf\t");
}

void	mat_print_name(char *name, mat *m)
{
	printf("%s", name);
	mat_printf(m, "%.1lf\t");
}

void	mat_print_eq(char *a, char *b, mat *ma, mat *mb, double tolerance)
{
	int eq;

	eq = mat_eq(ma, mb, tolerance);
	if (!eq)
		printf("%s != %s\n", a, b);
	else
		printf("%s == %s\n", a, b);
}