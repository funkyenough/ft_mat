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
	double max = fabs(m->data[k][k]);
	int maxi = k;
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