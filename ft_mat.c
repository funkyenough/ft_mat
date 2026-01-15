#include "ft_mat.h"

// Create a matrix of height row and width col, all elements initialized to 0.0.
mat	*mat_new(unsigned int num_rows, unsigned int num_cols)
{
	mat	*m;

	if (num_rows == 0 || num_cols == 0)
		return (NULL);
	m = calloc(1, sizeof(mat));
	if (!m)
		return (NULL);
	m->num_rows = num_rows;
	m->num_cols = num_cols;
	m->is_square = (num_rows == num_cols ? 1 : 0);
	m->data = calloc(m->num_rows, sizeof(*m->data));
	if (!(m->data))
	{
		free(m);
		return (NULL);
	}
	for (unsigned int i = 0; i < num_rows; i++)
	{
		m->data[i] = calloc(num_cols, sizeof(**m->data));
		if (!(m->data[i]))
		{
			free(m->data);
			free(m);
			return (NULL);
		}
	}
	return (m);
}

void	mat_free(mat *m)
{
	for (unsigned int i = 0; i < m->num_rows; i++)
		free(m->data[i]);
	free(m->data);
	free(m);
}

mat	*mat_cpy(mat *m)
{
	mat	*r;

	r = mat_new(m->num_rows, m->num_cols);
	r->num_rows = m->num_rows;
	r->num_cols = m->num_cols;
	r->is_square = m->is_square;
	for (unsigned int i = 0; i < m->num_rows; i++)
	{
		for (unsigned int j = 0; j < m->num_cols; j++)
			r->data[i][j] = m->data[i][j];
	}
	return (r);
}

double	rand_internal(double min, double max)
{
	double	d;

	d = (double)rand() / ((double)RAND_MAX + 1);
	return (min + d * (max - min));
}

mat	*mat_rnd(unsigned int num_rows, unsigned int num_cols, double min,
		double max)
{
	mat	*m;

	m = mat_new(num_rows, num_cols);
	for (unsigned int i = 0; i < num_rows; i++)
	{
		for (unsigned int j = 0; j < num_cols; j++)
			m->data[i][j] = rand_internal(min, max);
	}
	return (m);
}

mat	*mat_sqr(unsigned int size)
{
	return (mat_new(size, size));
}

mat	*mat_sqr_rnd(unsigned int size, double min, double max)
{
	return (mat_rnd(size, size, min, max));
}

mat	*mat_eye(unsigned int size)
{
	mat	*m;

	m = mat_sqr(size);
	for (unsigned int i = 0; i < m->num_rows; i++)
		m->data[i][i] = 1.0;
	return (m);
}

mat	*mat_fromfile(FILE *f)
{
	unsigned int	num_rows;
	unsigned int	num_cols;
	mat				*m;

	num_rows = 0;
	num_cols = 0;
	fscanf(f, "%d", &num_rows);
	fscanf(f, "%d", &num_cols);
	m = mat_new(num_rows, num_cols);
	for (unsigned int i = 0; i < m->num_rows; i++)
	{
		for (unsigned int j = 0; j < m->num_cols; j++)
			fscanf(f, "%lf", &m->data[i][j]); // here we are
	}
	return (m);
}

int	mat_eqdim(mat *m1, mat *m2)
{
	return (m1->num_rows == m2->num_rows) && (m1->num_cols == m2->num_cols);
}

int	mat_eq(mat *m1, mat *m2, double tolerance)
{
	if (!mat_eqdim(m1, m2))
		return (0);
	for (unsigned int i = 0; i < m1->num_rows; i++)
	{
		for (unsigned int j = 0; j < m1->num_cols; j++)
		{
			if (fabs(m1->data[i][j] - m2->data[i][j]) > tolerance)
				return (0);
		}
	}
	return (1);
}

void	mat_printf(mat *m, const char *d_fmt)
{
	fprintf(stdout, "\n");
	for (unsigned int i = 0; i < m->num_rows; i++)
	{
		for (unsigned int j = 0; j < m->num_cols; j++)
			fprintf(stdout, d_fmt, m->data[i][j]);
		fprintf(stdout, "\n");
	}
	fprintf(stdout, "\n");
}

void	mat_print(mat *m)
{
	mat_printf(m, "%.1lf  ");
}

mat	*mat_col_get(mat *m, unsigned int col)
{
	mat	*r;

	if (col >= m->num_cols)
		return (NULL);
	r = mat_new(m->num_rows, 1);
	for (unsigned int i = 0; i < m->num_rows; i++)
	{
		r->data[i][0] = m->data[i][col];
	}
	return (r);
}

mat	*mat_row_get(mat *m, unsigned int row)
{
	mat	*r;

	if (row >= m->num_rows)
		return (NULL);
	r = mat_new(1, m->num_cols);
	memcpy(r->data[0], m->data[row], m->num_cols * sizeof(*r->data[0]));
	return (r);
}

void	mat_all_set(mat *m, double value)
{
	for (unsigned int i = 0; i < m->num_rows; i++)
	{
		for (unsigned int j = 0; j < m->num_cols; j++)
			m->data[i][j] = value;
	}
}

int	mat_diag_set(mat *m, double value)
{
	if (!m->is_square)
		return (0);
	for (unsigned int i = 0; i < m->num_rows; i++)
		m->data[i][i] = value;
	return (1);
}

int	mat_row_mult_r(mat *m, unsigned int row, double num)
{
	if (row >= m->num_rows)
		return (0);
	for (unsigned int i = 0; i < m->num_cols; i++)
		m->data[row][i] *= num;
	return (1);
}

mat	*mat_row_mult(mat *m, unsigned int row, double num)
{
	mat	*r;

	r = mat_cpy(m);
	if (!mat_row_mult_r(r, row, num))
	{
		mat_free(r);
		return (NULL);
	}
	return (r);
}

int	mat_col_mult_r(mat *m, unsigned int col, double num)
{
	if (col >= m->num_cols)
		return (0);
	for (unsigned int i = 0; i < m->num_rows; i++)
		m->data[i][col] *= num;
	return (1);
}

mat	*mat_col_mult(mat *m, unsigned int col, double num)
{
	mat	*r;

	r = mat_cpy(m);
	if (!mat_col_mult_r(r, col, num))
	{
		mat_free(r);
		return (NULL);
	}
	return (r);
}

void	mat_smult_r(mat *m, double num)
{
	for (unsigned int i = 0; i < m->num_rows; i++)
	{
		for (unsigned int j = 0; j < m->num_cols; j++)
			m->data[i][j] *= num;
	}
}

mat	*mat_smult(mat *m, double num)
{
	mat	*r;

	r = mat_cpy(m);
	mat_smult_r(r, num);
	return (r);
}

mat	*mat_col_rem(mat *m, unsigned int column)
{
	mat	*r;

	if (column >= m->num_cols)
		return (NULL);
	r = mat_new(m->num_rows, m->num_cols - 1);
	for (unsigned int i = 0; i < r->num_rows; i++)
	{
		for (unsigned int j = 0; j < r->num_cols; j++)
		{
			if (j >= column)
				r->data[i][j] = m->data[i][j + 1];
			else
				r->data[i][j] = m->data[i][j];
		}
	}
	return (r);
}

mat	*mat_row_rem(mat *m, unsigned int row)
{
	mat	*r;

	if (row >= m->num_rows)
		return (NULL);
	r = mat_new(m->num_rows - 1, m->num_cols);
	for (unsigned int i = 0; i < r->num_rows; i++)
	{
		if (i >= row)
		{
			for (unsigned int j = 0; j < r->num_cols; j++)
				r->data[i][j] = m->data[i + 1][j];
		}
		else
		{
			for (unsigned int j = 0; j < r->num_cols; j++)
				r->data[i][j] = m->data[i][j];
		}
	}
	return (r);
}

int	mat_row_swap_r(mat *m, unsigned int row1, unsigned int row2)
{
	double	*swp;

	if (row1 >= m->num_rows || row2 >= m->num_rows)
		return (0);
	if (row1 == row2)
		return (1);
	swp = m->data[row1];
	m->data[row1] = m->data[row2];
	m->data[row2] = swp;
	return (1);
}

mat	*mat_row_swap(mat *m, unsigned int row1, unsigned int row2)
{
	mat	*r;

	r = mat_cpy(m);
	if (!mat_row_swap_r(r, row1, row2))
	{
		mat_free(r);
		return (NULL);
	}
	return (r);
}

int	mat_col_swap_r(mat *m, unsigned int col1, unsigned int col2)
{
	double	tmp;

	if (col1 >= m->num_cols || col2 >= m->num_cols)
		return (0);
	if (col1 == col2)
		return (1);
	for (unsigned int i = 0; i < m->num_rows; i++)
	{
		tmp = m->data[i][col1];
		m->data[i][col1] = m->data[i][col2];
		m->data[i][col2] = tmp;
	}
	return (1);
}

mat	*mat_col_swap(mat *m, unsigned int col1, unsigned int col2)
{
	mat	*r;

	r = mat_cpy(m);
	if (!mat_col_swap_r(r, col1, col2))
	{
		mat_free(r);
		return (NULL);
	}
	return (r);
}

// mat	*mat_cath(unsigned int mnum, mat **marr)
// {
// 	unsigned int	num_cols;

// 	if (mnum == 0 || !marr)
// 		return (NULL);
// 	if (mnum == 1)
// 		return (mat_cpy(marr[0]));
// 	num_cols = marr[0]->num_cols;
// 	for (unsigned int i = 1; i < mnum; i++)
// 	{
// 		if (!marr[mnum] || marr[mnum]->num_cols != num_cols)
// 			return (NULL);
// 	}
// }

int	main(void)
{
	mat *m = mat_eye(10);

	mat *n = mat_col_get(m, 2);
	mat_print(n);

	mat *o = mat_row_get(m, 2);
	mat_print(o);

	mat *r = mat_rnd(5, 3, 0.0, 10.0);
	mat_print(r);

	mat_diag_set(r, 5.0);
	mat_print(r);

	mat *p = mat_row_mult(r, 2, 10.0);
	mat_print(p);

	mat *q = mat_col_mult(p, 2, 10.0);
	mat_print(q);

	mat *s = mat_col_rem(q, 0);
	mat_print(s);

	mat *t = mat_row_rem(s, 0);
	mat_print(t);

	mat_row_swap_r(t, 0, 1);
	mat_print(t);

	mat_col_swap_r(t, 0, 1);
	mat_print(t);

	return (0);
}