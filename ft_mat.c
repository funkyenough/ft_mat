#include "ft_mat.h"

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

mat	*mat_rnd(unsigned int num_rows, unsigned int num_cols, double min,
		double max)
{
	mat	*m;

	m = mat_new(num_rows, num_cols);
	for (unsigned int i = 0; i < num_rows; i++)
	{
		for (unsigned int j = 0; j < num_cols; j++)
			m->data[i][j] = _rand_internal(min, max);
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

mat	*mat_row_addrow_r(mat *m, unsigned int to, unsigned int from,
		double multiplier)
{
	if (to >= m->num_rows || from >= m->num_rows)
		return (NULL);
	for (unsigned int i = 0; i < m->num_cols; i++)
		m->data[to][i] += m->data[from][i] * multiplier;
	return (m);
}

mat	*mat_row_addrow(mat *m, unsigned int to, unsigned int from,
		double multiplier)
{
	mat	*r;

	r = mat_cpy(m);
	if (mat_row_addrow_r(m, to, from, multiplier) == NULL)
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

int	mat_add_r(mat *m1, mat *m2)
{
	if (!mat_eqdim(m1, m2))
		return (0);
	for (unsigned int i = 0; i < m1->num_rows; i++)
	{
		for (unsigned int j = 0; j < m1->num_cols; j++)
			m1->data[i][j] += m2->data[i][j];
	}
	return (1);
}

mat	*mat_add(mat *m1, mat *m2)
{
	mat	*r;

	r = mat_cpy(m1);
	if (!mat_add_r(r, m2))
	{
		mat_free(r);
		return (NULL);
	}
	return (r);
}

int	mat_sub_r(mat *m1, mat *m2)
{
	if (!mat_eqdim(m1, m2))
		return (0);
	for (unsigned int i = 0; i < m1->num_rows; i++)
	{
		for (unsigned int j = 0; j < m1->num_cols; j++)
			m1->data[i][j] -= m2->data[i][j];
	}
	return (1);
}

mat	*mat_sub(mat *m1, mat *m2)
{
	mat	*r;

	r = mat_cpy(m1);
	if (!mat_sub_r(r, m2))
	{
		mat_free(r);
		return (NULL);
	}
	return (r);
}

mat	*mat_dot(mat *m1, mat *m2)
{
	mat	*r;

	if (m1->num_cols != m2->num_rows)
		return (NULL);
	r = mat_new(m1->num_rows, m2->num_cols);
	for (unsigned int i = 0; i < r->num_rows; i++)
	{
		for (unsigned int j = 0; j < r->num_cols; j++)
		{
			for (unsigned int k = 0; k < m1->num_cols; k++)
				r->data[i][j] += m1->data[i][k] * m2->data[k][j];
		}
	}
	return (r);
}

// Reduced Echelon Form (REF) satisfies the following conditions:
// 1. All rows having only zero entries are at the bottom
// 2. Leading entry of every non-zero row (pivot) is to the right of the pivot of every row above.
mat	*mat_ref(mat *m)
{
	mat	*r;

	int i, j, k, pivot;
	r = mat_cpy(m);
	i = 0, j = 0;
	while (i < r->num_rows && j < r->num_cols)
	{
		// Find the pivot where row i col j is non zero
		pivot = _mat_pivot_idx(r, i, j);
		if (pivot == -1)
		{
			j++;
			continue ;
		}
		if (pivot != i)
			mat_row_swap_r(r, pivot, i);
		mat_row_mult_r(r, i, 1 / r->data[i][j]);
		for (k = i + 1; k < r->num_rows; k++)
		{
			if (fabs(r->data[k][j]) > MIN_COEF)
				// Make any value on the same column as the pivot zero
				mat_row_addrow_r(r, k, i, -(r->data[k][j]));
		}
		i++;
		j++;
	}
	return (r);
}

// Also called Gauss-Jordan Elimination
// Built on top of REF, and satisfy extra condition:
// a given pivot is the only non zero entry in the column
mat	*mat_rref(mat *m)
{
	mat	*r;

	r = mat_ref(m); // get a REF matrix first
	for (int i = r->num_rows - 1; i >= 0; i--)
	{
		for (int j = 0; j < r->num_cols; j++)
		{
			// find the bottom pivot
			if (fabs(r->data[i][j]) > MIN_COEF)
			{
				// eliminate pivot column of every row above pivot row
				for (int k = i - 1; k >= 0; k--)
					mat_row_addrow_r(r, k, i, -(r->data[k][j]));
				break ;
			}
		}
	}
	return (r);
}

// Kind of surprised that Andrei is doing a shallow copy here
// Ah, because LUP are constructed specifically for LUP Decomposition
lup	*lup_new(mat *L, mat *U, mat *P, unsigned int num_perm)
{
	lup	*r;

	r = malloc(sizeof(lup));
	if (r == NULL)
		return (NULL);
	r->L = L;
	r->U = U;
	r->P = P;
	r->num_perm = num_perm;
	return (r);
}

void	lup_free(lup *lu)
{
	mat_free(lu->L);
	mat_free(lu->U);
	mat_free(lu->P);
	free(lu);
}

// Factorize a square matrix into upper and lower matrices
// ! Purpose yet unknown to me...
// Assuming that the matrix is not ordered in such a way that it is sorted according to pivot,
// a separate permutation matrix P is necessary to record what permutation is performed to it.
lup	*lup_solve(mat *m)
{
	double	mult;

	int pivot, num_perm = 0, i = 0, j = 0;
	mat *L, *U, *P;
	if (!m->is_square) // LUP cannot be applied on non square matrices
		return (NULL);
	L = mat_new(m->num_rows, m->num_cols); // Lower, initilize to 0
	U = mat_cpy(m);                        // Upper, initialize to copy of m
	P = mat_eye(m->num_rows);              // Perm matrix init to identity
	for (j = 0; j < m->num_cols; j++)
	{
		// find the maximum element in a given column
		pivot = _mat_abs_max(U, j);
		if (fabs(U->data[pivot][j]) < MIN_COEF)
			return (NULL);
		if (pivot != j)
		{
			mat_row_swap_r(L, pivot, j);
			mat_row_swap_r(U, pivot, j);
			mat_row_swap_r(P, pivot, j);
			num_perm++;
		}
		// Now that matrix is sorted, apply factoriation
		for (i = j + 1; i < m->num_rows; i++)
		{
			// Find the value to write to the Lower Matrix
			mult = U->data[i][j] / U->data[j][j];
			mat_row_addrow_r(U, i, j, -mult);
			L->data[i][j] = mult;
		}
	}
	// This is to avoid row swap messing up the diagonal
	mat_diag_set(L, 1.0);
	return (lup_new(L, U, P, num_perm));
}

int	main(void)
{
	srand(time(NULL));
	// mat *r = mat_rnd(5, 3, 0.0, 10.0);
	// mat_print(r);

	// mat_print(mat_col_get(r, 2));
	// mat_print(mat_row_get(r, 2));
	// mat_print(mat_diag_set(r, 5.0));
	// mat_print(mat_row_mult(r, 2, 10.0));
	// mat_print(mat_col_mult(r, 2, 10.0));
	// mat_print(mat_col_rem(r, 0));
	// mat_print(mat_row_rem(r, 0));
	// mat_print(mat_row_swap(r, 0, 1));
	// mat_print(mat_col_swap(r, 0, 1));

	mat *a = mat_rnd(5, 5, 0.0, 10.0);
	// mat *b = mat_rnd(5, 5, 0.0, 10.0);
	// mat_print(a);
	// mat_print(b);
	// mat_print(mat_add(a, b));
	// mat_print(mat_sub(a, b));
	// mat_print(mat_ref(a));
	// mat_print(mat_rref(a));

	lup *lu = lup_solve(a);
	if (!lu)
	{
		printf("LUP decomposition failed\n");
		return (1);
	}
	mat_print_name("P", lu->P);
	mat_print_name("A", a);
	mat_print_name("L", lu->L);
	mat_print_name("U", lu->U);
	mat_print_name("PxA", mat_dot(lu->P, a));
	mat_print_name("LxU", mat_dot(lu->L, lu->U));

	return (0);
}