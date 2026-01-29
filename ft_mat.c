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

// General case is O(n^3)
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

double	mat_trace(mat *m)
{
	double	r;

	r = 0.0;
	if (m->num_rows != m->num_cols)
	{
		printf("mat_trace: Not square\n");
		return (r);
	}
	for (int i = 0; i < m->num_rows; i++)
		r += m->data[i][i];
	return (r);
}

mat	*mat_transpose(mat *m)
{
	mat	*r;

	r = mat_new(m->num_cols, m->num_rows);
	for (int i = 0; i < m->num_rows; i++)
	{
		for (int j = 0; j < m->num_cols; j++)
			r->data[j][i] = m->data[i][j];
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

unsigned int	mat_rank(mat *m)
{
	mat	*rref;

	rref = mat_rref(m);
	for (int i = rref->num_rows - 1; i >= 0; i--)
	{
		for (int j = rref->num_cols - 1; j >= 0; j--)
		{
			if (fabs(rref->data[i][j]) > MIN_COEF)
			{
				mat_free(rref);
				return (i + 1);
			}
		}
	}
	mat_free(rref);
	return (0);
}

// Kind of surprised that Andrei is doing a shallow copy here
// Ah, because LUP are constructed specifically for LUP Decomposition
mat_lup	*mat_lup_new(mat *L, mat *U, mat *P, unsigned int num_perm)
{
	mat_lup	*r;

	r = malloc(sizeof(mat_lup));
	if (r == NULL)
		return (NULL);
	r->L = L;
	r->U = U;
	r->P = P;
	r->num_perm = num_perm;
	return (r);
}

void	mat_lup_free(mat_lup *lup)
{
	mat_free(lup->L);
	mat_free(lup->U);
	mat_free(lup->P);
	free(lup);
}

// Factorize a square matrix into upper and lower matrices
// Assuming that the matrix is not ordered in such a way that it is sorted according to pivot,
// a separate permutation matrix P is necessary to record what permutation is performed to it.
mat_lup	*mat_lup_solve(mat *m)
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
	return (mat_lup_new(L, U, P, num_perm));
}

// Forward substitution
// Solves L* x = b where L is a lower triangular matrix
// Starting from the top row (which has only one unknown)
// we solve each variable and substitute it into the rows below
mat	*ls_solve_fwd(mat *L, mat *b)
{
	double	tmp;
	mat		*x;

	int i, j;
	x = mat_new(L->num_rows, 1);
	for (i = 0; i < L->num_rows; i++)
	{
		tmp = b->data[i][0];
		for (j = 0; j < i; j++)
			// x->data[j][0] we have already solved
			// L->data[i][j] is the coefficient (multiplier, the a in aX)
			// We remove it from tmp one by one until only the var in question is left
			tmp -= L->data[i][j] * x->data[j][0];
		x->data[i][0] = tmp / L->data[i][i];
	}
	return (x);
}

// Backward Substitution
// Similar to Forward substitution
// except we start from the bottom row last column
mat	*ls_solve_bck(mat *U, mat *b)
{
	double	tmp;
	mat		*x;

	int i, j;
	x = mat_new(U->num_rows, 1);
	for (i = U->num_rows - 1; i >= 0; i--)
	{
		tmp = b->data[i][0];
		for (j = i + 1; j < U->num_cols; j++)
			tmp -= U->data[i][j] * x->data[j][0];
		x->data[i][0] = tmp / U->data[i][i];
	}
	return (x);
}

// Given a matrix a and result b, where a * x = b
// We rewrite
// P * a * x = P * b
// L * U * x = P * b
// L * y = P * b (sub y = U * x)
// finally solve x
mat	*ls_solve(mat_lup *lup, mat *b)
{
	mat *x, *y, *Pb;
	if (lup->L->num_rows != b->num_rows || b->num_cols != 1)
		return (NULL);
	Pb = mat_dot(lup->P, b);
	y = ls_solve_fwd(lup->L, Pb);
	x = ls_solve_bck(lup->U, y);
	mat_free(Pb);
	mat_free(y);
	return (x);
}

// Given a, compute Ia s.t. a * Ia == Ia * a = I
// Compute each col of Ia by solving the following
// Ia[col] = ls_solve(a, I[col])
// I can now understand why this takes so much time..
// The O(n) of this current algorithm is
mat	*mat_inv(mat_lup *lup)
{
	unsigned int	dim;

	int i, j;
	dim = lup->L->num_cols;
	mat *inv, *invx, *I, *Ix;
	inv = mat_sqr(dim);
	I = mat_eye(dim);
	for (j = 0; j < dim; j++)
	{
		Ix = mat_col_get(I, j);
		invx = ls_solve(lup, Ix);
		for (i = 0; i < invx->num_rows; i++)
			inv->data[i][j] = invx->data[i][0];
		mat_free(invx);
		mat_free(Ix);
	}
	mat_free(I);
	return (inv);
}

// Given square matrix A, compute det(A)
// So the straightforward way of computing determinant is quite messy
// But LUP actually gives us an easy way to compute it by leveraging
// a few properties
// 1. triangular matrix det(triangle) = product of diagonal
// 2. det(A * B) = det(A) * det(B)
// 3. det(cA) = c^n * det(A) where n is the number of permutations
// a => P * a = L * U
// det(P) * det(a) = det(L) * det(U)
// det(P) = (-1)^n, btw I don't know why the base is -1
// det(L) = product of diagonal = 1
// det(U) = product of diagonal
// det(a) = det(L) * det(U) / det(P)
// = 1 * det(U) / (-1)^n
// Sidenote: given our design of lup_solve, we will never have det(A) = 0
// because lup would have been null when no pivot exists
double	mat_det(mat_lup *lup)
{
	int		i;
	double	r;
	int		sign;

	sign = (lup->num_perm % 2 == 0) ? 1 : -1;
	r = 1.0;
	for (i = 0; i < lup->U->num_rows; i++)
		r *= lup->U->data[i][i];
	return (r * sign);
}

// Given matrix m1 and m2, compute dot product of m1[m1col] * m2[m2col]
double	vec_dot(mat *m1, mat *m2, unsigned int m1col, unsigned int m2col)
{
	double	dot;

	if (m1->num_rows != m2->num_rows || m1col >= m1->num_cols
		|| m2col >= m2->num_cols)
		printf("Vector Dot Product Failed\n");
	dot = 0.0;
	for (int i = 0; i < m1->num_rows; i++)
		dot += m1->data[i][m1col] * m2->data[i][m2col];
	return (dot);
}

// l_2 Euclidian Norm = sqrt(a_1^2 + a_2^2 + ... + a_n^2)
double	mat_col_l2norm(mat *m, unsigned int col)
{
	double	n;

	n = 0.0;
	for (int i = 0; i < m->num_rows; i++)
		n += m->data[i][col] * m->data[i][col];
	return (sqrt(n));
}

// Compute l2norm for all unit vectors
mat	*mat_l2norm(mat *m)
{
	mat	*r;

	r = mat_new(1, m->num_cols);
	for (int j = 0; j < m->num_cols; j++)
		r->data[0][j] = mat_col_l2norm(m, j);
	return (r);
}

int	mat_normalize_r(mat *m)
{
	mat	*l2norm;

	l2norm = mat_l2norm(m);
	for (int j = 0; j < m->num_cols; j++)
	{
		if (l2norm->data[0][j] < MIN_COEF)
		{
			mat_free(l2norm);
			return (0);
		}
		mat_col_mult_r(m, j, 1 / l2norm->data[0][j]);
	}
	mat_free(l2norm);
	return (1);
}

mat_qr	*mat_qr_new(void)
{
	mat_qr	*qr;

	qr = malloc(sizeof(mat_qr));
	return (qr);
}

void	mat_qr_free(mat_qr *qr)
{
	mat_free(qr->Q);
	mat_free(qr->R);
	free(qr);
}

mat_qr	*mat_qr_solve(mat *m)
{
	mat_qr	*qr;
	mat		*Q;
	mat		*R;
	double	dot;
	double	norm;

	Q = mat_cpy(m);
	R = mat_new(m->num_rows, m->num_cols);
	for (int j = 0; j < m->num_cols; j++)
	{
		for (int k = 0; k < j; k++)
		{
			// Compute the projection of vector k onto vector j
			dot = vec_dot(Q, Q, j, k);
			// Records how much column j points in the direction of column k
			R->data[k][j] = dot;
			// Subtract such projection from the vector to make them orthogonal to each other
			for (int i = 0; i < m->num_rows; i++)
				Q->data[i][j] -= dot * Q->data[i][k];
		}
		norm = mat_col_l2norm(Q, j);
		if (norm < MIN_COEF)
		{
			mat_free(Q);
			mat_free(R);
			return (NULL);
		}
		R->data[j][j] = norm;
		mat_col_mult_r(Q, j, 1.0 / norm);
	}
	qr = mat_qr_new();
	qr->Q = Q;
	qr->R = R;
	return (qr);
}

mat_eig	*mat_eig_new(mat *values, mat *vectors)
{
	mat_eig	*eig;

	eig = malloc(sizeof(mat_eig));
	eig->values = values;
	eig->vectors = vectors;
	return (eig);
}

void	mat_eig_free(mat_eig *eig)
{
	mat_free(eig->values);
	mat_free(eig->vectors);
	free(eig);
}

double	mat_off_diag_sum(mat *m)
{
	double	sum;

	sum = 0.0;
	for (int i = 0; i < m->num_rows; i++)
		for (int j = 0; j < i; j++)
			sum += fabs(m->data[i][j]);
	return (sum);
}
// Given A ∈ n * n, x ∈ n, l ∈ R
// Find l st Av = lv, where v is eigen vector, and l is eigen value
//
// Intuition:
// Consider A as a linear transformation from vector space V -> V
// eigen vector v is not rotated under transformation,
// but only stretched or squished by a scalar l.
//
// Mathematically:
// Av = lv = l * I * v = (lI)v (I as identity matrix)
// Av - (lI)v = (A - lI)v = 0
// Assuming that the eigen vector v is non-zero,
// determinant of (A - lI) must be zero.
// Therefore, we compute the  det(A - lI) = 0, where we subract A[i][i] by l.
// We compute the eigen value by computing the roots of the polynomial of l.
//
// Computationally, finding roots of polynomial is very expensive
// we leverage 2 primitives
// 1. QR Decomposition to compute the inverse of A
//

// Computes eigenvalues using basic QR iteration (no shift).
// Returns NULL if matrix has complex eigenvalues (detected by divergence).
mat_eig	*mat_eig_solve(mat *m)
{
	mat		*values;
	mat		*vectors;
	mat		*A;
	mat_qr	*qr;
	mat		*na;
	int		dim;
	double	prev_sum;
	double	curr_sum;
	mat_eig	*eig;

	dim = m->num_rows;
	vectors = mat_eye(dim);
	A = mat_cpy(m);
	prev_sum = mat_off_diag_sum(A);
	while (prev_sum > MIN_COEF)
	{
		qr = mat_qr_solve(A);
		na = mat_dot(qr->R, qr->Q);
		mat_free(A);
		A = na;
		vectors = mat_dot(vectors, qr->Q);
		mat_qr_free(qr);
		curr_sum = mat_off_diag_sum(A);
		if (curr_sum >= prev_sum)
		{
			printf("mat_eigenvalues: diverging (complex eigenvalues?)\n");
			mat_free(A);
			return (NULL);
		}
		prev_sum = curr_sum;
	}
	values = mat_new(dim, 1);
	for (int i = 0; i < dim; i++)
		values->data[i][0] = A->data[i][i];
	mat_free(A);
	eig = mat_eig_new(values, vectors);
	return (eig);
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
	// mat_print_name("tr", mat_transpose(r));

	mat *a = mat_rnd(3, 3, 0.0, 10.0);
	// mat *b = mat_rnd(5, 1, 0.0, 10.0);
	mat_print_name("a", a);
	// mat_print(b);
	// mat_print(mat_ref(a));
	// mat_print(mat_rref(a));
	// printf("rank is %d\n", mat_rank(a));

	// mat_lup *lup = mat_lup_solve(a);
	// if (!lup)
	// {
	// 	printf("LUP decomposition failed\n");
	// 	return (1);
	// }
	// mat_print_name("P", lup->P);
	// mat_print_name("A", a);
	// mat_print_name("L", lup->L);
	// mat_print_name("U", lup->U);
	// mat_print_name("PxA", mat_dot(lup->P, a));
	// mat_print_name("LxU", mat_dot(lup->L, lup->U));

	// mat *x = ls_solve(lup, b);
	// mat *Ax = mat_dot(a, x);
	// mat_print_name("Ax", Ax);
	// mat_print_name("b", b);
	// mat_print_eq("Ax", "b", Ax, b, MIN_COEF * 10);

	// mat *Ia = mat_inv(lup);
	// mat *aIa = mat_dot(a, Ia);
	// mat *Iaa = mat_dot(Ia, a);
	// mat *eye = mat_eye(5);

	// mat_print_name("Ia", Ia);
	// mat_print_name("aIa", aIa);
	// mat_print_name("Iaa", Iaa);
	// mat_print_eq("aIa", "I", aIa, eye, MIN_COEF);
	// mat_print_eq("Iaa", "I", Iaa, eye, MIN_COEF);

	// still not seeing the point of this tbh
	// printf("determinant of a is %f\n", mat_det(lup));

	// mat_qr *qr = mat_qr_solve(a);
	// mat_print_name("Q", qr->Q);
	// mat_print_name("R", qr->R);
	// mat *qr_dot = mat_dot(qr->Q, qr->R);
	// mat_print_name("QR", qr_dot);
	// mat_print_eq("Q * R", "a", qr_dot, a, MIN_COEF * 10);

	mat_eig *eig = mat_eig_solve(a);
	if (eig)
	{
		mat_print_name("eigenvalues", eig->values);
		mat_print_name("eigenvectors", eig->vectors);
	}
	return (0);
}