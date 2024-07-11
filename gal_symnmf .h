#ifndef SYMNMF_H
#define SYMNMF_H

// Matrix creation functions
double **create_similarity_matrix(int num_points, int num_features, double **data_points);
double **create_diagonal_matrix(int num_points, int num_features, double **data_points);
double **normalize_similarity_matrix(int num_points, int num_features, double **data_points);

// SYMNMF algorithm
double **calculate_symnmf(int k, int num_points, double **norm_matrix, double **H);

// Utility functions
void print_matrix(double **matrix, int rows, int cols);
void free_matrix(double **matrix, int rows);

// Additional utility functions (if needed)
double **init_matrix(int rows, int cols);
double euclidean_distance(const double *vec1, const double *vec2, int size);
double vector_sum(const double *vec, int size);
double **matrix_multiply(double **mat1, double **mat2, int rows1, int cols1, int cols2);
double **transpose_matrix(double **matrix, int rows, int cols);
void copy_matrix(double **dest, double **src, int rows, int cols);

#endif // SYMNMF_H