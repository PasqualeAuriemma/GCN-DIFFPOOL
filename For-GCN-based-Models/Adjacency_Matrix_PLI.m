function matrix = Adjacency_Matrix_PLI(matrix_temp1)
newData = matrix_temp1 - mean(matrix_temp1, 1);
pli_matrix = PLI(newData);
Absolute_PLI_matrix = abs(pli_matrix);
Eye_Matrix = eye(64, 64);
Adjacency_Matrix = Absolute_PLI_matrix - Eye_Matrix;
matrix = Adjacency_Matrix;
