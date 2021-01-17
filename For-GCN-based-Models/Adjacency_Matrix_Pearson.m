function matrix = Adjacency_Matrix_Pearson(matrix_temp1)
dataset_permute = permute(matrix_temp1,[2 1]);
Pearson_matrix_item = abs(corrcoef(dataset_permute));
Absolute_Pearson_matrix = Pearson_matrix_item;
Eye_Matrix = eye(64, 64);
Adjacency_Matrix = Absolute_Pearson_matrix - Eye_Matrix;
matrix = Adjacency_Matrix;