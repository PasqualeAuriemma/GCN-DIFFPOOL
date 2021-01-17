clear all
clc

format long

num_trial = 20;
%num_trial = 101;
path = 'Download_Raw_EEG_Data\20-Subjects\';
%path = 'Download_Raw_EEG_Data\100-Subjects\';

%% Read and Create Labels

file_Label = [path, 'Labels_1.mat'];
Labels1 = load(file_Label);
ne_Label = Labels1.Labels;

ne_Label = reshape(ne_Label, num_trial*84, 4);

ne_Label = ne_Label(:, :);

[r, t] = max(ne_Label, [], 2);

%%
% Read the Data and Create Dataset
Stack_Dataset = [];
ne_Dataset = [];


for pers = 1:num_trial

    new_Dataset_temp = [];
    for i = 1:64
        Dataset = [path, 'Dataset_', num2str(i), '.mat'];
        Dataset = load(Dataset);
        Dataset = Dataset.Dataset;
        Dataset = Dataset(pers, :, :);
        Dataset = reshape(Dataset, 84, 640);
        
        [row, column] = size(Dataset);
        Dataset = reshape(Dataset, 1, row, column);
        new_Dataset_temp = [new_Dataset_temp; Dataset];
    end
    new_Dataset_temp = permute(new_Dataset_temp,[2 1 3]);
    %[row1, column1, z1] = size(new_Dataset_temp);
    %new_Dataset_temp = reshape(new_Dataset_temp, 1, row1, column1, z1);
    ne_Dataset = [ne_Dataset; new_Dataset_temp]; 
end
ne_Dataset_tmp = ne_Dataset;
ne_Dataset = ne_Dataset - mean(ne_Dataset, 2);


Dataset_1 = [];
Dataset_2 = [];
Dataset_3 = [];
Dataset_4 = [];

new_Dataset_1 = [];
new_Label_1 = [];
new_Dataset_2 = [];
new_Label_2 = [];
new_Dataset_3 = [];
new_Label_3 = [];
new_Dataset_4 = [];
new_Label_4 = [];

for p = 1:size(t,1)
   if t(p) == 1
      Dataset_1 = [Dataset_1; ne_Dataset_tmp(p,:,:)];
      new_Dataset_1 = [new_Dataset_1; ne_Dataset(p,:,:)]; 
      new_Label_1 = [new_Label_1; ne_Label(p,:)];
   elseif t(p) == 2
      Dataset_2 = [Dataset_2; ne_Dataset_tmp(p,:,:)];
      new_Dataset_2 = [new_Dataset_2; ne_Dataset(p,:,:)]; 
      new_Label_2 = [new_Label_2; ne_Label(p,:)];
   elseif t(p) == 3
      Dataset_3 = [Dataset_3; ne_Dataset_tmp(p,:,:)];
      new_Dataset_3 = [new_Dataset_3; ne_Dataset(p,:,:)]; 
      new_Label_3 = [new_Label_3; ne_Label(p,:)];
   elseif t(p) == 4
      Dataset_4 = [Dataset_4; ne_Dataset_tmp(p,:,:)]; 
      new_Dataset_4 = [new_Dataset_4; ne_Dataset(p,:,:)]; 
      new_Label_4 = [new_Label_4; ne_Label(p,:)];   
   end
end
%%
matrix_temp1 = zeros(size(new_Dataset_1,2), size(new_Dataset_1,1) * size(new_Dataset_1,3));
index01 = 1;
index11= 640;
for pp = 1:size(new_Dataset_1,1)
    index01 = 1 + (640 * (pp - 1));
    index11 = 640 * pp;
    for pp1 = 1:size(new_Dataset_1,2)
       channel_temp = squeeze(new_Dataset_1(pp,pp1,:))';
       matrix_temp1(pp1, index01:index11) =  channel_temp;
    end
end

matrix_temp2 = zeros(size(new_Dataset_2,2), size(new_Dataset_2,1) * size(new_Dataset_2,3));
index02 = 1;
index12= 640;
for pp = 1:size(new_Dataset_2,1)
    index02 = 1 + (640 * (pp - 1));
    index12 = 640 * pp;
    for pp1 = 1:size(new_Dataset_2,2)
       channel_temp2 = squeeze(new_Dataset_2(pp,pp1,:))';
       matrix_temp2(pp1, index02:index12) =  channel_temp2;
    end
end

matrix_temp3 = zeros(size(new_Dataset_3,2), size(new_Dataset_3,1) * size(new_Dataset_3,3));
index03 = 1;
index13= 640;
for pp = 1:size(new_Dataset_3,1)
    index03 = 1 + (640 * (pp - 1));
    index13 = 640 * pp;
    for pp1 = 1:size(new_Dataset_3,2)
       channel_temp3 = squeeze(new_Dataset_3(pp,pp1,:))';
       matrix_temp3(pp1, index03:index13) =  channel_temp3;
    end
end

matrix_temp4 = zeros(size(new_Dataset_4,2), size(new_Dataset_4,1) * size(new_Dataset_4,3));
index04 = 1;
index14= 640;
for pp = 1:size(new_Dataset_4,1)
    index04 = 1 + (640 * (pp - 1));
    index14 = 640 * pp;
    for pp1 = 1:size(new_Dataset_4,2)
       channel_temp4 = squeeze(new_Dataset_4(pp,pp1,:))';
       matrix_temp4(pp1, index04:index14) =  channel_temp4;
    end
end

Adjacency_Matrix = Adjacency_Matrix_PLI(matrix_temp1);
diagonal_vector = sum(Adjacency_Matrix, 2);
Degree_Matrix = diag(diagonal_vector);
Laplacian_Matrix = Degree_Matrix - Adjacency_Matrix;
[row, column] = size(Laplacian_Matrix);
%pli_graph = [pli_graph; reshape(Laplacian_Matrix, 1, row, column)];

figure(1)
imagesc(Laplacian_Matrix)
axis square
title('pli_graph for 20 Subjects', 'FontName', 'Times New Roman', 'FontSize', 16, 'FontWeight', 'bold')
xlabel('Channels'), ylabel('Channels')
set(gca, 'FontName', 'Times New Roman', 'FontSize', 16, 'FontWeight', 'bold');
colorbar
print('pli_graph_for_20_Subjects', '-dpng',  '-r600')

% 2 MI ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Adjacency_Matrix2 = Adjacency_Matrix_PLI(matrix_temp2);
diagonal_vector2 = sum(Adjacency_Matrix2, 2);
Degree_Matrix2 = diag(diagonal_vector2);
Laplacian_Matrix2 = Degree_Matrix2 - Adjacency_Matrix2;
[row, column] = size(Laplacian_Matrix2);
%pli_graph2 = reshape(Laplacian_Matrix2, 1, row, column);

figure(2)
imagesc(Laplacian_Matrix2)
axis square
title('pli_graph for 20 Subjects', 'FontName', 'Times New Roman', 'FontSize', 16, 'FontWeight', 'bold')
xlabel('Channels'), ylabel('Channels')
set(gca, 'FontName', 'Times New Roman', 'FontSize', 16, 'FontWeight', 'bold');
colorbar
print('pli_graph_for_20_Subjects', '-dpng',  '-r600')

% 3 MI ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Adjacency_Matrix3 = Adjacency_Matrix_PLI(matrix_temp3);
diagonal_vector3 = sum(Adjacency_Matrix3, 2);
Degree_Matrix3 = diag(diagonal_vector3);
Laplacian_Matrix3 = Degree_Matrix3 - Adjacency_Matrix3;
[row, column] = size(Laplacian_Matrix3);
%pli_graph3 = reshape(Laplacian_Matrix3, 1, row, column);

figure(3)
imagesc(Laplacian_Matrix3)
axis square
title('pli_graph for 20 Subjects', 'FontName', 'Times New Roman', 'FontSize', 16, 'FontWeight', 'bold')
xlabel('Channels'), ylabel('Channels')
set(gca, 'FontName', 'Times New Roman', 'FontSize', 16, 'FontWeight', 'bold');
colorbar
print('pli_graph_for_20_Subjects', '-dpng',  '-r600')

% 4 MI ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Adjacency_Matrix4 = Adjacency_Matrix_PLI(matrix_temp4);
diagonal_vector4 = sum(Adjacency_Matrix4, 2);
Degree_Matrix4 = diag(diagonal_vector4);
Laplacian_Matrix4 = Degree_Matrix4 - Adjacency_Matrix4;
[row, column] = size(Laplacian_Matrix4);
%pli_graph4 = reshape(Laplacian_Matrix4, 1, row, column);

figure(4)
imagesc(Laplacian_Matrix4)
axis square
title('pli_graph for 20 Subjects', 'FontName', 'Times New Roman', 'FontSize', 16, 'FontWeight', 'bold')
xlabel('Channels'), ylabel('Channels')
set(gca, 'FontName', 'Times New Roman', 'FontSize', 16, 'FontWeight', 'bold');
colorbar
print('pli_graph_for_20_Subjects', '-dpng',  '-r600')

%%

pli_graph1 = []; pli_graph2 = []; pli_graph3 = []; pli_graph4 = [];

for pp = 1:size(new_Dataset_1,1)
    pli_graph1 = [pli_graph1; reshape(Adjacency_Matrix, 1, row, column)];
end

for pp = 1:size(new_Dataset_2,1)
    pli_graph2 = [pli_graph2; reshape(Adjacency_Matrix2, 1, row, column)];
end

for pp = 1:size(new_Dataset_3,1)
    pli_graph3 = [pli_graph3; reshape(Adjacency_Matrix3, 1, row, column)];
end

for pp = 1:size(new_Dataset_4,1)
    pli_graph4 = [pli_graph4; reshape(Adjacency_Matrix4, 1, row, column)];
end


pli_graph = [pli_graph1; pli_graph2; pli_graph3; pli_graph4];
new_Dataset = [Dataset_1; Dataset_2; Dataset_3; Dataset_4];
new_Label = [new_Label_1; new_Label_2; new_Label_3; new_Label_4];

number_item = size(new_Label, 1);
rk = randperm(number_item);
new_Label = new_Label(rk,:);
new_Dataset = new_Dataset(rk, :, :);
pli_graph = pli_graph(rk, :, :);

r = size(new_Dataset,1);

pli_graph_train = pli_graph(1:fix(r/10*9),     :, :);
pli_graph_test  = pli_graph(fix(r/10*9)+1:end, :, :);
training_set1   = new_Dataset(1:fix(r/10*9),     :, :);
test_set1       = new_Dataset(fix(r/10*9)+1:end, :, :);
label_training1 = new_Label(1:fix(r/10*9),       :);
label_test1     = new_Label(fix(r/10*9)+1:end,   :);

save(strcat(path, 'pli_graph_train.mat'), 'pli_graph_train','-v7.3');
save(strcat(path, 'pli_graph_test.mat'), 'pli_graph_test','-v7.3');
save(strcat(path, 'training_set_1.mat'), 'training_set1','-v7.3');
save(strcat(path, 'test_set_1.mat'), 'test_set1','-v7.3');
save(strcat(path, 'training_label_1.mat'), 'label_training1','-v7.3');
save(strcat(path, 'test_label_1.mat'), 'label_test1','-v7.3');

