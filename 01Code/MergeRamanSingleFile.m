% Merge all the different files together! 
% Goal: 1 File 
% Col 1: Run identifier, Col 2-3327: X, Col 3328:3333 y, y labels:
% Gluc, Lac, Nova
clc
clear all

%% Read 
folder = './Data/Ramanmsback/';
X1 = readmatrix([folder, '20200820_fb_r3_X_nbFlex.csv']);
y1 = readmatrix([folder, '20200820_fb_r3_y_nbFlex.csv']);
X2 = readmatrix([folder, '20200911_perf_r1_X_nbFlex.csv']);
y2 = readmatrix([folder, '20200911_perf_r1_y_nbFlex.csv']);
X3 = readmatrix([folder, '20210312_perf_r4_X_nbFlex.csv']);
y3 = readmatrix([folder, '20210312_perf_r4_y_nbFlex.csv']);
X4 = readmatrix([folder, '20210709_perf_R2_X_nbFlex.csv']);
y4 = readmatrix([folder, '20210709_perf_R2_y_nbFlex.csv']);
X5 = readmatrix([folder, '20211119_perf_R3_X_nbFlex.csv']);
y5 = readmatrix([folder, '20211119_perf_R3_y_nbFlex.csv']);
% Frequency 
x_vals = readmatrix([folder, '20200820_fb_r3spectra_X_nbFlex.csv']);

%% Process 
X1 = [ones(size(X1,1), 1), X1, y1(:, 1:4)];
X2 = [2*ones(size(X2,1), 1), X2, y2(:, 1:4)];
X3 = [3*ones(size(X3,1), 1), X3, y3(:, 1:4)];
X4 = [4*ones(size(X4,1), 1), X4, y4(:, 1:4)];
X5 = [5*ones(size(X5,1), 1), X5, y5(:, 1:4)];
X = [X1;X2;X3;X4;X5];
%%
T = array2table(X);
%T = array2table(data_raman);
T.Properties.VariableNames(1) = {...
    'RunId 1:fb_r3, 2:perf_r1, 3:perf_r4, 4:perf_R2, 4:perf_R2'};
for i=2:3327
    T.Properties.VariableNames(i) = {[int2str(x_vals(i-1)), 'cm^{-1}']};
end
T.Properties.VariableNames(3328:3331) = {'Gluc', 'Lac', 'Gln', 'NH4'};

%% Save
writetable(T,'RamanMsback.csv')