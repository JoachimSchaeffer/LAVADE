load('FTIR_train.mat')
load('FTIR_val.mat')
load('FTIR_test.mat')

%% Merge Data
X = [x_train; x_val; x_test];
y = [y_train; y_val; y_test];
clear x_train x_val x_test
clear y_train y_val y_test
% The data has 6 groups that are very easy to detect and well separated, 
% thus this apporoach is stable for this specific dataset. 
group = kmeans(y, 6);

%%
save('FTIR_data', 'X', 'y', 'freq', 'group')

%%
load('FTIR_data')
new_group = group;
% based on groups would have to be adjusted again, because random seed
% wasn't set!
new_group(group==1) = 0;
new_group(group==2) = 4;
new_group(group==3) = 5;
new_group(group==4) = 1;
new_group(group==5) = 3;
new_group(group==6) = 2;
group = new_group;
clear new_group
save('FTIR_data', 'X', 'y', 'freq', 'group')
%% 
figure 
scatter(group, y)
