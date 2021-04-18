%% Script to run case studies on LAVADE

%% Clean up!
clc; clear all; close all;
warning('off')
%% Initialize stuff
nb_tests = 1000; 

%% Start the case studies
Case_study.DC = case_study(nb_tests, "DC");
Case_study.DSC = case_study(nb_tests, "DSC");
Case_study.NC = case_study(nb_tests, "NC");


%% Printing of Latex Table in the console 
input.dataFormat = {'%.3f'};
input.tableBorders = 0;
latexDC = 1;
latexDSC = 1;
latexNC = 1;

if latexDC==1
    input.data = [Case_study.DC.results{:,2},Case_study.DC.results{:,8}, Case_study.DC.results{:,5}, Case_study.DC.results{:,11}];
    input.tableColLabels = {'$\mu (RSS_{Train})$', '$\sigma RSS_{Train}$', '$\mu RSS_{Test}$', '$\sigma RSS_{Test}$'};
    input.tableRowLabels = {'PLS (1)','PLS (2)','PCR (1)','PCR (2)', 'LASSO', 'RR', 'EN'};
    latexTable(input);
    
    input.data = [Case_study.DC.results{:,3},Case_study.DC.results{:,9}, Case_study.DC.results{:,6}, Case_study.DC.results{:,12}];
    input.tableColLabels = {'$\mu (RMSE_{Train})$', '$\sigma RMSE_{Train}$', '$\mu RMSE_{Test}$', '$\sigma RMSE_{Test}$'};
    latexTable(input);
end

if latexDSC==1
    input.data = [Case_study.DSC.results{:,2},Case_study.DSC.results{:,8}, Case_study.DSC.results{:,5}, Case_study.DSC.results{:,11}];
    input.tableColLabels = {'$\mu (RSS_{Train})$', '$\sigma RSS_{Train}$', '$\mu RSS_{Test}$', '$\sigma RSS_{Test}$'};
    input.tableRowLabels = {'PLS (1)','PLS (2)','PCR (1)','PCR (2)', 'LASSO', 'RR', 'EN'};
    latexTable(input);
    
    
    input.data = [Case_study.DSC.results{:,3},Case_study.DSC.results{:,9}, Case_study.DSC.results{:,6}, Case_study.DSC.results{:,12}];
    input.tableColLabels = {'$\mu (RMSE_{Train})$', '$\sigma RMSE_{Train}$', '$\mu RMSE_{Test}$', '$\sigma RMSE_{Test}$'};
    latexTable(input);
end

if latexNC==1
    input.data = [Case_study.NC.results{:,2},Case_study.NC.results{:,8}, Case_study.NC.results{:,5}, Case_study.NC.results{:,11}];
    input.tableColLabels = {'$\mu (RSS_{Train})$', '$\sigma RSS_{Train}$', '$\mu RSS_{Test}$', '$\sigma RSS_{Test}$'};
    input.tableRowLabels = {'PLS (1)','PLS (2)','PLS (3)','PLS (4)','PLS (5)','PLS (20)', ...
        'PCR (1)','PCR (2)','PCR (3)','PCR (4)','PCR (5)','PCR (20)', ...
        'RR ($\alpha = 0.001$)','RR10 ($\alpha = 10$)','LASSO', 'EN'};
    latexTable(input);
    
    input.data = [Case_study.NC.results{:,3},Case_study.NC.results{:,9}, Case_study.NC.results{:,6}, Case_study.NC.results{:,12}];
    input.tableColLabels = {'$\mu (RMSE_{Train})$', '$\sigma RMSE_{Train}$', '$\mu RMSE_{Test}$', '$\sigma RMSE_{Test}$'};
    latexTable(input);
end


%% Functions 

function stats = case_study(nb_tests, case_name)
    max_nloop = nb_tests;
    
    % section still under development
    RowNames = {'PLS1', 'PLS2', 'PLS3','PLS4','PLS5','PLS20'...
            'PCR1','PCR2','PCR3','PCR4','PCR5','PCR20'};
    VariableNames = {'R2_Train', 'R2_Test'};
    r2_table = cell2table(cell(12,2), 'VariableNames', VariableNames,...
            'RowNames', RowNames);
    
    app = lavade_exported;
    
    for i = 1:max_nloop
        init(app)
        if case_name == "DC"
        elseif case_name == "DSC"
            app.StandardizeInputsCheckBox.Value = true;
            
        elseif case_name == "NC"
            app.SNRLeftSlider.Value = 20;
            app.RightSlider.Value = 20;   
            app.SignalSlider.Value = 20;
            app.SigEEditField.Value = 2;
            app.SigSEditField.Value = 2;
            app.NoiseCheckBox.Value = true;
        end 
        

        % PLS Section 
        app.MethodDropDown.Value = "PLS";
        
        app.ComponentsEditField.Value = 1;
        init(app)
        stats.pls1(i,:) = app.stats; 

        app.ComponentsEditField.Value = 2;
        init(app)
        stats.pls2(i,:) = app.stats; 
        
        % Noisy case needs more details
        if case_name == "NC"
           
            app.ComponentsEditField.Value = 3;
            init(app)
            stats.pls3(i,:) = app.stats;  

            app.ComponentsEditField.Value = 4;
            init(app)
            stats.pls4(i,:) = app.stats; 

            app.ComponentsEditField.Value = 5;
            init(app)
            stats.pls5(i,:) = app.stats; 

            app.ComponentsEditField.Value = 20;
            init(app)
            stats.pls20(i,:) = app.stats; 

        end
            
        % PCR Section 
        app.MethodDropDown.Value = "PCR";
        app.ComponentsEditField.Value = 1;
        init(app)
        stats.pcr1(i,:) = app.stats; 

        app.ComponentsEditField.Value = 2;
        init(app)
        stats.pcr2(i,:) = app.stats; 
        
        % Noisy case needs more details
        if case_name == "NC"
           
            app.ComponentsEditField.Value = 3;
            init(app)
            stats.pcr3(i,:) = app.stats; 

            app.ComponentsEditField.Value = 4;
            init(app)
            stats.pcr4(i,:) = app.stats; 

            app.ComponentsEditField.Value = 5;
            init(app)
            stats.pcr5(i,:) = app.stats; 

            app.ComponentsEditField.Value = 20;
            init(app)
            stats.pcr20(i,:) = app.stats; 

        end
        
        % LASSO Section 
        app.MethodDropDown.Value = "LASSO";
        app.RegularizationEditField.Value = 0.001;
        init(app)
        stats.lasso(i,:) = app.stats; 

        % RR Section 
        app.MethodDropDown.Value = "RR";
        app.RegularizationEditField.Value = 0.001;
        init(app)
        stats.rrd(i,:) = app.stats; 
        
        if case_name == "NC"
            app.RegularizationEditField.Value = 10;
            init(app)
            stats.rr10(i,:) = app.stats; 
        end

        % EN Section 
        app.MethodDropDown.Value = "EN";
        app.RegularizationEditField.Value = 0.001;
        init(app)
        stats.EN(i,:) = app.stats; 
        
        % Print some stuff in the console
        if mod(i,10)==0 
            i
            case_name
        end
        
    end 


    %% Initialize and start filling results structure
    Variable_names = {'Mean_R2_Train', 'Mean_RSS_Train', 'Mean_RMSE_Train', ...
        'Mean_R2_Test', 'Mean_RSS_Test', 'Mean_RMSE_Test', ...
        'Std_R2_Train', 'Std_RSS_Train', 'Std_RMSE_Train',...
        'Std_R2_Test', 'Std_RSS_Test', 'Std_RMSE_Test'};
    num_vars = 12;
    stats.results  = cell2table(cell(0,num_vars), 'VariableNames', Variable_names);
    
    fns = fieldnames(stats);
    for i=1:length(fns)-1
        [mean_, std_] = create_stats(stats.(fns{i}));
        for j=1:int64(num_vars/2)
            stats.results(fns{i},j) = {mean_(j)};
            stats.results(fns{i},j+int64(num_vars/2)) = {std_(j)};
        end 
    end

end 

%% Helper Function

function [mean_,std_] = create_stats(vector)
        mean_ = mean(vector); 
        std_ = std(vector);
end



%{
% Now use this table as input in our input struct:
input.data = T;
input.data = [Case_study.DC.results(:,2),Case_study.DC.results(:,5), Case_study.DC.results(:,8), Case_study.DC.results(:,11)];
input.data = [Case_study.DC.results(:,3),Case_study.DC.results(:,6), Case_study.DC.results(:,9), Case_study.DC.results(:,12)];
input.data = [Case_study.DC.results(:,2:3),Case_study.DC.results(:,5:6), Case_study.DC.results(:,8:9), Case_study.DC.results(:,11:12)];


% Set the row format of the data values (in this example we want to use
% integers only):
input.dataFormat = {'%.3f'};

% Column alignment ('l'=left-justified, 'c'=centered,'r'=right-justified):
input.tableColumnAlignment = 'c';

% Switch table borders on/off:
input.tableBorders = 1;

% Switch to generate a complete LaTex document or just a table:
input.makeCompleteLatexDocument = 1;

% Now call the function to generate LaTex code:
latex = latexTable(input);
%} 
