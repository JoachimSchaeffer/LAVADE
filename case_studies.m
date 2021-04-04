%% Script to run case studies on LAVADE

%% Clean up!
clc; clear all; close all;

%% Initialize stuff
nb_tests = 200; 
nb_workers = 4;

%% Start the case studies
%% Default case



%% These variables seem to be necessary for parallelization. 
% Let me know in case there's a smarter way to solve that issue! 
% MATLAB seem to 'baby' the used a lot and I didn't find a way to pass more
% low level instrutions to the workers. 

Case_study.DC = case_study(nb_tests, nb_workers, "DC");
Case_study.DSC = case_study(nb_tests, nb_workers, "DSC");
Case_study.NC = case_study(nb_tests, nb_workers, "NC");



function results = case_study(nb_tests, nb_workers, case_name)
    tic
    max_nloop = int64(nb_tests/nb_workers);
    
    if case_name == "..."
        RowNames = {'PLS1', 'PLS2', 'PLS3','PLS4','PLS5','PLS20'...
            'PCR1','PCR2','PCR3','PCR4','PCR5','PCR20'};
        VariableNames = {'R2_Train', 'R2_Test'};
        r2_table = cell2table(cell(12,2), 'VariableNames', VariableNames,...
            'RowNames', RowNames);
    else 
        r2_train_pls1 = [];
        r2_test_pls1 = [];
        r2_train_pls2 = [];
        r2_test_pls2 = [];
        r2_train_pls3 = [];
        r2_test_pls3 = [];
        r2_train_pls4 = [];
        r2_test_pls4 = [];
        r2_train_pls5 = [];
        r2_test_pls5 = [];
        r2_train_pls20 = [];
        r2_test_pls20 = [];

        r2_train_pcr1 = [];
        r2_test_pcr1 = [];
        r2_train_pcr2 = [];
        r2_test_pcr2 = [];
        
        r2_train_pcr3 = [];
        r2_test_pcr3 = [];
        r2_train_pcr4 = [];
        r2_test_pcr4 = [];
        r2_train_pcr5 = [];
        r2_test_pcr5 = [];
        r2_train_pcr20 = [];
        r2_test_pcr20 = [];
    end
    

    r2_train_cca = [];
    r2_test_cca = [];

    r2_train_lasso = [];
    r2_test_lasso = [];

    r2_train_rr = [];
    r2_test_rr = [];

    r2_train_en = [];
    r2_test_en = [];


    tic

    parfor i = 1:nb_workers
        app = lavade_exported;
        
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

        r2train_ = [];
        r2test_ = [];
        for j= 1:max_nloop
            init(app)
            r2test_(j) = app.r2_test; 
            r2train_(j) = app.r2_train; 
        end
        r2_train_pls1(i, :) = r2train_; 
        r2_test_pls1(i,:) =  r2test_;


        app.ComponentsEditField.Value = 2;

        r2train_ = [];
        r2test_ = [];
        for j= 1:max_nloop
            init(app)
            r2test_(j) = app.r2_test; 
            r2train_(j) = app.r2_train; 
        end
        r2_train_pls2(i, :) = r2train_; 
        r2_test_pls2(i,:) =  r2test_;
        
        % Noisy case needs more details
        if case_name == "NC"
           
            app.ComponentsEditField.Value = 3;

            r2train_ = [];
            r2test_ = [];
            for j= 1:max_nloop
                init(app)
                r2test_(j) = app.r2_test; 
                r2train_(j) = app.r2_train; 
            end
            r2_train_pls3(i, :) = r2train_; 
            r2_test_pls3(i,:) =  r2test_;


            app.ComponentsEditField.Value = 4;

            r2train_ = [];
            r2test_ = [];
            for j= 1:max_nloop
                init(app)
                r2test_(j) = app.r2_test; 
                r2train_(j) = app.r2_train; 
            end
            r2_train_pls4(i, :) = r2train_; 
            r2_test_pls4(i,:) =  r2test_;

            app.ComponentsEditField.Value = 5;

            r2train_ = [];
            r2test_ = [];
            for j= 1:max_nloop
                init(app)
                r2test_(j) = app.r2_test; 
                r2train_(j) = app.r2_train; 
            end
            r2_train_pls5(i, :) = r2train_; 
            r2_test_pls5(i,:) =  r2test_;


            app.ComponentsEditField.Value = 20;

            r2train_ = [];
            r2test_ = [];
            for j= 1:max_nloop
                init(app)
                r2test_(j) = app.r2_test; 
                r2train_(j) = app.r2_train; 
            end
            r2_train_pls20(i, :) = r2train_; 
            r2_test_pls20(i,:) =  r2test_;

        end
            


        % PCR Section 
        app.MethodDropDown.Value = "PCR";
        app.ComponentsEditField.Value = 1;

        r2train_ = [];
        r2test_ = [];
        for j= 1:max_nloop
            init(app)
            r2test_(j) = app.r2_test; 
            r2train_(j) = app.r2_train; 
        end
        r2_train_pcr1(i, :) = r2train_; 
        r2_test_pcr1(i,:) =  r2test_;


        app.ComponentsEditField.Value = 2;

        r2train_ = [];
        r2test_ = [];
        for j= 1:max_nloop
            init(app)
            r2test_(j) = app.r2_test; 
            r2train_(j) = app.r2_train; 
        end
        r2_train_pcr2(i, :) = r2train_; 
        r2_test_pcr2(i,:) =  r2test_;

        % Noisy case needs more details
        if case_name == "NC"
           
            app.ComponentsEditField.Value = 3;

            r2train_ = [];
            r2test_ = [];
            for j= 1:max_nloop
                init(app)
                r2test_(j) = app.r2_test; 
                r2train_(j) = app.r2_train; 
            end
            r2_train_pcr3(i, :) = r2train_; 
            r2_test_pcr3(i,:) =  r2test_;


            app.ComponentsEditField.Value = 4;

            r2train_ = [];
            r2test_ = [];
            for j= 1:max_nloop
                init(app)
                r2test_(j) = app.r2_test; 
                r2train_(j) = app.r2_train; 
            end
            r2_train_pcr4(i, :) = r2train_; 
            r2_test_pcr4(i,:) =  r2test_;

            app.ComponentsEditField.Value = 5;

            r2train_ = [];
            r2test_ = [];
            for j= 1:max_nloop
                init(app)
                r2test_(j) = app.r2_test; 
                r2train_(j) = app.r2_train; 
            end
            r2_train_pcr5(i, :) = r2train_; 
            r2_test_pcr5(i,:) =  r2test_;


            app.ComponentsEditField.Value = 20;

            r2train_ = [];
            r2test_ = [];
            for j= 1:max_nloop
                init(app)
                r2test_(j) = app.r2_test; 
                r2train_(j) = app.r2_train; 
            end
            r2_train_pcr20(i, :) = r2train_; 
            r2_test_pcr20(i,:) =  r2test_;

        end

        % CCA Section 
        app.MethodDropDown.Value = "CCA";

        r2train_ = [];
        r2test_ = [];
        for j= 1:max_nloop
            init(app)
            r2test_(j) = app.r2_test; 
            r2train_(j) = app.r2_train; 
        end
        r2_train_cca(i, :) = r2train_; 
        r2_test_cca(i,:) =  r2test_;


        % LASSO Section 
        app.MethodDropDown.Value = "LASSO";

        r2train_ = [];
        r2test_ = [];
        for j= 1:max_nloop
            init(app)
            r2test_(j) = app.r2_test; 
            r2train_(j) = app.r2_train; 
        end
        r2_train_lasso(i, :) = r2train_; 
        r2_test_lasso(i,:) =  r2test_;

        % RR Section 
        app.MethodDropDown.Value = "RR";

        r2train_ = [];
        r2test_ = [];
        for j= 1:max_nloop
            init(app)
            r2test_(j) = app.r2_test; 
            r2train_(j) = app.r2_train; 
        end
        r2_train_rr(i, :) = r2train_; 
        r2_test_rr(i,:) =  r2test_;

        % EN Section 
        app.MethodDropDown.Value = "EN";

        r2train_ = [];
        r2test_ = [];
        for j= 1:max_nloop
            init(app)
            r2test_(j) = app.r2_test; 
            r2train_(j) = app.r2_train; 
        end
        r2_train_en(i, :) = r2train_; 
        r2_test_en(i,:) =  r2test_;

    end 

    toc 


    %% Initialize and start filling results structure

    results.stats  = cell2table(cell(0,4), 'VariableNames', {'Mean_R2_Train', ...
        'Std_R2_Train', 'Mean_R2_Test', 'Std_R2_Test' });

    results.pls1.r2train = reshape(r2_train_pls1,1,numel(r2_train_pls1));
    results.pls1.r2test = reshape(r2_test_pls1,1,numel(r2_test_pls1)); 
    results.pls2.r2train = reshape(r2_train_pls2,1,numel(r2_train_pls2));
    results.pls2.r2test = reshape(r2_test_pls2,1,numel(r2_test_pls2)); 
    
    if case_name =="NC"
        results.pls3.r2train = reshape(r2_train_pls3,1,numel(r2_train_pls3));
        results.pls3.r2test = reshape(r2_test_pls3,1,numel(r2_test_pls3)); 
        results.pls4.r2train = reshape(r2_train_pls4,1,numel(r2_train_pls4));
        results.pls4.r2test = reshape(r2_test_pls4,1,numel(r2_test_pls4)); 
        results.pls5.r2train = reshape(r2_train_pls5,1,numel(r2_train_pls5));
        results.pls5.r2test = reshape(r2_test_pls5,1,numel(r2_test_pls5)); 
        results.pls20.r2train = reshape(r2_train_pls20,1,numel(r2_train_pls20));
        results.pls20.r2test = reshape(r2_test_pls20,1,numel(r2_test_pls20)); 
    end

    results.pcr1.r2train = reshape(r2_train_pcr1,1,numel(r2_train_pcr1));
    results.pcr1.r2test = reshape(r2_test_pcr1,1,numel(r2_test_pcr1)); 
    results.pcr2.r2train = reshape(r2_train_pcr2,1,numel(r2_train_pcr2));
    results.pcr2.r2test = reshape(r2_test_pcr2,1,numel(r2_test_pcr2)); 

    if case_name =="NC"
        results.pcr3.r2train = reshape(r2_train_pcr3,1,numel(r2_train_pcr3));
        results.pcr3.r2test = reshape(r2_test_pcr3,1,numel(r2_test_pcr3)); 
        results.pcr4.r2train = reshape(r2_train_pcr4,1,numel(r2_train_pcr4));
        results.pcr4.r2test = reshape(r2_test_pcr4,1,numel(r2_test_pcr4)); 
        results.pcr5.r2train = reshape(r2_train_pcr5,1,numel(r2_train_pcr5));
        results.pcr5.r2test = reshape(r2_test_pcr5,1,numel(r2_test_pcr5)); 
        results.pcr20.r2train = reshape(r2_train_pcr20,1,numel(r2_train_pcr20));
        results.pcr20.r2test = reshape(r2_test_pcr20,1,numel(r2_test_pcr20)); 
    end
    
    results.cca.r2train = reshape(r2_train_cca,1,numel(r2_train_cca));
    results.cca.r2test = reshape(r2_test_cca,1,numel(r2_test_cca)); 

    results.lasso.r2train = reshape(r2_train_lasso,1,numel(r2_train_lasso));
    results.lasso.r2test = reshape(r2_test_lasso,1,numel(r2_test_lasso)); 

    results.rr.r2train = reshape(r2_train_rr,1,numel(r2_train_rr));
    results.rr.r2test = reshape(r2_test_rr,1,numel(r2_test_rr)); 

    results.en.r2train = reshape(r2_train_en,1,numel(r2_train_en));
    results.en.r2test = reshape(r2_test_en,1,numel(r2_test_en)); 


    [results.stats.Mean_R2_Train('PLS1'),results.stats.Std_R2_Train('PLS1')] = stats(results.pls1.r2train);
    [results.stats.Mean_R2_Test('PLS1'),results.stats.Std_R2_Test('PLS1')] = stats(results.pls1.r2test);
    [results.stats.Mean_R2_Train('PLS2'),results.stats.Std_R2_Train('PLS2')] = stats(results.pls2.r2train);
    [results.stats.Mean_R2_Test('PLS2'),results.stats.Std_R2_Test('PLS2')] = stats(results.pls2.r2test);

    [results.stats.Mean_R2_Train('PCR1'),results.stats.Std_R2_Train('PCR1')] = stats(results.pcr1.r2train);
    [results.stats.Mean_R2_Test('PCR1'),results.stats.Std_R2_Test('PCR1')] = stats(results.pcr1.r2test);
    [results.stats.Mean_R2_Train('PCR2'),results.stats.Std_R2_Train('PCR2')] = stats(results.pcr2.r2train);
    [results.stats.Mean_R2_Test('PCR2'),results.stats.Std_R2_Test('PCR2')] = stats(results.pcr2.r2test);
    
    if case_name == "NC"
        [results.stats.Mean_R2_Train('PLS3'),results.stats.Std_R2_Train('PLS3')] = stats(results.pls3.r2train);
        [results.stats.Mean_R2_Test('PLS3'),results.stats.Std_R2_Test('PLS3')] = stats(results.pls3.r2test);
        [results.stats.Mean_R2_Train('PLS4'),results.stats.Std_R2_Train('PLS4')] = stats(results.pls4.r2train);
        [results.stats.Mean_R2_Test('PLS4'),results.stats.Std_R2_Test('PLS4')] = stats(results.pls4.r2test);
        [results.stats.Mean_R2_Train('PLS5'),results.stats.Std_R2_Train('PLS5')] = stats(results.pls5.r2train);
        [results.stats.Mean_R2_Test('PLS5'),results.stats.Std_R2_Test('PLS5')] = stats(results.pls5.r2test);
        [results.stats.Mean_R2_Train('PLS20'),results.stats.Std_R2_Train('PLS20')] = stats(results.pls20.r2train);
        [results.stats.Mean_R2_Test('PLS20'),results.stats.Std_R2_Test('PLS20')] = stats(results.pls20.r2test);
        
        [results.stats.Mean_R2_Train('PCR3'),results.stats.Std_R2_Train('PCR3')] = stats(results.pcr3.r2train);
        [results.stats.Mean_R2_Test('PCR3'),results.stats.Std_R2_Test('PCR3')] = stats(results.pcr3.r2test);
        [results.stats.Mean_R2_Train('PCR4'),results.stats.Std_R2_Train('PCR4')] = stats(results.pcr4.r2train);
        [results.stats.Mean_R2_Test('PCR4'),results.stats.Std_R2_Test('PCR4')] = stats(results.pcr4.r2test);
        [results.stats.Mean_R2_Train('PCR5'),results.stats.Std_R2_Train('PCR5')] = stats(results.pcr5.r2train);
        [results.stats.Mean_R2_Test('PCR5'),results.stats.Std_R2_Test('PCR5')] = stats(results.pcr5.r2test);
        [results.stats.Mean_R2_Train('PCR20'),results.stats.Std_R2_Train('PCR20')] = stats(results.pcr20.r2train);
        [results.stats.Mean_R2_Test('PCR20'),results.stats.Std_R2_Test('PCR20')] = stats(results.pcr20.r2test);
    end 


    [results.stats.Mean_R2_Train('CCA'),results.stats.Std_R2_Train('CCA')] = stats(results.cca.r2train);
    [results.stats.Mean_R2_Test('CCA'),results.stats.Std_R2_Test('CCA')] = stats(results.cca.r2test);

    [results.stats.Mean_R2_Train('LASSO'),results.stats.Std_R2_Train('LASSO')] = stats(results.lasso.r2train);
    [results.stats.Mean_R2_Test('LASSO'),results.stats.Std_R2_Test('LASSO')] = stats(results.lasso.r2test);

    [results.stats.Mean_R2_Train('RR'),results.stats.Std_R2_Train('RR')] = stats(results.rr.r2train);
    [results.stats.Mean_R2_Test('RR'),results.stats.Std_R2_Test('RR')] = stats(results.rr.r2test);

    [results.stats.Mean_R2_Train('EN'),results.stats.Std_R2_Train('EN')] = stats(results.en.r2train);
    [results.stats.Mean_R2_Test('EN'),results.stats.Std_R2_Test('EN')] = stats(results.en.r2test);

end 
%% Helper Function

function [mean_,std_] = stats(vector)
        mean_ = mean(vector); 
        std_ = std(vector);
end
