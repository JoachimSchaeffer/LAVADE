classdef lavade_exported < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        LAVADEUIFigure                  matlab.ui.Figure
        GroupingofDataCheckBox          matlab.ui.control.CheckBox
        PositionSingnalSlider           matlab.ui.control.Slider
        PositionofSignalLabel           matlab.ui.control.Label
        RightVEditField                 matlab.ui.control.NumericEditField
        RightVEditFieldLabel            matlab.ui.control.Label
        LeftVEditField                  matlab.ui.control.NumericEditField
        LeftVEditFieldLabel             matlab.ui.control.Label
        EndVEditField                   matlab.ui.control.NumericEditField
        EndVEditFieldLabel              matlab.ui.control.Label
        DatasetDropDown                 matlab.ui.control.DropDown
        DatasetDropDownLabel            matlab.ui.control.Label
        ENAlphaSlider                   matlab.ui.control.Slider
        ENAlphaSliderLabel              matlab.ui.control.Label
        Panel                           matlab.ui.container.Panel
        HoldCheckBox                    matlab.ui.control.CheckBox
        StandardizeDataCheckBox         matlab.ui.control.CheckBox
        SigEEditField                   matlab.ui.control.NumericEditField
        SigEEditFieldLabel              matlab.ui.control.Label
        SigSEditField                   matlab.ui.control.NumericEditField
        SigSEditFieldLabel              matlab.ui.control.Label
        ShuffleSplitsButton             matlab.ui.control.Button
        RegularizationEditField         matlab.ui.control.NumericEditField
        RegularizationEditFieldLabel    matlab.ui.control.Label
        ComponentsEditField             matlab.ui.control.NumericEditField
        ComponentsEditFieldLabel        matlab.ui.control.Label
        AddNoiseCheckBox                matlab.ui.control.CheckBox
        LatentVariableMethodDemonstratorLabel  matlab.ui.control.Label
        DrawNewSampleButton             matlab.ui.control.Button
        SigREditField                   matlab.ui.control.NumericEditField
        SigREditFieldLabel              matlab.ui.control.Label
        SigLEditField                   matlab.ui.control.NumericEditField
        SigLEditFieldLabel              matlab.ui.control.Label
        SNRySlider                      matlab.ui.control.Slider
        SNRyLabel                       matlab.ui.control.Label
        StartVEditField                 matlab.ui.control.NumericEditField
        StartVEditFieldLabel            matlab.ui.control.Label
        ExperimentsEditField            matlab.ui.control.NumericEditField
        ExperimentsEditFieldLabel       matlab.ui.control.Label
        relDatapointsEditField          matlab.ui.control.NumericEditField
        relDatapointsEditFieldLabel     matlab.ui.control.Label
        DatapointsEditField             matlab.ui.control.NumericEditField
        DatapointsEditFieldLabel        matlab.ui.control.Label
        TrainTestSplitRatioSlider       matlab.ui.control.Slider
        TrainTestSplitRatioSliderLabel  matlab.ui.control.Label
        SNRXSlider                      matlab.ui.control.Slider
        SNRXLabel                       matlab.ui.control.Label
        MethodDropDown                  matlab.ui.control.DropDown
        MethodDropDownLabel             matlab.ui.control.Label
        UIAxesReg                       matlab.ui.control.UIAxes
        UIAxesRegTrain                  matlab.ui.control.UIAxes
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LAVADE (Latent Variable Demonstrator Software)
% Developed with Matlab 2021a
% Author: Joachim Schaeffer
% Email:  joachim.schaeffer@posteo.de
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
    
    % Change access to public to be able to run case 
    % studies with an external script
    properties (Access = private)
        % Properties that can be accessed in all functions 
        % below, via app.VarName
        DataObj;    % custom object of class defined separately
        % Subplots in the panel section, these properties must 
        % be manually set up here for the subplots to work 
        % in the matlab GUI/App
        ax1;
        ax2;
        rank_text;       % Text for displaying the rank in ax1
    end
    
    % Change access to public to be able to run case 
    % studies with an external script
    methods (Access = private)
        
        function greek_letters(app)
        % Should run only when the app is started!
        % Cretae Panel Subplots 
            app.Panel.AutoResizeChildren = 'off';
            app.ax1 = subplot(2,1,1,'Parent',app.Panel);
            app.ax2 = subplot(2,1,2,'Parent',app.Panel);
            % Greek letters in text fields must be initialized here 
            app.SigLEditFieldLabel.Text = char(963);
            app.SigREditFieldLabel.Text = char(963);
            app.SigSEditFieldLabel.Text = char(963);
            app.SigEEditFieldLabel.Text = char(963);
            app.LeftVEditFieldLabel.Text = strcat("Left  ", char(956));
            app.RightVEditFieldLabel.Text = strcat("Right  ", char(956));
            app.StartVEditFieldLabel.Text = strcat("Start  ", char(956));
            app.EndVEditFieldLabel.Text = strcat("End  ", char(956));
            app.ENAlphaSliderLabel.Text = strcat("EN ", char(945));
        end
            
        function init(app)
        % Sets up the workspace and initializes variables
            app.DataObj = Data;
            app.DataObj.name = app.DatasetDropDown.Value;
            noise_change = 0;
            
            switch app.DataObj.name
                case 'Example'
                    % Generate Examples
                    app.DataObj.measurements = app.ExperimentsEditField.Value;
                    sigma_start = app.SigSEditField.Value;
                    app.DataObj.start_value = normrnd(app.StartVEditField.Value, sigma_start, app.DataObj.measurements, 1);
        
                    sigma_end = app.SigEEditField.Value;
                    app.DataObj.final_value = normrnd(app.EndVEditField.Value, sigma_end, app.DataObj.measurements, 1);
        
                    value_left = app.LeftVEditField.Value;
                    sigma_left = app.SigLEditField.Value;
                    
                    value_right = app.RightVEditField.Value;
                    sigma_right = app.SigREditField.Value;
                    
                    app.DataObj.target_left = normrnd(value_left, sigma_left, app.DataObj.measurements, 1);
                    app.DataObj.target_right = normrnd(value_right, sigma_right, app.DataObj.measurements, 1);

                    % Creating the X vector is outsourced, because this function
                    % is also called when changing the position of the signal; In
                    % that case we don't want to draw new samples. 
                    create_signal(app.DataObj, app, noise_change);
                case  'Paracetamol'
                    create_signal(app.DataObj, app, noise_change); 
                case 'LFP'
                    create_signal(app.DataObj, app, noise_change); 
            end 
            add_noise(app.DataObj, app.AddNoiseCheckBox.Value, app.SNRXSlider.Value, app.SNRySlider.Value);
        end
    
        function [r2, rss, rmse] = fit_stats(~, y, yfit)
        % Helper function to calculate prediction accuracies 
        % Syntax partially copied from 
        % https://ch.mathworks.com/help/matlab/data_analysis/linear-regression.html#f1-15010
        % https://ch.mathworks.com/matlabcentral/answers/4064-rmse-root-mean-square-error
            yresid = y - yfit;
            SSresid = sum(yresid.^2);
            SStotal = (length(y)-1) * var(y);
            r2 = 1 - SSresid/SStotal;
            rss = SSresid;
            rmse = sqrt(mean(yresid.^2));
        end
        
        function [Z, mx, stdx] = normalize_train(~, X)
        % Helper function to normalize the the columns of X to 
        % mean zero, and standard deviation one.
        % Syntax copied from Matlab 'ridge' function. 
        % Reason: The normalize function in the Matlab 2019b 
        % release does not return mean and std. Furthermore, 
        % normalize in 2019b retunrs a column of Nan's in case 
        % the column is constant
            n = size(X, 1);
            mx = mean(X);
            stdx = std(X,0,1);
            idx = find(abs(stdx) < sqrt(eps(class(stdx)))); 
            if any(idx)
              stdx(idx) = 1;
            end
            
            MX = mx(ones(n,1),:);
            STDX = stdx(ones(n,1),:);
            Z = (X - MX) ./ STDX;
            if any(idx)
              Z(:,idx) = 1;
            end
        end
        
        function Z = normalize_test(~, X, mx, stdx)
        % Helper function to normalize the test data based 
        % on the train data stats.
        % Based on syntax from Matlab 'ridge' function. 
            Z = (X-mx)./stdx;
            % Make sure, that constant columns are scaled to 1 and not to 0
            if sum(~any(Z)) > 0 
                Z(:, ~any(Z)) = 1;
            end 
        end
        
        function b = ridge_regression(~, y, X, k)
        % Custom implementation of the ridge regression function. 
        % Reason: The matlab ridge function ALWAYS standardized the
        % data. Foir this demonstrator we want to able to compare RR
        % also for the non-standardized case. 
        % This function is largely identical with matlabs ridge
        % implementation, except that the standardization was removed. 

            % Check that matrix (X) and left hand side (y) have compatible dimensions
            [n,p] = size(X);

            [n1,collhs] = size(y);
            if n~=n1 
                error(message('stats:ridge:InputSizeMismatch')); 
            end 

            if collhs ~= 1
                error(message('stats:ridge:InvalidData')); 
            end

            % Remove any missing values
            wasnan = (isnan(y) | any(isnan(X),2));
            if (any(wasnan))
               y(wasnan) = [];
               X(wasnan,:) = [];
               %n = length(y);
            end

            % The original matlab ridge function performs normalization
            % here
            Z = X-mean(X);
            y = y-mean(y);
            % Compute the ridge coefficient estimates using the technique of
            % adding pseudo observations having y=0 and X'X = k*I.
            pseudo = sqrt(k(1)) * eye(p);
            Zplus  = [Z;pseudo];
            yplus  = [y;zeros(p,1)];
            
            % Set up an array to hold the results
            nk = numel(k);
            
            % Compute the coefficient estimates
            b = Zplus\yplus;

            if nk>1
               % Fill in more entries after first expanding b.  We did not pre-
               % allocate b because we want the backslash above to determine its class.
               b(end,nk) = 0;
               for j=2:nk
                  Zplus(end-p+1:end,:) = sqrt(k(j)) * eye(p);
                  b(:,j) = Zplus\yplus;
               end
            end
        end
        
        function visibility_dataset(app)
        % Function to change visibility according to the 
        % selected dataset 
            switch app.DatasetDropDown.Value
                case 'Example'
                    vis = 'on';
                    visopp = 'off';
                    app.TrainTestSplitRatioSlider.Limits = [0.1 0.9];
                    app.TrainTestSplitRatioSlider.MajorTicks = [0.1 0.3 0.5 0.7 0.9];
                case 'Paracetamol'
                    vis = 'off';
                    visopp = 'on';
                    app.TrainTestSplitRatioSlider.Limits = [0.3 0.7];
                    app.TrainTestSplitRatioSlider.MajorTicks = [0.3 0.5 0.7];
                case 'LFP'
                    vis = 'off';
                    visopp = 'on';
                    app.TrainTestSplitRatioSlider.Limits = [0.1 0.9];
                    app.TrainTestSplitRatioSlider.MajorTicks = [0.1 0.3 0.5 0.7 0.9];
            end
            app.GroupingofDataCheckBox.Visible = visopp; 
            app.SigSEditField.Visible = vis;
            app.SigEEditField.Visible = vis;
            app.SigLEditField.Visible = vis;
            app.SigREditField.Visible = vis;
            app.SigSEditFieldLabel.Visible = vis;
            app.SigEEditFieldLabel.Visible = vis;
            app.SigLEditFieldLabel.Visible = vis;
            app.SigREditFieldLabel.Visible = vis;
            
            app.StartVEditField.Visible = vis;
            app.EndVEditField.Visible = vis;
            app.LeftVEditField.Visible = vis;
            app.RightVEditField.Visible = vis;
            app.StartVEditFieldLabel.Visible = vis;
            app.EndVEditFieldLabel.Visible = vis;
            app.LeftVEditFieldLabel.Visible = vis;
            app.RightVEditFieldLabel.Visible = vis;
            
            app.PositionSingnalSlider.Visible = vis;
            app.PositionofSignalLabel.Visible = vis;
            
            app.ExperimentsEditField.Visible = vis;
            app.relDatapointsEditField.Visible = vis;
            app.DatapointsEditField.Visible = vis;
            app.ExperimentsEditFieldLabel.Visible = vis;
            app.relDatapointsEditFieldLabel.Visible = vis;
            app.DatapointsEditFieldLabel.Visible = vis;
            
            app.DrawNewSampleButton.Visible = vis;
        end
        
        function visibility(app)
        % Function to change visibility according to the 
        % selected model 
            switch app.MethodDropDown.Value
                case 'PLS'
                    app.ComponentsEditField.Visible = 'on';
                    app.ComponentsEditFieldLabel.Visible = 'on';
                    app.RegularizationEditFieldLabel.Visible = 'off';
                    app.RegularizationEditField.Visible = 'off';
                    app.ENAlphaSlider.Visible = 'off';
                    app.ENAlphaSliderLabel.Visible = 'off';
            
                case 'CCA'
                    app.ComponentsEditField.Visible = 'off';
                    app.ComponentsEditFieldLabel.Visible = 'off';
                    app.RegularizationEditFieldLabel.Visible = 'off';
                    app.RegularizationEditField.Visible = 'off';
                    app.ENAlphaSlider.Visible = 'off';
                    app.ENAlphaSliderLabel.Visible = 'off';
                
                case 'PCR'
                    app.ComponentsEditField.Visible = 'on';
                    app.ComponentsEditFieldLabel.Visible = 'on';
                    app.RegularizationEditFieldLabel.Visible = 'off';
                    app.RegularizationEditField.Visible = 'off';
                    app.ENAlphaSlider.Visible = 'off';
                    app.ENAlphaSliderLabel.Visible = 'off';
  
                case 'RR'
                    app.RegularizationEditFieldLabel.Visible = 'on';
                    app.RegularizationEditField.Visible = 'on';
                    app.ComponentsEditField.Visible = 'off';
                    app.ComponentsEditFieldLabel.Visible = 'off';
                    app.ENAlphaSlider.Visible = 'off';
                    app.ENAlphaSliderLabel.Visible = 'off';
                
                case 'LASSO'
                    app.RegularizationEditFieldLabel.Visible = 'on';
                    app.RegularizationEditField.Visible = 'on';
                    app.ComponentsEditField.Visible = 'off';
                    app.ComponentsEditFieldLabel.Visible = 'off';
                    app.ENAlphaSlider.Visible = 'off';
                    app.ENAlphaSliderLabel.Visible = 'off';
                
                case 'EN'
                    app.RegularizationEditFieldLabel.Visible = 'on';
                    app.RegularizationEditField.Visible = 'on';
                    app.ComponentsEditField.Visible = 'off';
                    app.ComponentsEditFieldLabel.Visible = 'off';
                    app.ENAlphaSlider.Visible = 'on';
                    app.ENAlphaSliderLabel.Visible = 'on';
            end
        end
        
        function split_dataset(app, new)
        % Train Test Split
            group = app.GroupingofDataCheckBox.Value;
            fraction_test = app.TrainTestSplitRatioSlider.Value;            
            u_lim = train_test_split(app.DataObj, fraction_test, new, group);
            % Make sure that number of components is feasible
            app.ComponentsEditField.Limits = [1 u_lim];
        end
        
        
        function latent_variable_methods(app, plot)
        % This function is the core of the software tool and 
        % performs the (latent) variable regressions
        % PLS, CCA + Regression, PCA + Regression (PCR)
        % Ridge Regression (RR), LASSO and Elastic Net (EN)
        % Afterward it calls a function to plots the results

            r_xt = size(app.DataObj.X_test, 1);
            r_xtr = size(app.DataObj.X_train, 1);
            
            if app.StandardizeDataCheckBox.Value == true
                [X_train, mx, stdx] = normalize_train(app, app.DataObj.X_train);
                X_test = normalize_test(app, app.DataObj.X_test, mx, stdx);
            else
                X_train = app.DataObj.X_train;
                X_test = app.DataObj.X_test;
            end
            y_train = app.DataObj.y_train;
            y_test = app.DataObj.y_test;
            
            % Apply Latent Variable Methods
            % Canonical correlation analysis
            % Data is projected by performing cca on the training data.
            % Subsequently a least squares regression is performed on the
            % projected data
            visibility(app);
            
            switch app.MethodDropDown.Value
                case 'CCA'
                    %[A,B,r,U,V,stats] = canoncorr(X_train, y_train);
                    [A, B, ~, ~, ~, ~] = canoncorr(X_train, y_train);
                    coeff = A/B;
                    const_tr = ones(r_xtr, 1);
                    X_train_ = [const_tr, X_train*coeff];
                    b = regress(y_train-mean(y_train), X_train_); 
                    const_t = ones(r_xt, 1);
                    y_pred_test = [const_t, X_test*coeff]*b + mean(y_train);
                    y_pred_train = X_train_*b + mean(y_train);
                
                case 'PLS'
                    ncomp = int64(app.ComponentsEditField.Value);
                    % [XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(X_train, y_train, ncomp);
                    [~, ~, ~, ~, BETA, ~, ~, ~] = plsregress(X_train, y_train, ncomp);
                    coeff = BETA;
                    const = ones(r_xt, 1);
                    y_pred_test = [const, X_test]*BETA;
                    y_pred_train = [ones(r_xtr, 1), X_train]*BETA;
                
                case 'PCR'
                    % PCA + Regression: The data is first projected onto the
                    % #component proncipal component direction. Subsequently
                    % regression is performed 
                    [PCALoadings, PCAScores, ~] = pca(X_train,'Economy',false);
                    ncomp = int64(app.ComponentsEditField.Value);
                    betaPCR = regress(y_train-mean(y_train), PCAScores(:,1:ncomp));
                    betaPCR = PCALoadings(:,1:ncomp)*betaPCR;
                    betaPCR = [mean(y_train) - mean(X_train)*betaPCR; betaPCR];
                    coeff = betaPCR;
                    y_pred_test = [ones(r_xt,1) X_test]*betaPCR;
                    y_pred_train = [ones(r_xtr,1) X_train]*betaPCR;
            
                case 'RR'
                    % Ridge Regression, Least Square Regression with L2-norm
                    % penalty on the weights

                    k = app.RegularizationEditField.Value;
                    % RR has a standardization build in! The data is ALWAYS
                    % standardized --> Custom implementation for this tool.  
                    % B = ridge(y_train,X_train,k,0); %0 Scaled=0 restores coeff in org. space
                    % y_pred = B(1) + X_test*B(2:end);
                    % y_pred_train = B(1) + X_train*B(2:end);
                
                    B = ridge_regression(app, y_train,X_train,k);
                    y_pred_test = (X_test-mean(X_train))*B + mean(y_train);
                    y_pred_train = (X_train-mean(X_train))*B + mean(y_train);
                    coeff = B;
                         
                case 'LASSO'
                    % LASSO, Least Square Regression
                    k = app.RegularizationEditField.Value;
                    [B,FitInfo] = lasso(X_train, y_train, 'Lambda', k, 'Standardize',false);
                    y_pred_test = X_test*B + FitInfo.Intercept;
                    y_pred_train = X_train*B + FitInfo.Intercept;
                    coeff = B;
                
                case 'EN'
                    % Elastic Net, weighted combination of L1 & L2 Norm penalty on
                    % the weigts. 
                    k = app.RegularizationEditField.Value;
                    alpha = app.ENAlphaSlider.Value;
                    [B,FitInfo] = lasso(X_train, y_train, 'Lambda', k, 'Alpha', alpha, 'Standardize', false);
                    y_pred_test = X_test*B + FitInfo.Intercept;
                    y_pred_train = X_train*B + FitInfo.Intercept;
                    coeff = B;
            end
            if strcmp(app.DatasetDropDown.Value, 'LFP')
                y_train = 10.^y_train;
                y_test = 10.^y_test;
                y_pred_train = 10.^y_pred_train;
                y_pred_test = 10.^y_pred_test;
            end
                            
            [r2_train, rss_train, rmse_train] = fit_stats(app, y_pred_train, y_train);
            [r2_test, rss_test, rmse_test] = fit_stats(app, y_pred_test, y_test);
            % Save stats in a single variable
            stats = [r2_train, rss_train, rmse_train, r2_test, rss_test, rmse_test];
            
            if plot == true
                plot_figures(app, X_train, X_test, y_train, y_test, y_pred_train, y_pred_test, coeff, stats);
            end
            
        end
        
        function plot_figures(app, X_train, X_test, y_train, y_test, y_pred_train, y_pred_test, coeff, stats)
        % This function plots the data, regression coefficients
        % as well as the regression on the train and test data
            
            % Remove old plots
            cla(app.ax1)
            if app.HoldCheckBox.Value ~= true
                cla(app.ax2)
            end
            cla(app.UIAxesReg)
            cla(app.UIAxesRegTrain)

            % Plot the Data X  
            % Matlab warning: Let me know in case theres anything one can
            % do about it. The reason for this syntax is that ax1 and ax2 
            % shall be subplots in panel1 and thus the AppDesigner doesnt
            % recognize them as UIAxes, even though they are (Warning in
            % the editor, but no error when executing the code) 
            title(app.ax1, app.DatasetDropDown.Value)
            xlabel(app.ax1, app.DataObj.x_label_text)
            
            ylabel(app.ax1, app.DataObj.y_label_text)
            app.ax1.PlotBoxAspectRatio = [1.78515625 1 1];
            app.ax1.FontSize = 15;
            app.ax1.NextPlot = 'add';
            app.ax1.XGrid = 'on';
            app.ax1.YGrid = 'on';
            app.ax1.Box = 'on';
            app.ax1.XLim = [app.DataObj.x_values(1) app.DataObj.x_values(end)];
            [sxtr,~] = size(app.DataObj.X_train);
            [sxte,~] = size(app.DataObj.X_test);
            
            plot(app.ax1,app.DataObj.x_values, X_train(1,:), 'color', '#0072BD', 'DisplayName','Training');
            plot(app.ax1, app.DataObj.x_values, X_test(1,:),'color', '#D95319', 'DisplayName','Test');
            
            if strcmp(app.DatasetDropDown.Value, 'LFP')
                legend(app.ax1, 'Location', 'southeast', 'AutoUpdate','off');
            else
                legend(app.ax1, 'AutoUpdate','off');
            end
            
            for i=2:sxtr
                plot(app.ax1, app.DataObj.x_values, X_train(i,:), 'color', '#0072BD');
            end
            
            for i=2:sxte
               plot(app.ax1, app.DataObj.x_values, X_test(i,:),'color', '#D95319');
            end
            
            
            % Add text displaying rank fof the datamatrix
            if strcmp(app.DatasetDropDown.Value, 'LFP')
                y_pos = app.ax1.YLim(1)+0.075*(app.ax1.YLim(2)-app.ax1.YLim(1));
            else
                y_pos = app.ax1.YLim(2)-0.075*(app.ax1.YLim(2)-app.ax1.YLim(1));
            end
            
            rank_t = sprintf('Rank X = %.0f', rank(app.DataObj.X));
            app.rank_text = text(app.ax1, 0.03*(app.ax1.XLim(2)-app.ax1.XLim(1))+app.ax1.XLim(1), y_pos, rank_t, 'FontSize', 15);
            
            % Plot the regression coefficients
            title(app.ax2, 'Regression Coefficients')
            xlabel(app.ax2, app.DataObj.x_label_text)
            ylabel(app.ax2, char(946))
            app.ax2.PlotBoxAspectRatio = [1.78515625 1 1];
            app.ax2.FontSize = 15;
            app.ax2.NextPlot = 'add';
            app.ax2.XGrid = 'on';
            app.ax2.YGrid = 'on';
            app.ax2.Box = 'on';
            app.ax2.XLim = [app.DataObj.x_values(1) app.DataObj.x_values(end)];
            % Distinguish the case in which the model has a constant term
            if length(coeff) == size(app.DataObj.x_values, 2)
                % Model doesn't have a constant term
                plot(app.ax2, app.DataObj.x_values', coeff);
                %t = sprintf('c = %.2f', 0);
                %y_pos = app.ax2.YLim(2)-0.1*(app.ax2.YLim(2)-app.ax2.YLim(1));
            else
                % Model has a constant term
                plot(app.ax2, app.DataObj.x_values', coeff(2:end));
                %t = sprintf('c = %.2f', coeff(1));
                %y_pos = app.ax2.YLim(2)-0.1*(app.ax2.YLim(2)-app.ax2.YLim(1));
            end

            % Regression Plots
            % Train 
            line = linspace(min(min(y_train), min(y_pred_train)), max(max(y_train), max(y_pred_train)), 10);
            plot(app.UIAxesRegTrain, line, line)

            scatter(app.UIAxesRegTrain, y_train, y_pred_train, 30, 'filled');
            axis(app.UIAxesRegTrain, 'equal');
            app.UIAxesRegTrain.Box = 'on';
            
            text_train = sprintf('RMSE = %.2g', stats(3));
            y_pos_train = app.UIAxesRegTrain.YLim(2)-0.15*(app.UIAxesRegTrain.YLim(2)-app.UIAxesRegTrain.YLim(1));
            x_pos_train = app.UIAxesRegTrain.XLim(1)+0.08*(app.UIAxesRegTrain.XLim(2)-app.UIAxesRegTrain.XLim(1));
            text(app.UIAxesRegTrain, x_pos_train, y_pos_train, text_train, 'FontSize', 15);
            
            % Test
            line = linspace(min(min(y_test), min(y_pred_test)), max(max(y_test), max(y_pred_test)), 10);
            plot(app.UIAxesReg, line, line)
            title(app.UIAxesRegTrain, append(app.DataObj.output_text, ' on Training Data'))
            scatter(app.UIAxesReg, y_test, y_pred_test, 30, 'filled');
            axis(app.UIAxesReg, 'equal');
            app.UIAxesReg.Box = 'on';
 
            text_test = sprintf('RMSE = %.2g', stats(6));
            y_pos_test = app.UIAxesReg.YLim(2)-0.15*(app.UIAxesReg.YLim(2)-app.UIAxesReg.YLim(1));
            x_pos_test = app.UIAxesReg.XLim(1)+0.08*(app.UIAxesReg.XLim(2)-app.UIAxesReg.XLim(1));
            text(app.UIAxesReg, x_pos_test, y_pos_test, text_test, 'FontSize', 15);
            % Axes could be linked, but I dont think it's necessary.
            % There is still a bug in the line below. 
            % linkaxes([app.UIAxesReg app.UIAxesRegTrain],'xy');
        end
    end

    % Callbacks that handle component events
    methods (Access = private)

        % Code that executes after component creation
        function startupFcn(app)
        % Only called when starting the app 
            greek_letters(app);
            visibility_dataset(app);
            init(app);
            split_dataset(app, 1);
            latent_variable_methods(app, 1);
        end

        % Value changed function: ComponentsEditField, 
        % ENAlphaSlider, MethodDropDown, RegularizationEditField, 
        % StandardizeDataCheckBox
        function UpdatePlotsButtonPushed2(app, event)
        % Runs the regression method and plots in case method 
        % or hyperparameters are changed    
            latent_variable_methods(app, 1);
        end

        % Callback function: DatapointsEditField, 
        % DrawNewSampleButton, EndVEditField, ExperimentsEditField, 
        % LeftVEditField, PositionSingnalSlider, RightVEditField, 
        % SigEEditField, SigLEditField, SigREditField, 
        % SigSEditField, StartVEditField, relDatapointsEditField
        function DrawNewSampleButtonPushed(app, event)
        % Draws new samples, only triggered in the Example case, 
        % otherwise values are invisible
            init(app);
            split_dataset(app, 1);
            latent_variable_methods(app, 1);
        end

        % Value changed function: AddNoiseCheckBox, SNRXSlider, 
        % SNRySlider
        function SNRleftSliderValueChanged(app, event)
        % Triggers addition to noise 
            if app.AddNoiseCheckBox.Value
                create_signal(app.DataObj, app, 1);
                add_noise(app.DataObj, 1, app.SNRXSlider.Value, app.SNRySlider.Value);
            else
                create_signal(app.DataObj, app, 1);
            end
            split_dataset(app, 0);
            latent_variable_methods(app, 1);
        end

        % Value changed function: DatasetDropDown
        function DatasetDropDownValueChanged(app, event)
        % Adapts visivbility and inits the app in case the
        % dataset is changed    
            visibility_dataset(app);
            init(app); 
            split_dataset(app, 1);
            latent_variable_methods(app, 1);
        end

        % Value changing function: TrainTestSplitRatioSlider
        function TrainTestChange(app, event)
        % Calls th efunction to change the train test ration, avoids
        % reloadign the dataset 
            split_dataset(app, 0);
            latent_variable_methods(app, 1);
        end

        % Callback function: GroupingofDataCheckBox, 
        % ShuffleSplitsButton
        function ShuffleSplitsButtonPushed(app, event)
        % Makes a new random split of the data without reloading
        % the dataset
            split_dataset(app, 1);
            latent_variable_methods(app, 1);
        end
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Create LAVADEUIFigure and hide until all components are created
            app.LAVADEUIFigure = uifigure('Visible', 'off');
            app.LAVADEUIFigure.Color = [1 1 1];
            app.LAVADEUIFigure.Position = [100 100 1300 850];
            app.LAVADEUIFigure.Name = 'LAVADE';

            % Create UIAxesRegTrain
            app.UIAxesRegTrain = uiaxes(app.LAVADEUIFigure);
            title(app.UIAxesRegTrain, 'Regression on Training Data')
            xlabel(app.UIAxesRegTrain, 'y_{true}')
            ylabel(app.UIAxesRegTrain, 'y_{pred}')
            app.UIAxesRegTrain.PlotBoxAspectRatio = [1.03413654618474 1 1];
            app.UIAxesRegTrain.XGrid = 'on';
            app.UIAxesRegTrain.YGrid = 'on';
            app.UIAxesRegTrain.FontSize = 15;
            app.UIAxesRegTrain.MinorGridAlpha = 0.1;
            app.UIAxesRegTrain.NextPlot = 'add';
            app.UIAxesRegTrain.Position = [585 15 320 320];

            % Create UIAxesReg
            app.UIAxesReg = uiaxes(app.LAVADEUIFigure);
            title(app.UIAxesReg, 'Regression on Test Data')
            xlabel(app.UIAxesReg, 'y_{true}')
            ylabel(app.UIAxesReg, 'y_{pred}')
            app.UIAxesReg.PlotBoxAspectRatio = [1.03413654618474 1 1];
            app.UIAxesReg.XGrid = 'on';
            app.UIAxesReg.YGrid = 'on';
            app.UIAxesReg.FontSize = 15;
            app.UIAxesReg.NextPlot = 'add';
            app.UIAxesReg.Position = [944 15 320 320];

            % Create MethodDropDownLabel
            app.MethodDropDownLabel = uilabel(app.LAVADEUIFigure);
            app.MethodDropDownLabel.BackgroundColor = [1 1 1];
            app.MethodDropDownLabel.HorizontalAlignment = 'right';
            app.MethodDropDownLabel.FontSize = 16.5;
            app.MethodDropDownLabel.Position = [687 743 62 22];
            app.MethodDropDownLabel.Text = 'Method';

            % Create MethodDropDown
            app.MethodDropDown = uidropdown(app.LAVADEUIFigure);
            app.MethodDropDown.Items = {'PLS', 'PCR', 'CCA', 'RR', 'LASSO', 'EN'};
            app.MethodDropDown.ValueChangedFcn = createCallbackFcn(app, @UpdatePlotsButtonPushed2, true);
            app.MethodDropDown.FontSize = 16.5;
            app.MethodDropDown.BackgroundColor = [1 1 1];
            app.MethodDropDown.Position = [764 743 100 22];
            app.MethodDropDown.Value = 'PLS';

            % Create SNRXLabel
            app.SNRXLabel = uilabel(app.LAVADEUIFigure);
            app.SNRXLabel.FontSize = 15;
            app.SNRXLabel.Position = [730 439 61 22];
            app.SNRXLabel.Text = 'SNR: X';

            % Create SNRXSlider
            app.SNRXSlider = uislider(app.LAVADEUIFigure);
            app.SNRXSlider.Limits = [10 70];
            app.SNRXSlider.MajorTicks = [10 20 30 40 50 60 70];
            app.SNRXSlider.ValueChangedFcn = createCallbackFcn(app, @SNRleftSliderValueChanged, true);
            app.SNRXSlider.MinorTicks = [1 5 10 15 20 25 30 35 40 45 50 55 65];
            app.SNRXSlider.FontSize = 14;
            app.SNRXSlider.Position = [795 459 150 3];
            app.SNRXSlider.Value = 30;

            % Create TrainTestSplitRatioSliderLabel
            app.TrainTestSplitRatioSliderLabel = uilabel(app.LAVADEUIFigure);
            app.TrainTestSplitRatioSliderLabel.HorizontalAlignment = 'right';
            app.TrainTestSplitRatioSliderLabel.FontSize = 15;
            app.TrainTestSplitRatioSliderLabel.Position = [607 372 144 22];
            app.TrainTestSplitRatioSliderLabel.Text = 'Train-Test Split Ratio';

            % Create TrainTestSplitRatioSlider
            app.TrainTestSplitRatioSlider = uislider(app.LAVADEUIFigure);
            app.TrainTestSplitRatioSlider.Limits = [0.1 0.9];
            app.TrainTestSplitRatioSlider.MajorTicks = [0.1 0.3 0.5 0.7 0.9];
            app.TrainTestSplitRatioSlider.ValueChangingFcn = createCallbackFcn(app, @TrainTestChange, true);
            app.TrainTestSplitRatioSlider.FontSize = 15;
            app.TrainTestSplitRatioSlider.Position = [767 394 194 3];
            app.TrainTestSplitRatioSlider.Value = 0.7;

            % Create DatapointsEditFieldLabel
            app.DatapointsEditFieldLabel = uilabel(app.LAVADEUIFigure);
            app.DatapointsEditFieldLabel.HorizontalAlignment = 'right';
            app.DatapointsEditFieldLabel.FontSize = 16.5;
            app.DatapointsEditFieldLabel.Position = [988 603 95 22];
            app.DatapointsEditFieldLabel.Text = '#Datapoints';

            % Create DatapointsEditField
            app.DatapointsEditField = uieditfield(app.LAVADEUIFigure, 'numeric');
            app.DatapointsEditField.Limits = [10 10000];
            app.DatapointsEditField.ValueChangedFcn = createCallbackFcn(app, @DrawNewSampleButtonPushed, true);
            app.DatapointsEditField.Position = [1101 603 50 22];
            app.DatapointsEditField.Value = 50;

            % Create relDatapointsEditFieldLabel
            app.relDatapointsEditFieldLabel = uilabel(app.LAVADEUIFigure);
            app.relDatapointsEditFieldLabel.HorizontalAlignment = 'right';
            app.relDatapointsEditFieldLabel.FontSize = 16.5;
            app.relDatapointsEditFieldLabel.Position = [961 563 122 22];
            app.relDatapointsEditFieldLabel.Text = '#rel. Datapoints';

            % Create relDatapointsEditField
            app.relDatapointsEditField = uieditfield(app.LAVADEUIFigure, 'numeric');
            app.relDatapointsEditField.Limits = [2 5000];
            app.relDatapointsEditField.ValueChangedFcn = createCallbackFcn(app, @DrawNewSampleButtonPushed, true);
            app.relDatapointsEditField.Position = [1101 563 50 22];
            app.relDatapointsEditField.Value = 10;

            % Create ExperimentsEditFieldLabel
            app.ExperimentsEditFieldLabel = uilabel(app.LAVADEUIFigure);
            app.ExperimentsEditFieldLabel.HorizontalAlignment = 'right';
            app.ExperimentsEditFieldLabel.FontSize = 16.5;
            app.ExperimentsEditFieldLabel.Position = [976 643 107 22];
            app.ExperimentsEditFieldLabel.Text = '#Experiments';

            % Create ExperimentsEditField
            app.ExperimentsEditField = uieditfield(app.LAVADEUIFigure, 'numeric');
            app.ExperimentsEditField.Limits = [2 30000];
            app.ExperimentsEditField.ValueChangedFcn = createCallbackFcn(app, @DrawNewSampleButtonPushed, true);
            app.ExperimentsEditField.Position = [1101 643 50 22];
            app.ExperimentsEditField.Value = 30;

            % Create StartVEditFieldLabel
            app.StartVEditFieldLabel = uilabel(app.LAVADEUIFigure);
            app.StartVEditFieldLabel.HorizontalAlignment = 'right';
            app.StartVEditFieldLabel.FontSize = 16.5;
            app.StartVEditFieldLabel.Position = [675 664 56 22];
            app.StartVEditFieldLabel.Text = 'Start V';

            % Create StartVEditField
            app.StartVEditField = uieditfield(app.LAVADEUIFigure, 'numeric');
            app.StartVEditField.ValueChangedFcn = createCallbackFcn(app, @DrawNewSampleButtonPushed, true);
            app.StartVEditField.FontSize = 15;
            app.StartVEditField.Position = [737 664 33 22];
            app.StartVEditField.Value = 2;

            % Create SNRyLabel
            app.SNRyLabel = uilabel(app.LAVADEUIFigure);
            app.SNRyLabel.FontSize = 15;
            app.SNRyLabel.Position = [986 439 52 22];
            app.SNRyLabel.Text = 'SNR: y';

            % Create SNRySlider
            app.SNRySlider = uislider(app.LAVADEUIFigure);
            app.SNRySlider.Limits = [10 70];
            app.SNRySlider.MajorTicks = [1 10 20 30 40 50 60 70];
            app.SNRySlider.ValueChangedFcn = createCallbackFcn(app, @SNRleftSliderValueChanged, true);
            app.SNRySlider.MinorTicks = [1 5 10 15 20 25 30 35 40 45 50 55 65];
            app.SNRySlider.FontSize = 14;
            app.SNRySlider.Position = [1056 459 147 3];
            app.SNRySlider.Value = 30;

            % Create SigLEditFieldLabel
            app.SigLEditFieldLabel = uilabel(app.LAVADEUIFigure);
            app.SigLEditFieldLabel.HorizontalAlignment = 'right';
            app.SigLEditFieldLabel.FontSize = 16.5;
            app.SigLEditFieldLabel.Position = [795 584 38 22];
            app.SigLEditFieldLabel.Text = 'SigL';

            % Create SigLEditField
            app.SigLEditField = uieditfield(app.LAVADEUIFigure, 'numeric');
            app.SigLEditField.Limits = [0 Inf];
            app.SigLEditField.ValueChangedFcn = createCallbackFcn(app, @DrawNewSampleButtonPushed, true);
            app.SigLEditField.FontSize = 15;
            app.SigLEditField.Position = [838 584 33 22];
            app.SigLEditField.Value = 2;

            % Create SigREditFieldLabel
            app.SigREditFieldLabel = uilabel(app.LAVADEUIFigure);
            app.SigREditFieldLabel.HorizontalAlignment = 'right';
            app.SigREditFieldLabel.FontSize = 16.5;
            app.SigREditFieldLabel.Position = [791 544 41 22];
            app.SigREditFieldLabel.Text = 'SigR';

            % Create SigREditField
            app.SigREditField = uieditfield(app.LAVADEUIFigure, 'numeric');
            app.SigREditField.Limits = [0 Inf];
            app.SigREditField.ValueChangedFcn = createCallbackFcn(app, @DrawNewSampleButtonPushed, true);
            app.SigREditField.FontSize = 15;
            app.SigREditField.Position = [838 544 33 22];
            app.SigREditField.Value = 2;

            % Create DrawNewSampleButton
            app.DrawNewSampleButton = uibutton(app.LAVADEUIFigure, 'push');
            app.DrawNewSampleButton.ButtonPushedFcn = createCallbackFcn(app, @DrawNewSampleButtonPushed, true);
            app.DrawNewSampleButton.FontSize = 15;
            app.DrawNewSampleButton.FontWeight = 'bold';
            app.DrawNewSampleButton.Position = [1064 382 147 27];
            app.DrawNewSampleButton.Text = 'Draw New Sample';

            % Create LatentVariableMethodDemonstratorLabel
            app.LatentVariableMethodDemonstratorLabel = uilabel(app.LAVADEUIFigure);
            app.LatentVariableMethodDemonstratorLabel.FontSize = 28;
            app.LatentVariableMethodDemonstratorLabel.FontWeight = 'bold';
            app.LatentVariableMethodDemonstratorLabel.Position = [70 754 518 37];
            app.LatentVariableMethodDemonstratorLabel.Text = 'Latent Variable Method Demonstrator';

            % Create AddNoiseCheckBox
            app.AddNoiseCheckBox = uicheckbox(app.LAVADEUIFigure);
            app.AddNoiseCheckBox.ValueChangedFcn = createCallbackFcn(app, @SNRleftSliderValueChanged, true);
            app.AddNoiseCheckBox.Text = 'Add Noise';
            app.AddNoiseCheckBox.FontSize = 15;
            app.AddNoiseCheckBox.Position = [613 440 104 22];

            % Create ComponentsEditFieldLabel
            app.ComponentsEditFieldLabel = uilabel(app.LAVADEUIFigure);
            app.ComponentsEditFieldLabel.HorizontalAlignment = 'right';
            app.ComponentsEditFieldLabel.FontSize = 16.5;
            app.ComponentsEditFieldLabel.Position = [973 683 110 22];
            app.ComponentsEditFieldLabel.Text = '#Components';

            % Create ComponentsEditField
            app.ComponentsEditField = uieditfield(app.LAVADEUIFigure, 'numeric');
            app.ComponentsEditField.Limits = [1 100];
            app.ComponentsEditField.ValueChangedFcn = createCallbackFcn(app, @UpdatePlotsButtonPushed2, true);
            app.ComponentsEditField.Position = [1101 683 50 22];
            app.ComponentsEditField.Value = 1;

            % Create RegularizationEditFieldLabel
            app.RegularizationEditFieldLabel = uilabel(app.LAVADEUIFigure);
            app.RegularizationEditFieldLabel.HorizontalAlignment = 'right';
            app.RegularizationEditFieldLabel.FontSize = 16.5;
            app.RegularizationEditFieldLabel.Position = [973 722 110 22];
            app.RegularizationEditFieldLabel.Text = 'Regularization';

            % Create RegularizationEditField
            app.RegularizationEditField = uieditfield(app.LAVADEUIFigure, 'numeric');
            app.RegularizationEditField.Limits = [0 Inf];
            app.RegularizationEditField.ValueChangedFcn = createCallbackFcn(app, @UpdatePlotsButtonPushed2, true);
            app.RegularizationEditField.Position = [1101 723 50 22];
            app.RegularizationEditField.Value = 0.001;

            % Create ShuffleSplitsButton
            app.ShuffleSplitsButton = uibutton(app.LAVADEUIFigure, 'push');
            app.ShuffleSplitsButton.ButtonPushedFcn = createCallbackFcn(app, @ShuffleSplitsButtonPushed, true);
            app.ShuffleSplitsButton.FontSize = 15;
            app.ShuffleSplitsButton.FontWeight = 'bold';
            app.ShuffleSplitsButton.Position = [1082 348 109 27];
            app.ShuffleSplitsButton.Text = 'Shuffle Splits';

            % Create SigSEditFieldLabel
            app.SigSEditFieldLabel = uilabel(app.LAVADEUIFigure);
            app.SigSEditFieldLabel.HorizontalAlignment = 'right';
            app.SigSEditFieldLabel.FontSize = 16.5;
            app.SigSEditFieldLabel.Position = [790 664 40 22];
            app.SigSEditFieldLabel.Text = 'SigS';

            % Create SigSEditField
            app.SigSEditField = uieditfield(app.LAVADEUIFigure, 'numeric');
            app.SigSEditField.Limits = [0 Inf];
            app.SigSEditField.ValueChangedFcn = createCallbackFcn(app, @DrawNewSampleButtonPushed, true);
            app.SigSEditField.FontSize = 15;
            app.SigSEditField.Position = [838 664 33 22];

            % Create SigEEditFieldLabel
            app.SigEEditFieldLabel = uilabel(app.LAVADEUIFigure);
            app.SigEEditFieldLabel.HorizontalAlignment = 'right';
            app.SigEEditFieldLabel.FontSize = 16.5;
            app.SigEEditFieldLabel.Position = [791 624 39 22];
            app.SigEEditFieldLabel.Text = 'SigE';

            % Create SigEEditField
            app.SigEEditField = uieditfield(app.LAVADEUIFigure, 'numeric');
            app.SigEEditField.Limits = [0 Inf];
            app.SigEEditField.ValueChangedFcn = createCallbackFcn(app, @DrawNewSampleButtonPushed, true);
            app.SigEEditField.FontSize = 15;
            app.SigEEditField.Position = [838 624 33 22];

            % Create StandardizeDataCheckBox
            app.StandardizeDataCheckBox = uicheckbox(app.LAVADEUIFigure);
            app.StandardizeDataCheckBox.ValueChangedFcn = createCallbackFcn(app, @UpdatePlotsButtonPushed2, true);
            app.StandardizeDataCheckBox.Text = {'Standardize Data'; ''};
            app.StandardizeDataCheckBox.FontSize = 16.5;
            app.StandardizeDataCheckBox.Position = [704 703 149 22];

            % Create Panel
            app.Panel = uipanel(app.LAVADEUIFigure);
            app.Panel.ForegroundColor = [1 1 1];
            app.Panel.BorderType = 'none';
            app.Panel.BackgroundColor = [1 1 1];
            app.Panel.Position = [10 -15 560 730];

            % Create HoldCheckBox
            app.HoldCheckBox = uicheckbox(app.Panel);
            app.HoldCheckBox.Text = 'Hold';
            app.HoldCheckBox.Position = [454 298 47 22];

            % Create ENAlphaSliderLabel
            app.ENAlphaSliderLabel = uilabel(app.LAVADEUIFigure);
            app.ENAlphaSliderLabel.HorizontalAlignment = 'right';
            app.ENAlphaSliderLabel.FontSize = 15;
            app.ENAlphaSliderLabel.Position = [904 775 68 22];
            app.ENAlphaSliderLabel.Text = 'EN Alpha';

            % Create ENAlphaSlider
            app.ENAlphaSlider = uislider(app.LAVADEUIFigure);
            app.ENAlphaSlider.Limits = [1e-07 1];
            app.ENAlphaSlider.MajorTicks = [0.05 0.25 0.5 0.75 1];
            app.ENAlphaSlider.ValueChangedFcn = createCallbackFcn(app, @UpdatePlotsButtonPushed2, true);
            app.ENAlphaSlider.MinorTicks = [0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1];
            app.ENAlphaSlider.FontSize = 13;
            app.ENAlphaSlider.Position = [993 794 150 3];
            app.ENAlphaSlider.Value = 0.5;

            % Create DatasetDropDownLabel
            app.DatasetDropDownLabel = uilabel(app.LAVADEUIFigure);
            app.DatasetDropDownLabel.BackgroundColor = [1 1 1];
            app.DatasetDropDownLabel.HorizontalAlignment = 'right';
            app.DatasetDropDownLabel.FontSize = 16.5;
            app.DatasetDropDownLabel.Position = [687 784 62 22];
            app.DatasetDropDownLabel.Text = 'Dataset';

            % Create DatasetDropDown
            app.DatasetDropDown = uidropdown(app.LAVADEUIFigure);
            app.DatasetDropDown.Items = {'Example', 'Paracetamol', 'LFP'};
            app.DatasetDropDown.ValueChangedFcn = createCallbackFcn(app, @DatasetDropDownValueChanged, true);
            app.DatasetDropDown.FontSize = 16.5;
            app.DatasetDropDown.BackgroundColor = [1 1 1];
            app.DatasetDropDown.Position = [764 784 100 22];
            app.DatasetDropDown.Value = 'Example';

            % Create EndVEditFieldLabel
            app.EndVEditFieldLabel = uilabel(app.LAVADEUIFigure);
            app.EndVEditFieldLabel.HorizontalAlignment = 'right';
            app.EndVEditFieldLabel.FontSize = 16.5;
            app.EndVEditFieldLabel.Position = [673 624 58 22];
            app.EndVEditFieldLabel.Text = 'End   V';

            % Create EndVEditField
            app.EndVEditField = uieditfield(app.LAVADEUIFigure, 'numeric');
            app.EndVEditField.ValueChangedFcn = createCallbackFcn(app, @DrawNewSampleButtonPushed, true);
            app.EndVEditField.FontSize = 15;
            app.EndVEditField.Position = [737 624 33 22];
            app.EndVEditField.Value = -5;

            % Create LeftVEditFieldLabel
            app.LeftVEditFieldLabel = uilabel(app.LAVADEUIFigure);
            app.LeftVEditFieldLabel.HorizontalAlignment = 'right';
            app.LeftVEditFieldLabel.FontSize = 16.5;
            app.LeftVEditFieldLabel.Position = [670 584 62 22];
            app.LeftVEditFieldLabel.Text = 'Left    V';

            % Create LeftVEditField
            app.LeftVEditField = uieditfield(app.LAVADEUIFigure, 'numeric');
            app.LeftVEditField.ValueChangedFcn = createCallbackFcn(app, @DrawNewSampleButtonPushed, true);
            app.LeftVEditField.FontSize = 15;
            app.LeftVEditField.Position = [737 584 33 22];
            app.LeftVEditField.Value = -1;

            % Create RightVEditFieldLabel
            app.RightVEditFieldLabel = uilabel(app.LAVADEUIFigure);
            app.RightVEditFieldLabel.HorizontalAlignment = 'right';
            app.RightVEditFieldLabel.FontSize = 16.5;
            app.RightVEditFieldLabel.Position = [668 544 64 22];
            app.RightVEditFieldLabel.Text = 'Right  V';

            % Create RightVEditField
            app.RightVEditField = uieditfield(app.LAVADEUIFigure, 'numeric');
            app.RightVEditField.ValueChangedFcn = createCallbackFcn(app, @DrawNewSampleButtonPushed, true);
            app.RightVEditField.FontSize = 15;
            app.RightVEditField.Position = [737 544 33 22];
            app.RightVEditField.Value = -4;

            % Create PositionofSignalLabel
            app.PositionofSignalLabel = uilabel(app.LAVADEUIFigure);
            app.PositionofSignalLabel.HorizontalAlignment = 'right';
            app.PositionofSignalLabel.FontSize = 15;
            app.PositionofSignalLabel.Position = [691 495 122 22];
            app.PositionofSignalLabel.Text = 'Position of Signal';

            % Create PositionSingnalSlider
            app.PositionSingnalSlider = uislider(app.LAVADEUIFigure);
            app.PositionSingnalSlider.Limits = [0 1];
            app.PositionSingnalSlider.ValueChangedFcn = createCallbackFcn(app, @DrawNewSampleButtonPushed, true);
            app.PositionSingnalSlider.FontSize = 14;
            app.PositionSingnalSlider.Position = [841 514 259 3];
            app.PositionSingnalSlider.Value = 0.5;

            % Create GroupingofDataCheckBox
            app.GroupingofDataCheckBox = uicheckbox(app.LAVADEUIFigure);
            app.GroupingofDataCheckBox.ValueChangedFcn = createCallbackFcn(app, @ShuffleSplitsButtonPushed, true);
            app.GroupingofDataCheckBox.Text = 'Grouping of Data';
            app.GroupingofDataCheckBox.FontSize = 15;
            app.GroupingofDataCheckBox.Position = [613 406 137 22];
            app.GroupingofDataCheckBox.Value = true;

            % Show the figure after all components are created
            app.LAVADEUIFigure.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = lavade_exported

            % Create UIFigure and components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.LAVADEUIFigure)

            % Execute the startup function
            runStartupFcn(app, @startupFcn)

            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.LAVADEUIFigure)
        end
    end
end