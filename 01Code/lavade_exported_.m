classdef lavade_exported < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        LAVADEUIFigure                matlab.ui.Figure
        UIAxesReg                     matlab.ui.control.UIAxes
        MethodDropDownLabel           matlab.ui.control.Label
        MethodDropDown                matlab.ui.control.DropDown
        SNRLeftSliderLabel            matlab.ui.control.Label
        SNRLeftSlider                 matlab.ui.control.Slider
        TrainTestSplitSlider_3Label   matlab.ui.control.Label
        TrainTestSplitSlider          matlab.ui.control.Slider
        DatapointsEditFieldLabel      matlab.ui.control.Label
        DatapointsEditField           matlab.ui.control.NumericEditField
        relevantDatapointsEditFieldLabel  matlab.ui.control.Label
        relevantDatapointsEditField   matlab.ui.control.NumericEditField
        ExperimentsEditFieldLabel     matlab.ui.control.Label
        ExperimentsEditField          matlab.ui.control.NumericEditField
        StartVEditFieldLabel          matlab.ui.control.Label
        StartVEditField               matlab.ui.control.NumericEditField
        EndVLabel                     matlab.ui.control.Label
        EndVEditField                 matlab.ui.control.NumericEditField
        SignalSliderLabel             matlab.ui.control.Label
        SignalSlider                  matlab.ui.control.Slider
        RightSliderLabel              matlab.ui.control.Label
        RightSlider                   matlab.ui.control.Slider
        LeftVLabel                    matlab.ui.control.Label
        LeftVEditField                matlab.ui.control.NumericEditField
        SigLEditFieldLabel            matlab.ui.control.Label
        SigLEditField                 matlab.ui.control.NumericEditField
        RightVLabel                   matlab.ui.control.Label
        RightVEditField               matlab.ui.control.NumericEditField
        SigREditFieldLabel            matlab.ui.control.Label
        SigREditField                 matlab.ui.control.NumericEditField
        DrawNewSampleButton           matlab.ui.control.Button
        LatentVariableMethodDemonstratorLabel  matlab.ui.control.Label
        PositionofSignalLabel         matlab.ui.control.Label
        PositionSingnalSlider         matlab.ui.control.Slider
        NoiseCheckBox                 matlab.ui.control.CheckBox
        ComponentsEditFieldLabel      matlab.ui.control.Label
        ComponentsEditField           matlab.ui.control.NumericEditField
        RegularizationEditFieldLabel  matlab.ui.control.Label
        RegularizationEditField       matlab.ui.control.NumericEditField
        UpdatePlotsButton             matlab.ui.control.Button
        UIAxesRegTrain                matlab.ui.control.UIAxes
        SigSEditFieldLabel            matlab.ui.control.Label
        SigSEditField                 matlab.ui.control.NumericEditField
        SigEEditFieldLabel            matlab.ui.control.Label
        SigEEditField                 matlab.ui.control.NumericEditField
        StandardizeInputsCheckBox     matlab.ui.control.CheckBox
        Panel                         matlab.ui.container.Panel
        HoldCheckBox                  matlab.ui.control.CheckBox
        ENAlphaSliderLabel            matlab.ui.control.Label
        ENAlphaSlider                 matlab.ui.control.Slider
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LAVADE (Latent Variable Demonstrator Software)
% Developed with Matlab 2019b
% Author: Joachim Schaeffer
% Email:  joachim.schaeffer@posteo.de
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  

    % Changed access to public to be able to run my case studies
    % from matlab directly
    properties (Access = public)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Varibales that can be accessed in all functions below, via
        % app.VarName
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % Variables related to the data X and z 
        X;
        X_std;
        X_train; 
        X_test; 
        y;
        y_train;
        y_pred_train;
        y_test;
        y_pred_test;
        % coefficients
        coeff;
        % individual sections of X
        rand_uc_l;       %  random UnCorrrelated section Left
        rand_uc_r;       %  random UnCorrrelated section Right
        noise_uc_l;      %  noise UnCorrelated Left 
        noise_uc_r;      %  noise UnCorrelated Right 
        signal;          %  "Signal section" y = slope in this section
        noise_signal;    %  Noise of the signal section 
        start_value;     %  Mean of first datapoint
        final_value;     %  Mean of last datapoint 
        target_left;     %  Mean of datapoint after which the signal section starts 
        target_right;    %  Mean of last datapoint of the signal section 
        measurements;    %  # of measurements, #of rows in the X matrix   
        % Fitting statistics
        r2_train; 
        r2_test; 
        rss_train;
        rss_test; 
        rmse_train; 
        rmse_test;
        stats;
        % Subplots in the panel section
        ax1;
        ax2;
        rank_text;       % Text for displaying the rank in ax1
        const_term_text; % Text for the constant term in ax2 
        
    end
    
    % Changed access to public to be able to run my case studies
    % from matlab directly
    methods (Access = public)
        
    function init(app)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % This function is executed when starting the app & when drawing
            % new examples. Thus, it sets up the workspace and initializes 
            % variables, & runs the first regression based on the initialization
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            % Cretae Panel Subplots 
            app.Panel.AutoResizeChildren = 'off';
            app.ax1 = subplot(2,1,1,'Parent',app.Panel);
            app.ax2 = subplot(2,1,2,'Parent',app.Panel);
            % Greek letters in text fields must be initialized here 
            app.SigLEditFieldLabel.Text = char(963);
            app.SigREditFieldLabel.Text = char(963);
            app.SigSEditFieldLabel.Text = char(963);
            app.SigEEditFieldLabel.Text = char(963);
            app.LeftVLabel.Text = strcat("Left  ", char(956));
            app.RightVLabel.Text = strcat("Right  ", char(956));
            app.StartVEditFieldLabel.Text = strcat("Start  ", char(956));
            app.EndVLabel.Text = strcat("End  ", char(956));
            app.ENAlphaSliderLabel.Text = strcat("EN ", char(945));
            
            % Generate Examples
            app.measurements = app.ExperimentsEditField.Value;
            sigma_start = app.SigSEditField.Value;
            app.start_value = normrnd(app.StartVEditField.Value, sigma_start, app.measurements, 1);

            sigma_end = app.SigEEditField.Value;
            app.final_value = normrnd(app.EndVEditField.Value, sigma_end, app.measurements, 1);

            value_left = app.LeftVEditField.Value;
            sigma_left = app.SigLEditField.Value;
            
            value_right = app.RightVEditField.Value;
            sigma_right = app.SigREditField.Value;
            
            app.target_left = normrnd(value_left, sigma_left, app.measurements, 1);
            app.target_right = normrnd(value_right, sigma_right, app.measurements, 1);
            
            data_points_signal = app.relevantDatapointsEditField.Value;
            
            % The y we want to predict: Slope of the signal section
            % Here you can also insert different responses
            app.y = (app.target_right-app.target_left)/data_points_signal;
            
            % Cresating the X vector is outsourced, because this function
            % is also called when changing the position of the signal; In
            % that case we don't want to draw new samples. 
            create_signal(app);
        end
    
        function init_light(app)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Function intended to use only when class is exported for
            % monte carlo experiments
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            latent_variable_methods(app);
            % plot_figures(app);
        end
    
        function [r2, rss, rmse] = fit_stats(~, y, yfit)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Helper function to calculate the R^2 value. 
            % Syntax partially copied from 
            % https://ch.mathworks.com/help/matlab/data_analysis/linear-regression.html#f1-15010
            % https://ch.mathworks.com/matlabcentral/answers/4064-rmse-root-mean-square-error
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            yresid = y - yfit;
            SSresid = sum(yresid.^2);
            SStotal = (length(y)-1) * var(y);
            r2 = 1 - SSresid/SStotal;
            rss = SSresid;
            rmse = sqrt(mean(yresid.^2));
        end
        
        function [Z, mx, stdx] = normalize_train(~, X)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Helper function to normalize the data.
            % Syntax copied from Matlab 'ridge' function. 
            % Reason: The normalize function in the Matlab 2019b release
            % does not return mean and std. Furthermore, normalize in 2019b
            % retunrs a column of Nan's in case the column is constant
            % Normalize the columns of X to mean zero, and standard deviation one.
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Helper function to normalize the test data based on the train
            % data stats.
            % Based on syntax from Matlab 'ridge' function. 
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            Z = (X-mx)./stdx;
            % Make sure, that constant columns are scaled to 1 and not to 0
            if sum(~any(Z)) > 0 
                Z(:, ~any(Z)) = 1;
            end 
        end
        
        function b = ridge_regression(~, y, X, k)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Custom implementation of the ridge regression function. 
            % Reason: The matlab ridge function ALWAYS standardized the
            % data. Foir this demonstrator we want to able to compare RR
            % also for the non-standardized case. 
            % This function is largely identical with matlabs ridge
            % implementation, except that the standardization was removed. 
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
               n = length(y);
            end

            % The original matlab ridge function performs normalization
            % here
            Z = X;

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
        
        function visibility(app, mode)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Helper fuction for readibility
            % makes relevant items visble and unvisible for the respective
            % latent variable approach
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            if strcmp(mode, 'PLS')
                app.ComponentsEditField.Visible = 'on';
                app.ComponentsEditFieldLabel.Visible = 'on';
                app.RegularizationEditFieldLabel.Visible = 'off';
                app.RegularizationEditField.Visible = 'off';
                app.ENAlphaSlider.Visible = 'off';
                app.ENAlphaSliderLabel.Visible = 'off';
            
            elseif strcmp(mode, 'CCA')
                app.ComponentsEditField.Visible = 'off';
                app.ComponentsEditFieldLabel.Visible = 'off';
                app.RegularizationEditFieldLabel.Visible = 'off';
                app.RegularizationEditField.Visible = 'off';
                app.ENAlphaSlider.Visible = 'off';
                app.ENAlphaSliderLabel.Visible = 'off';
                
            elseif strcmp(mode, 'PCR')
                app.ComponentsEditField.Visible = 'on';
                app.ComponentsEditFieldLabel.Visible = 'on';
                app.RegularizationEditFieldLabel.Visible = 'off';
                app.RegularizationEditField.Visible = 'off';
                app.ENAlphaSlider.Visible = 'off';
                app.ENAlphaSliderLabel.Visible = 'off';
  
            elseif strcmp(mode, 'RR')
                app.RegularizationEditFieldLabel.Visible = 'on';
                app.RegularizationEditField.Visible = 'on';
                app.ComponentsEditField.Visible = 'off';
                app.ComponentsEditFieldLabel.Visible = 'off';
                app.ENAlphaSlider.Visible = 'off';
                app.ENAlphaSliderLabel.Visible = 'off';
                
            elseif strcmp(mode, 'LASSO')
                app.RegularizationEditFieldLabel.Visible = 'on';
                app.RegularizationEditField.Visible = 'on';
                app.ComponentsEditField.Visible = 'off';
                app.ComponentsEditFieldLabel.Visible = 'off';
                app.ENAlphaSlider.Visible = 'off';
                app.ENAlphaSliderLabel.Visible = 'off';
                
            elseif strcmp(mode, 'EN')
                app.RegularizationEditFieldLabel.Visible = 'on';
                app.RegularizationEditField.Visible = 'on';
                app.ComponentsEditField.Visible = 'off';
                app.ComponentsEditFieldLabel.Visible = 'off';
                app.ENAlphaSlider.Visible = 'on';
                app.ENAlphaSliderLabel.Visible = 'on';
            end
        end
        
        function create_signal(app)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % This function crestes the data matrix X consisting of
            % # of Experiments rows
            % # of Datapoints columns =
            % #"Uncorrleated points left" + #Sinal points + #"Uncorrelated points right")
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            % create noisy uncorrelated data on the left and right of the
            % signal section             
            noise_l = app.SNRLeftSlider.Value;
            noise_r = app.RightSlider.Value;          
            noise_s = app.SignalSlider.Value;
            
            datapoints = app.DatapointsEditField.Value;
            data_points_signal = app.relevantDatapointsEditField.Value;
            data_points_left = int64(app.PositionSingnalSlider.Value*(datapoints-data_points_signal));     
            data_points_right = datapoints - (data_points_left+data_points_signal);
            
            app.rand_uc_l = zeros(app.measurements, data_points_left+1);
            app.rand_uc_r = zeros(app.measurements, data_points_right+1);
            app.noise_uc_l = zeros(app.measurements, data_points_left+1);
            app.noise_uc_r = zeros(app.measurements, data_points_right+1);
            app.signal = zeros(app.measurements, data_points_signal);
            app.noise_signal = zeros(app.measurements, data_points_signal);
            
            for i=1:app.measurements
                app.rand_uc_l(i, :) = linspace(app.start_value(i), app.target_left(i), data_points_left+1);
                app.noise_uc_l(i, :) = awgn(app.rand_uc_l(i, :), noise_l, 'measured');
                
                app.rand_uc_r(i, :) = linspace(app.target_right(i), app.final_value(i), data_points_right+1);
                app.noise_uc_r(i, :) = awgn(app.rand_uc_r(i, :), noise_r, 'measured');

                app.signal(i, :) =  linspace(app.target_left(i), app.target_right(i), data_points_signal);
                app.noise_signal(i, :) = awgn(app.signal(i, :), noise_s, 'measured');
            end
            
            % Build the datamatrix X
            if app.NoiseCheckBox.Value == true
                app.X = [app.noise_uc_l(:, 1:end-1), app.noise_signal, app.noise_uc_r(:, 2:end)];
            else
                app.X = [app.rand_uc_l(:, 1:end-1), app.signal, app.rand_uc_r(:, 2:end)];
            end
       
            % Train Test Split
            fraction_test = app.TrainTestSplitSlider.Value;            
            split_ind = int64(fraction_test*app.measurements);
            
            app.X_train = app.X(1:split_ind, :);
            app.X_test = app.X(split_ind+1:end, :);

            app.y_train = app.y(1:split_ind, :);
            app.y_test = app.y(split_ind+1:end, :);
            
            % Make sure that number of components is feasible
            app.ComponentsEditField.Limits = [1 size(app.X_train,1)-1];
            
            % Apply the latent varibel method 
            latent_variable_methods(app);
            
            % Plot everything
            plot_figures(app);
        end
        
        function latent_variable_methods(app)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % This function is the core of the software tool and performs
            % the (latent) variable regressions
            % PLS, CCA + Regression, PCA + Regression (PCR)
            % Ridge Regression (RR), LASSO and Elastic Net (EN)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            r_xt = size(app.X_test, 1);
            r_xtr = size(app.X_train, 1);
            
            if app.StandardizeInputsCheckBox.Value == true
                [app.X_train, mx, stdx] = normalize_train(app, app.X_train);
                app.X_test = normalize_test(app, app.X_test, mx, stdx);
                app.X_std = normalize_test(app, app.X, mx, stdx);
            end
            
            % Apply Latent Variable Methods
            % Canonical correlation analysis
            % Data is projected by performing cca on the training data.
            % Subsequently a least squares regression is performed on the
            % projected data
            if strcmp(app.MethodDropDown.Value, 'CCA')
                visibility(app, 'CCA');
                [A,B,r,U,V,stats] = canoncorr(app.X_train, app.y_train);
                %[A, B, ~, ~, ~, ~] = canoncorr(X_train, y_train);
                app.coeff = A/B;
                const_tr = ones(r_xtr, 1);
                X_train_ = [const_tr, app.X_train*app.coeff];
                b = regress(app.y_train-mean(app.y_train), X_train_); 
                const_t = ones(r_xt, 1);
                app.y_pred_test = [const_t, app.X_test*app.coeff]*b + mean(app.y_train);
                app.y_pred_train = X_train_*b + mean(app.y_train);
                
            % PLS regression based on the matlab implementation of PLS   
            elseif strcmp(app.MethodDropDown.Value, 'PLS')
                visibility(app, 'PLS');
                ncomp = int64(app.ComponentsEditField.Value);
                % [XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(X_train, y_train, ncomp);
                [~, ~, ~, ~, BETA, ~, ~, ~] = plsregress(app.X_train, app.y_train, ncomp);
                app.coeff = BETA;
                const = ones(r_xt, 1);
                app.y_pred_test = [const, app.X_test]*BETA;
                app.y_pred_train = [ones(r_xtr, 1), app.X_train]*BETA;
            
            % PCA + Regression: The data is first projected onto the
            % #component proncipal component direction. Subsequently
            % regression is performed 
            elseif strcmp(app.MethodDropDown.Value, 'PCR')
                visibility(app, 'PCR');
                % [PCALoadings, PCAScores, PCAVar] = pca(X_train,'Economy',false);
                [PCALoadings, PCAScores, ~] = pca(app.X_train,'Economy',false);
                ncomp = int64(app.ComponentsEditField.Value);
                betaPCR = regress(app.y_train-mean(app.y_train), PCAScores(:,1:ncomp));
                betaPCR = PCALoadings(:,1:ncomp)*betaPCR;
                betaPCR = [mean(app.y_train) - mean(app.X_train)*betaPCR; betaPCR];
                app.coeff = betaPCR;
                app.y_pred_test = [ones(r_xt,1) app.X_test]*betaPCR;
                app.y_pred_train = [ones(r_xtr,1) app.X_train]*betaPCR;
            
            % Ridge Regression, Least Square Regression with L2-norm
            % penalty on the weights
            elseif strcmp(app.MethodDropDown.Value, 'RR')
                visibility(app, 'RR');
                k = app.RegularizationEditField.Value;
                % RR has a standardization build in! The data is ALWAYS
                % standardized --> Custom implementation for this tool.  
                % B = ridge(y_train,X_train,k,0); %0 Scaled=0 restores coeff in org. space
                % y_pred = B(1) + X_test*B(2:end);
                % y_pred_train = B(1) + X_train*B(2:end);
                
                B = ridge_regression(app, app.y_train,app.X_train,k);
                app.y_pred_test = app.X_test*B;
                app.y_pred_train = app.X_train*B;
                app.coeff = B;
                
            % LASSO, Least Square Regression with L1-norm
            % penalty on the weights   
            elseif strcmp(app.MethodDropDown.Value, 'LASSO')
                visibility(app, 'LASSO');
                k = app.RegularizationEditField.Value;
                [B,FitInfo] = lasso(app.X_train, app.y_train, 'Lambda', k, 'Standardize',false);
                app.y_pred_test = app.X_test*B + FitInfo.Intercept;
                app.y_pred_train = app.X_train*B + FitInfo.Intercept;
                app.coeff = B;
                
            % Elastic Net, weighted combination of L1 & L2 Norm penalty on
            % the weigts. 
            elseif strcmp(app.MethodDropDown.Value, 'EN')
                visibility(app, 'EN');
                k = app.RegularizationEditField.Value;
                alpha = app.ENAlphaSlider.Value;
                [B,FitInfo] = lasso(app.X_train, app.y_train, 'Lambda', k, 'Alpha', alpha, 'Standardize', false);
                app.y_pred_test = app.X_test*B + FitInfo.Intercept;
                app.y_pred_train = app.X_train*B + FitInfo.Intercept;
                app.coeff = B;

            end
            [app.r2_train, app.rss_train, app.rmse_train] = fit_stats(app, app.y_pred_train, app.y_train);
            [app.r2_test, app.rss_test, app.rmse_test] = fit_stats(app, app.y_pred_test, app.y_test);
            % Save stats in a single variable
            app.stats = [app.r2_train, app.rss_train, app.rmse_train, app.r2_test, app.rss_test, app.rmse_test];
        end
        
        function plot_figures(app)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % This function plots the data, regression coefficients
            % as well as the regression on the train and test data
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
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
            title(app.ax1, 'Data')
            xlabel(app.ax1, 'idx')
            ylabel(app.ax1, 'X')
            app.ax1.PlotBoxAspectRatio = [1.78515625 1 1];
            app.ax1.FontSize = 15;
            app.ax1.NextPlot = 'add';
            app.ax1.XGrid = 'on';
            app.ax1.YGrid = 'on';
            
            [r,~] = size(app.X);
            if app.StandardizeInputsCheckBox.Value == true
                for i=1:r
                    plot(app.ax1, app.X_std(i,:));
                end
            else
                for i=1:r
                    plot(app.ax1, app.X(i,:));
                end
            end
            
            % Add text displaying rank fof the datamatrix
            y_pos = app.ax1.YLim(2)-0.05*(app.ax1.YLim(2)-app.ax1.YLim(1));
            rank_t = sprintf('Rank X = %.0f', rank(app.X));
            app.rank_text = text(app.ax1, 2, y_pos, rank_t, 'FontSize', 15);
            
            
            % Plot the regression coefficients
            title(app.ax2, 'Regression Coefficients')
            xlabel(app.ax2, 'idx')
            ylabel(app.ax2, char(946))
            app.ax2.PlotBoxAspectRatio = [1.78515625 1 1];
            app.ax2.FontSize = 15;
            app.ax2.NextPlot = 'add';
            app.ax2.XGrid = 'on';
            app.ax2.YGrid = 'on';

            % Distinguish the case in which the model has a constant term
            if length(app.coeff) == size(app.X, 2)
                % Model doesn't have a constant term
                plot(app.ax2, app.coeff);
                t = sprintf('c = %.2f', 0);
                y_pos = app.ax2.YLim(2)-0.2*(app.ax2.YLim(2)-app.ax2.YLim(1));
            else
                % Model has a constant term
                plot(app.ax2, app.coeff(2:end));
                t = sprintf('c = %.2f', app.coeff(1));
                y_pos = app.ax2.YLim(2)-0.2*(app.ax2.YLim(2)-app.ax2.YLim(1));
            end

            %text = text(app.ax2, 2, y_pos, t, 'FontSize', 15);
            % Display conatnt term only when hold is off
            if app.HoldCheckBox.Value ~= true
                app.const_term_text = text(app.ax2, 2, y_pos, t, 'FontSize', 15);
            else
                delete(app.const_term_text)
            end
            
            % Regression Plots
            % Train 
            line = linspace(min(min(app.y_train), min(app.y_pred_train)), max(max(app.y_train), max(app.y_pred_train)), 10);
            plot(app.UIAxesRegTrain, line, line)
            
            scatter(app.UIAxesRegTrain, app.y_pred_train, app.y_train);
            axis(app.UIAxesRegTrain, 'equal');
            
            text_train = sprintf('R^2 = %.2f', app.r2_train);
            y_pos_train = app.UIAxesRegTrain.YLim(2)-0.15*(app.UIAxesRegTrain.YLim(2)-app.UIAxesRegTrain.YLim(1));
            x_pos_train = app.UIAxesRegTrain.XLim(1)+0.08*(app.UIAxesRegTrain.XLim(2)-app.UIAxesRegTrain.XLim(1));
            text(app.UIAxesRegTrain, x_pos_train, y_pos_train, text_train, 'FontSize', 15);
            
            % Test
            line = linspace(min(min(app.y_test), min(app.y_pred_test)), max(max(app.y_test), max(app.y_pred_test)), 10);
            plot(app.UIAxesReg, line, line)
            
            scatter(app.UIAxesReg, app.y_pred_test, app.y_test);
            axis(app.UIAxesReg, 'equal');
            
            text_test = sprintf('R^2 = %.2f', app.r2_test);
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
            init(app);
        end

        % Callback function: ComponentsEditField, ENAlphaSlider, 
        % MethodDropDown, RegularizationEditField, 
        % StandardizeInputsCheckBox, UpdatePlotsButton
        function UpdatePlotsButtonPushed2(app, event)
            latent_variable_methods(app);
            plot_figures(app);
        end

        % Callback function: DatapointsEditField, 
        % DrawNewSampleButton, EndVEditField, ExperimentsEditField, 
        % LeftVEditField, RightVEditField, SigEEditField, 
        % SigLEditField, SigREditField, SigSEditField, 
        % StartVEditField, relevantDatapointsEditField
        function DrawNewSampleButtonPushed(app, event)
            init(app)
        end

        % Value changed function: NoiseCheckBox, 
        % PositionSingnalSlider, RightSlider, SNRLeftSlider, 
        % SignalSlider, TrainTestSplitSlider
        function SNRleftSliderValueChanged(app, event)
            create_signal(app); 
            latent_variable_methods(app);
            plot_figures(app);
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

            % Create UIAxesReg
            app.UIAxesReg = uiaxes(app.LAVADEUIFigure);
            title(app.UIAxesReg, 'Regression on Test Data')
            xlabel(app.UIAxesReg, 'y_{pred}')
            ylabel(app.UIAxesReg, 'y_{true}')
            app.UIAxesReg.PlotBoxAspectRatio = [1.03413654618474 1 1];
            app.UIAxesReg.FontSize = 15;
            app.UIAxesReg.XTick = [0 0.5 1];
            app.UIAxesReg.NextPlot = 'add';
            app.UIAxesReg.XGrid = 'on';
            app.UIAxesReg.YGrid = 'on';
            app.UIAxesReg.BackgroundColor = [1 1 1];
            app.UIAxesReg.Position = [605 15 320 320];

            % Create MethodDropDownLabel
            app.MethodDropDownLabel = uilabel(app.LAVADEUIFigure);
            app.MethodDropDownLabel.BackgroundColor = [1 1 1];
            app.MethodDropDownLabel.HorizontalAlignment = 'right';
            app.MethodDropDownLabel.FontSize = 16.5;
            app.MethodDropDownLabel.Position = [688 763 62 22];
            app.MethodDropDownLabel.Text = 'Method';

            % Create MethodDropDown
            app.MethodDropDown = uidropdown(app.LAVADEUIFigure);
            app.MethodDropDown.Items = {'PLS', 'CCA', 'PCR', 'RR', 'LASSO', 'EN'};
            app.MethodDropDown.ValueChangedFcn = createCallbackFcn(app, @UpdatePlotsButtonPushed2, true);
            app.MethodDropDown.FontSize = 16.5;
            app.MethodDropDown.BackgroundColor = [1 1 1];
            app.MethodDropDown.Position = [765 763 100 22];
            app.MethodDropDown.Value = 'CCA';

            % Create SNRLeftSliderLabel
            app.SNRLeftSliderLabel = uilabel(app.LAVADEUIFigure);
            app.SNRLeftSliderLabel.HorizontalAlignment = 'right';
            app.SNRLeftSliderLabel.FontSize = 15;
            app.SNRLeftSliderLabel.Position = [680 451 66 22];
            app.SNRLeftSliderLabel.Text = 'SNR Left';

            % Create SNRLeftSlider
            app.SNRLeftSlider = uislider(app.LAVADEUIFigure);
            app.SNRLeftSlider.Limits = [1 50];
            app.SNRLeftSlider.MajorTicks = [1 10 20 30 40 50];
            app.SNRLeftSlider.ValueChangedFcn = createCallbackFcn(app, @SNRleftSliderValueChanged, true);
            app.SNRLeftSlider.MinorTicks = [1 5 10 15 20 25 30 35 40 45 50];
            app.SNRLeftSlider.FontSize = 14;
            app.SNRLeftSlider.Position = [769 471 95 3];
            app.SNRLeftSlider.Value = 30;

            % Create TrainTestSplitSlider_3Label
            app.TrainTestSplitSlider_3Label = uilabel(app.LAVADEUIFigure);
            app.TrainTestSplitSlider_3Label.HorizontalAlignment = 'right';
            app.TrainTestSplitSlider_3Label.FontSize = 15;
            app.TrainTestSplitSlider_3Label.Position = [587 394 105 22];
            app.TrainTestSplitSlider_3Label.Text = 'Train-Test Split';

            % Create TrainTestSplitSlider
            app.TrainTestSplitSlider = uislider(app.LAVADEUIFigure);
            app.TrainTestSplitSlider.Limits = [0.1 0.9];
            app.TrainTestSplitSlider.ValueChangedFcn = createCallbackFcn(app, @SNRleftSliderValueChanged, true);
            app.TrainTestSplitSlider.FontSize = 15;
            app.TrainTestSplitSlider.Position = [713 403 264 3];
            app.TrainTestSplitSlider.Value = 0.7;

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

            % Create relevantDatapointsEditFieldLabel
            app.relevantDatapointsEditFieldLabel = uilabel(app.LAVADEUIFigure);
            app.relevantDatapointsEditFieldLabel.HorizontalAlignment = 'right';
            app.relevantDatapointsEditFieldLabel.FontSize = 16.5;
            app.relevantDatapointsEditFieldLabel.Position = [926 563 157 22];
            app.relevantDatapointsEditFieldLabel.Text = '#relevant Datapoints';

            % Create relevantDatapointsEditField
            app.relevantDatapointsEditField = uieditfield(app.LAVADEUIFigure, 'numeric');
            app.relevantDatapointsEditField.Limits = [2 5000];
            app.relevantDatapointsEditField.ValueChangedFcn = createCallbackFcn(app, @DrawNewSampleButtonPushed, true);
            app.relevantDatapointsEditField.Position = [1101 563 50 22];
            app.relevantDatapointsEditField.Value = 10;

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
            app.StartVEditFieldLabel.Position = [669 683 56 22];
            app.StartVEditFieldLabel.Text = 'Start V';

            % Create StartVEditField
            app.StartVEditField = uieditfield(app.LAVADEUIFigure, 'numeric');
            app.StartVEditField.ValueChangedFcn = createCallbackFcn(app, @DrawNewSampleButtonPushed, true);
            app.StartVEditField.FontSize = 15;
            app.StartVEditField.Position = [731 683 33 22];
            app.StartVEditField.Value = 2;

            % Create EndVLabel
            app.EndVLabel = uilabel(app.LAVADEUIFigure);
            app.EndVLabel.HorizontalAlignment = 'right';
            app.EndVLabel.FontSize = 16.5;
            app.EndVLabel.Position = [667 643 58 22];
            app.EndVLabel.Text = 'End   V';

            % Create EndVEditField
            app.EndVEditField = uieditfield(app.LAVADEUIFigure, 'numeric');
            app.EndVEditField.ValueChangedFcn = createCallbackFcn(app, @DrawNewSampleButtonPushed, true);
            app.EndVEditField.FontSize = 15;
            app.EndVEditField.Position = [731 643 33 22];
            app.EndVEditField.Value = -5;

            % Create SignalSliderLabel
            app.SignalSliderLabel = uilabel(app.LAVADEUIFigure);
            app.SignalSliderLabel.HorizontalAlignment = 'right';
            app.SignalSliderLabel.FontSize = 15;
            app.SignalSliderLabel.Position = [889 451 47 22];
            app.SignalSliderLabel.Text = 'Signal';

            % Create SignalSlider
            app.SignalSlider = uislider(app.LAVADEUIFigure);
            app.SignalSlider.Limits = [1 50];
            app.SignalSlider.MajorTicks = [1 10 20 30 40 50];
            app.SignalSlider.ValueChangedFcn = createCallbackFcn(app, @SNRleftSliderValueChanged, true);
            app.SignalSlider.MinorTicks = [1 5 10 15 20 25 30 35 40 45 50];
            app.SignalSlider.FontSize = 14;
            app.SignalSlider.Position = [954 471 95 3];
            app.SignalSlider.Value = 30;

            % Create RightSliderLabel
            app.RightSliderLabel = uilabel(app.LAVADEUIFigure);
            app.RightSliderLabel.HorizontalAlignment = 'right';
            app.RightSliderLabel.FontSize = 15;
            app.RightSliderLabel.Position = [1075 451 41 22];
            app.RightSliderLabel.Text = 'Right';

            % Create RightSlider
            app.RightSlider = uislider(app.LAVADEUIFigure);
            app.RightSlider.Limits = [1 50];
            app.RightSlider.MajorTicks = [1 10 20 30 40 50];
            app.RightSlider.ValueChangedFcn = createCallbackFcn(app, @SNRleftSliderValueChanged, true);
            app.RightSlider.MinorTicks = [1 5 10 15 20 25 30 35 40 45 50];
            app.RightSlider.FontSize = 14;
            app.RightSlider.Position = [1139 471 95 3];
            app.RightSlider.Value = 30;

            % Create LeftVLabel
            app.LeftVLabel = uilabel(app.LAVADEUIFigure);
            app.LeftVLabel.HorizontalAlignment = 'right';
            app.LeftVLabel.FontSize = 16.5;
            app.LeftVLabel.Position = [659 603 67 22];
            app.LeftVLabel.Text = ' Left    V';

            % Create LeftVEditField
            app.LeftVEditField = uieditfield(app.LAVADEUIFigure, 'numeric');
            app.LeftVEditField.ValueChangedFcn = createCallbackFcn(app, @DrawNewSampleButtonPushed, true);
            app.LeftVEditField.FontSize = 15;
            app.LeftVEditField.Position = [731 603 33 22];
            app.LeftVEditField.Value = -1;

            % Create SigLEditFieldLabel
            app.SigLEditFieldLabel = uilabel(app.LAVADEUIFigure);
            app.SigLEditFieldLabel.HorizontalAlignment = 'right';
            app.SigLEditFieldLabel.FontSize = 16.5;
            app.SigLEditFieldLabel.Position = [789 603 38 22];
            app.SigLEditFieldLabel.Text = 'SigL';

            % Create SigLEditField
            app.SigLEditField = uieditfield(app.LAVADEUIFigure, 'numeric');
            app.SigLEditField.Limits = [0 Inf];
            app.SigLEditField.ValueChangedFcn = createCallbackFcn(app, @DrawNewSampleButtonPushed, true);
            app.SigLEditField.FontSize = 15;
            app.SigLEditField.Position = [832 603 33 22];
            app.SigLEditField.Value = 2;

            % Create RightVLabel
            app.RightVLabel = uilabel(app.LAVADEUIFigure);
            app.RightVLabel.HorizontalAlignment = 'right';
            app.RightVLabel.FontSize = 16.5;
            app.RightVLabel.Position = [662 563 64 22];
            app.RightVLabel.Text = 'Right  V';

            % Create RightVEditField
            app.RightVEditField = uieditfield(app.LAVADEUIFigure, 'numeric');
            app.RightVEditField.ValueChangedFcn = createCallbackFcn(app, @DrawNewSampleButtonPushed, true);
            app.RightVEditField.FontSize = 15;
            app.RightVEditField.Position = [731 563 33 22];
            app.RightVEditField.Value = -4;

            % Create SigREditFieldLabel
            app.SigREditFieldLabel = uilabel(app.LAVADEUIFigure);
            app.SigREditFieldLabel.HorizontalAlignment = 'right';
            app.SigREditFieldLabel.FontSize = 16.5;
            app.SigREditFieldLabel.Position = [785 563 41 22];
            app.SigREditFieldLabel.Text = 'SigR';

            % Create SigREditField
            app.SigREditField = uieditfield(app.LAVADEUIFigure, 'numeric');
            app.SigREditField.Limits = [0 Inf];
            app.SigREditField.ValueChangedFcn = createCallbackFcn(app, @DrawNewSampleButtonPushed, true);
            app.SigREditField.FontSize = 15;
            app.SigREditField.Position = [832 563 33 22];
            app.SigREditField.Value = 2;

            % Create DrawNewSampleButton
            app.DrawNewSampleButton = uibutton(app.LAVADEUIFigure, 'push');
            app.DrawNewSampleButton.ButtonPushedFcn = createCallbackFcn(app, @DrawNewSampleButtonPushed, true);
            app.DrawNewSampleButton.FontSize = 15;
            app.DrawNewSampleButton.FontWeight = 'bold';
            app.DrawNewSampleButton.Position = [1064 350 147 27];
            app.DrawNewSampleButton.Text = 'Draw New Sample';

            % Create LatentVariableMethodDemonstratorLabel
            app.LatentVariableMethodDemonstratorLabel = uilabel(app.LAVADEUIFigure);
            app.LatentVariableMethodDemonstratorLabel.FontSize = 28;
            app.LatentVariableMethodDemonstratorLabel.FontWeight = 'bold';
            app.LatentVariableMethodDemonstratorLabel.Position = [70 754 518 37];
            app.LatentVariableMethodDemonstratorLabel.Text = 'Latent Variable Method Demonstrator';

            % Create PositionofSignalLabel
            app.PositionofSignalLabel = uilabel(app.LAVADEUIFigure);
            app.PositionofSignalLabel.HorizontalAlignment = 'right';
            app.PositionofSignalLabel.FontSize = 15;
            app.PositionofSignalLabel.Position = [691 509 122 22];
            app.PositionofSignalLabel.Text = 'Position of Signal';

            % Create PositionSingnalSlider
            app.PositionSingnalSlider = uislider(app.LAVADEUIFigure);
            app.PositionSingnalSlider.Limits = [0 1];
            app.PositionSingnalSlider.ValueChangedFcn = createCallbackFcn(app, @SNRleftSliderValueChanged, true);
            app.PositionSingnalSlider.FontSize = 14;
            app.PositionSingnalSlider.Position = [841 528 259 3];
            app.PositionSingnalSlider.Value = 0.5;

            % Create NoiseCheckBox
            app.NoiseCheckBox = uicheckbox(app.LAVADEUIFigure);
            app.NoiseCheckBox.ValueChangedFcn = createCallbackFcn(app, @SNRleftSliderValueChanged, true);
            app.NoiseCheckBox.Text = 'Noise';
            app.NoiseCheckBox.FontSize = 15;
            app.NoiseCheckBox.Position = [610 451 60 22];

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

            % Create UpdatePlotsButton
            app.UpdatePlotsButton = uibutton(app.LAVADEUIFigure, 'push');
            app.UpdatePlotsButton.ButtonPushedFcn = createCallbackFcn(app, @UpdatePlotsButtonPushed2, true);
            app.UpdatePlotsButton.FontSize = 15;
            app.UpdatePlotsButton.FontWeight = 'bold';
            app.UpdatePlotsButton.Position = [1084 385 108 27];
            app.UpdatePlotsButton.Text = {'Update Plots'; ''};

            % Create UIAxesRegTrain
            app.UIAxesRegTrain = uiaxes(app.LAVADEUIFigure);
            title(app.UIAxesRegTrain, 'Regression on Train Data')
            xlabel(app.UIAxesRegTrain, 'y_{pred}')
            ylabel(app.UIAxesRegTrain, 'y_{true}')
            app.UIAxesRegTrain.PlotBoxAspectRatio = [1.03413654618474 1 1];
            app.UIAxesRegTrain.FontSize = 15;
            app.UIAxesRegTrain.MinorGridAlpha = 0.1;
            app.UIAxesRegTrain.XTick = [0 0.5 1];
            app.UIAxesRegTrain.NextPlot = 'add';
            app.UIAxesRegTrain.XGrid = 'on';
            app.UIAxesRegTrain.YGrid = 'on';
            app.UIAxesRegTrain.BackgroundColor = [1 1 1];
            app.UIAxesRegTrain.Position = [938 15 320 320];

            % Create SigSEditFieldLabel
            app.SigSEditFieldLabel = uilabel(app.LAVADEUIFigure);
            app.SigSEditFieldLabel.HorizontalAlignment = 'right';
            app.SigSEditFieldLabel.FontSize = 16.5;
            app.SigSEditFieldLabel.Position = [784 683 40 22];
            app.SigSEditFieldLabel.Text = 'SigS';

            % Create SigSEditField
            app.SigSEditField = uieditfield(app.LAVADEUIFigure, 'numeric');
            app.SigSEditField.Limits = [0 Inf];
            app.SigSEditField.ValueChangedFcn = createCallbackFcn(app, @DrawNewSampleButtonPushed, true);
            app.SigSEditField.FontSize = 15;
            app.SigSEditField.Position = [832 683 33 22];

            % Create SigEEditFieldLabel
            app.SigEEditFieldLabel = uilabel(app.LAVADEUIFigure);
            app.SigEEditFieldLabel.HorizontalAlignment = 'right';
            app.SigEEditFieldLabel.FontSize = 16.5;
            app.SigEEditFieldLabel.Position = [785 643 39 22];
            app.SigEEditFieldLabel.Text = 'SigE';

            % Create SigEEditField
            app.SigEEditField = uieditfield(app.LAVADEUIFigure, 'numeric');
            app.SigEEditField.Limits = [0 Inf];
            app.SigEEditField.ValueChangedFcn = createCallbackFcn(app, @DrawNewSampleButtonPushed, true);
            app.SigEEditField.FontSize = 15;
            app.SigEEditField.Position = [832 643 33 22];

            % Create StandardizeInputsCheckBox
            app.StandardizeInputsCheckBox = uicheckbox(app.LAVADEUIFigure);
            app.StandardizeInputsCheckBox.ValueChangedFcn = createCallbackFcn(app, @UpdatePlotsButtonPushed2, true);
            app.StandardizeInputsCheckBox.Text = {'Standardize Inputs'; ''};
            app.StandardizeInputsCheckBox.FontSize = 16.5;
            app.StandardizeInputsCheckBox.Position = [705 723 160 22];

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

            % Show the figure after all components are created
            app.LAVADEUIFigure.Visible = 'off';
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