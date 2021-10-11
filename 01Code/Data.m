classdef Data < handle
    properties
        % general properties!
        name;
        active_dataset;
        X;
        group; % underlying implicit group of the data. 
        y;
        X_train;
        X_test;
        y_train;
        y_test;
        idx_para; % (random) index for train and test, respecting groups
        x_label_text; 
        x_values;
        % properties on;ly neede for the example case, but its fine
        % if they are empty otherwise
        start_value;     %  Mean of first datapoint
        final_value;     %  Mean of last datapoint 
        target_left;     %  Mean of datapoint after which the signal section starts 
        target_right;    %  Mean of last datapoint of the signal section 
        measurements;
        %
        noise_l;
        noise_r;
        noise_s;
    end

    properties (Constant)
        
    end
    
     methods
         function create_signal(obj, app, noise_change)
            
            switch obj.name
                % Loads the respective data and creates the objects
                % these objects will then later on be used
                case 'Example'
                   
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    % This function crestes the data matrix X consisting of
                    % # of Experiments rows
                    % # of Datapoints columns =
                    % #"Uncorrleated points left" + #Sinal points + #"Uncorrelated points right")
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    datapoints = app.DatapointsEditField.Value;
                    data_points_signal = app.relevantDatapointsEditField.Value;
                    data_points_left = int64(app.PositionSingnalSlider.Value*(datapoints-data_points_signal));     
                    data_points_right = datapoints - (data_points_left+data_points_signal);

                    rand_uc_l = zeros(obj.measurements, data_points_left+1);
                    rand_uc_r = zeros(obj.measurements, data_points_right+1);
                    signal = zeros(obj.measurements, data_points_signal);

                    for i=1:obj.measurements
                        rand_uc_l(i, :) = linspace(obj.start_value(i), obj.target_left(i), data_points_left+1);
                        rand_uc_r(i, :) = linspace(obj.target_right(i), obj.final_value(i), data_points_right+1);
                        signal(i, :) =  linspace(obj.target_left(i), obj.target_right(i), data_points_signal);
                    end
                    % Build the datamatrix X
                    obj.X = [rand_uc_l(:, 1:end-1), signal, rand_uc_r(:, 2:end)];         
                    % The y we want to predict: Slope of the signal section
                    % Here you can also insert different responses 
                    obj.y = (obj.target_right-obj.target_left)/data_points_signal;
                    obj.x_label_text = 'index';
                    obj.x_values = linspace(1,size(obj.X,2),size(obj.X,2));
                    obj.active_dataset = 'Example';

                case 'Paracetamol'
                    % Check, whether dataset was alreday loaded & noise
                    % button changed?
                    if strcmp(obj.name, obj.active_dataset)==0 || noise_change
                        %% Load data 
                        load FTIR_data
                        obj.X = X;
                        obj.y = y;
                        obj.group = group;
                        clear X y group
                    end
                    obj.measurements = size(obj.X,1);
                    obj.active_dataset = 'Paracetamol';
                    obj.x_label_text = 'Frequency [Hz]';
                    obj.x_values = freq;
                case 'LFP' 
                    if strcmp(obj.name, obj.active_dataset)==0 || noise_change
                        %% Load data 
                        load LFP
                        obj.X = flip(DQ_100_10,2);
                        obj.y = y;
                        obj.group = group;
                        clear X y group
                    end
                    obj.measurements = size(obj.X,1);
                    obj.active_dataset = 'LFP';
                    obj.x_label_text = 'V';
                    obj.x_values = linspace(2.0,3.5,1000);
            end
         end

         function add_noise(obj, noise, noise_X, noise_y)
            if noise
                obj.X = awgn(obj.X, noise_X, 'measured');
                obj.y = awgn(obj.y, noise_y, 'measured');
            end
         end 
         
         function u_lim = train_test_split(obj, fraction_train, new)
            switch obj.name
                % Loads the respective data and creates the objects
                % these objects will then later on be used
                case 'Example'
                    % Train Test Split
                    if new
                        obj.idx_para = randperm(obj.measurements);
                    end
                    split_ind = int64(fraction_train*obj.measurements);
                    u_lim = cast(split_ind-1, 'double');
                    
                    % Train Test Split
                    %obj.X_train = obj.X(1:split_ind, :);
                    %obj.X_test = obj.X(split_ind+1:end, :);

                    %obj.y_train = obj.y(1:split_ind, :);
                    %obj.y_test = obj.y(split_ind+1:end, :);
                    idx_train = obj.idx_para(1:round(fraction_train*obj.measurements));
                    idx_test = obj.idx_para(round(fraction_train*obj.measurements):end);
                    
                    obj.X_train = obj.X(idx_train, :);
                    obj.X_test = obj.X(idx_test, :);

                    obj.y_train = obj.y(idx_train, :);
                    obj.y_test = obj.y(idx_test, :);
                case 'Paracetamol'
                    %if biased data in test dataset (default) 
                    % 5 groups that are valid for training
                    groups = 5;
                    if new
                        obj.idx_para = randperm(groups);
                    end

                    train_groups = obj.idx_para(1:ceil(fraction_train*groups));
                    
                    idx_test = sum(obj.group==train_groups,2)==0;
                    idx_train = sum(obj.group==train_groups,2)==1;

                    obj.X_train = obj.X(idx_train,1:364);
                    obj.X_test = obj.X(idx_test,1:364);

                    obj.y_train = obj.y(idx_train);
                    obj.y_test = obj.y(idx_test);
                    
                    u_lim = cast(size(obj.X_train,1)-1, 'double');
                case 'LFP'
                    % if biased data in test dataset (default) 
                    % 5 groups that are valid for training
                    groups = size(unique(obj.group),1);
                    if new
                        obj.idx_para = randperm(groups);
                    end

                    train_groups = obj.idx_para(1:ceil(fraction_train*groups));
                    
                    idx_test = sum(obj.group==train_groups,2)==0;
                    idx_train = sum(obj.group==train_groups,2)==1;

                    obj.X_train = obj.X(idx_train,:);
                    obj.X_test = obj.X(idx_test,:);

                    obj.y_train = obj.y(idx_train);
                    obj.y_test = obj.y(idx_test);
                    
                    u_lim = cast(size(obj.X_train,1)-1, 'double');
            end
         end
       
             
     end 
    
end 