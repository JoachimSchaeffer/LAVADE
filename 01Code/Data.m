classdef Data < handle
    properties
        % general properties of the class
        name;           % Name of the dataset, assigned in LAVADE
        active_dataset; % Name of the dataset currently laoded
        X;              % X data
        group;          % underlying implicit group of the data. 
        y;              % modeling objective    
        X_train;
        X_test;
        y_train;
        y_test;
        idx_para;       % (random) index for training and test set
        x_label_text;   % x-axis label of the data X
        y_label_text;   % y-xis label of the data X
        output_text;    %
        x_values;
        % properties only neede for the example case, but its fine
        % if they are empty otherwise
        start_value;    %  Mean of first datapoint
        final_value;    %  Mean of last datapoint 
        target_left;    %  Mean of datapoint after which the signal section starts 
        target_right;   %  Mean of last datapoint of the signal section 
        measurements;   %  #rows of X
    end

    properties (Constant)
    end
    
    methods
        function create_signal(obj, app, noise_change)
        % Function to create the data ('Example') or load data for
        % the 'LFP' and 'Paracetamol' case
        % Example case: Creation of the data matrix X consisting of
        % # of Experiments: rows
        % # of Datapoints:  columns
           
            switch obj.name
                case 'Example'
                    datapoints = app.DatapointsEditField.Value;
                    data_points_signal = app.relDatapointsEditField.Value;
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
                    obj.x_label_text = 'Index';
                    obj.y_label_text = 'Value'; 
                    obj.x_values = linspace(1,size(obj.X,2),size(obj.X,2));
                    obj.active_dataset = 'Example';
                    obj.output_text = 'Slope';

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
                    obj.output_text = 'Concentration';
                    obj.x_label_text = 'Wavenumber [cm^{-1}]';
                    obj.y_label_text = 'Absorbance';
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
                    obj.output_text = 'Cycle Life';
                    obj.x_label_text = 'Voltage (V)';
                    obj.y_label_text = '\DeltaQ_{100-10}';
                    obj.x_values = linspace(2.0,3.5,1000);
            end
        end

        function add_noise(obj, noise, noise_X, noise_y)
        % Adding noise to the data matrix X as well as y  
            if noise
                obj.X = awgn(obj.X, noise_X, 'measured');
                obj.y = awgn(obj.y, noise_y, 'measured');
            end
        end 
         
        function u_lim = train_test_split(obj, fraction_train, new, group)
        % splits the dataset into training and test dataset, accroding
        % to the fraction. Respects the grouping structure if requested
        % to do so for the 'Paracetamol' and 'LFP' dataset 
            if strcmp(obj.name, 'LFP') || strcmp(obj.name, 'Paracetamol') && group
                switch obj.name
                    case 'Paracetamol'
                        % Biased data in test dataset (default) 
                        % 5 groups that are valid for training
                        groups = 5;
                    case 'LFP'
                        % if biased data in test dataset (default) 
                        % 5 groups that are valid for training
                        groups = size(unique(obj.group),1);
                end 

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

            if strcmp(obj.name, 'Example') || ~group
                if new
                    obj.idx_para = randperm(obj.measurements);
                end
                split_ind = int64(fraction_train*obj.measurements);
                u_lim = cast(split_ind-1, 'double');

                idx_train = obj.idx_para(1:round(fraction_train*obj.measurements));
                idx_test = obj.idx_para(round(fraction_train*obj.measurements):end);

                obj.X_train = obj.X(idx_train, :);
                obj.X_test = obj.X(idx_test, :);

                obj.y_train = obj.y(idx_train, :);
                obj.y_test = obj.y(idx_test, :);
            end
        end
     end 
end 