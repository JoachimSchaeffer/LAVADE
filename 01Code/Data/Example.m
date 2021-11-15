classdef Example < Data_class
    %LFP Summary of this class goes here
    
    properties
        name; 
        group;
        x_label_text;   % x-axis label of the data X
        y_label_text;   % y-xis label of the data X
        output_text;    %
        measurements;
        x_values;
        vis;            % Visibility details in LAVADE file
        vis_cb;         % Visibility details in LAVADE file
        ttr_limits;     % Train test Ratio Limits
        ttr_majorticks; % Train test Ratio Major Ticks
        % properties only neede for the example case
        %start_value;    %  Mean of first datapoint
        %final_value;    %  Mean of last datapoint 
        %target_left;    %  Mean of datapoint after which the signal section starts 
        %target_right;   %  Mean of last datapoint of the signal section 
    end
    
    methods
        function obj = Example(app)
            measurements = app.ExperimentsEditField.Value;

            sigma_left = app.SigLEditField.Value;
            target_left = normrnd(app.LeftVEditField.Value, sigma_left, measurements, 1);
            
            sigma_right = app.SigREditField.Value;
            target_right = normrnd(app.RightVEditField.Value, sigma_right, measurements, 1);
            
            sigma_start = app.SigSEditField.Value;
            start_value = normrnd(app.StartVEditField.Value, sigma_start, measurements, 1);
            
            sigma_end = app.SigEEditField.Value;
            final_value = normrnd(app.EndVEditField.Value, sigma_end, measurements, 1);
            
            datapoints = app.DatapointsEditField.Value;
            data_points_signal = app.relDatapointsEditField.Value;
            data_points_left = int64(app.PositionSingnalSlider.Value*(datapoints-data_points_signal));     
            data_points_right = datapoints - (data_points_left+data_points_signal);

            rand_uc_l = zeros(measurements, data_points_left+1);
            rand_uc_r = zeros(measurements, data_points_right+1);
            signal = zeros(measurements, data_points_signal);
                        
            for i=1:measurements
                rand_uc_l(i, :) = linspace(start_value(i), target_left(i), data_points_left+1);
                rand_uc_r(i, :) = linspace(target_right(i), final_value(i), data_points_right+1);
                signal(i, :) =  linspace(target_left(i), target_right(i), data_points_signal);
            end
            % Build the datamatrix X
            X = [rand_uc_l(:, 1:end-1), signal, rand_uc_r(:, 2:end)];         
            % The y we want to predict: Slope of the signal section
            % Here you can also insert different responses 
            y = (target_right-target_left)/data_points_signal;
            obj = obj@Data_class(X, y);
            obj.measurements = measurements;
            obj.x_label_text = 'Index';
            obj.y_label_text = 'Value'; 
            obj.x_values = linspace(1,size(obj.X,2),size(obj.X,2));
            obj.name = 'Example';
            obj.output_text = 'Slope';

            obj.ttr_limits = [0.1 0.9];
            obj.ttr_majorticks = [0.1 0.3 0.5 0.7 0.9];
            obj.vis = 'on';
            obj.vis_cb = 'off';
        end
        
        function train_test_split(obj, fraction_train, new, ~)
            % splits the dataset into training and test dataset, accroding
            % to the fraction. Respects the grouping structure if requested
            % to do so
            train_test_split@Data_class(obj, fraction_train, new, 0)
        end
    end
end