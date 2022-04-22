classdef LFP < Data_class
    % Summary of this class goes here
    
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
        
    end
    
    methods
        function obj = LFP()
            % Construct an instance of this class
            % is a subclass of the dataclass
            % Load data 
            data_struct = load('LFP.mat');
            X = flip(data_struct.DQ_100_10,2);
            obj = obj@Data_class(X, data_struct.y);
            obj.group = data_struct.group;
            obj.measurements = size(obj.X,1);
            
            obj.name = 'LFP'; 
            obj.output_text = 'Cycle Life';
            obj.x_label_text = 'Voltage (V)';
            obj.y_label_text = '\DeltaQ_{100-10}';
            obj.x_values = linspace(2.0,3.5,1000);
            obj.ttr_limits = [0.1 0.9];
            obj.ttr_majorticks = [0.1 0.3 0.5 0.7 0.9];
            obj.vis = 'off';
            obj.vis_cb = 'on';
        end
        
        function train_test_split(obj, fraction_train, new, group)
            % splits the dataset into training and test dataset, accroding
            % to the fraction. Respects the grouping structure if requested
            % to do so
            if group
                groups = size(unique(obj.group),1);
                if new
                    obj.idx_para = randperm(groups);
                end
                train_groups = obj.idx_para(1:ceil(fraction_train*groups));
                obj.idx_test = sum(obj.group==train_groups,2)==0;
                obj.idx_train = sum(obj.group==train_groups,2)==1;
            end
            train_test_split@Data_class(obj, fraction_train, new, group)
        end
        
    end
end