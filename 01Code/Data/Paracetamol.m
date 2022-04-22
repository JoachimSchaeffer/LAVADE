classdef Paracetamol < Data_class
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
        function obj = Paracetamol()
            % Construct an instance of this class
            % is a subclass of the dataclass
            % Load data 
            data_struct = load('Paracetamol.mat');
            obj = obj@Data_class(data_struct.X, data_struct.y);
            obj.group = data_struct.group;
            obj.measurements = size(obj.X, 1);
            
            obj.name = 'Paracetamol';
            obj.output_text = 'Concentration';
            obj.x_label_text = 'Wavenumber [cm^{-1}]';
            obj.y_label_text = 'Absorbance';
            obj.x_values = data_struct.freq;

            obj.ttr_limits = [0.3 0.7];
            obj.ttr_majorticks = [0.3 0.5 0.7];
            obj.vis = 'off';
            obj.vis_cb = 'on';
        end
        
        function train_test_split(obj, fraction_train, new, group)
            % splits the dataset into training and test dataset, accroding
            % to the fraction. Respects the grouping structure if requested
            % to do so
            if group
                groups = 5;
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