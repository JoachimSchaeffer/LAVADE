classdef LFP < Data
    %LFP Summary of this class goes here
    
    properties
        name; 
        group;
        x_label_text;   % x-axis label of the data X
        y_label_text;   % y-xis label of the data X
        output_text;    %
        x_values;
    end
    
    methods
        function obj = LFP()
            %LFP Construct an instance of this class
            %   LFP is a subclass of the dataclass
            obj.name = 'LFP'; 
            % Load data 
            data_struct = load('LFP.mat');
            
            obj.X = flip(data_struct.DQ_100_10,2);
            obj.y = data_struct.y;
            obj.group = data_struct.group;
              
            obj.measurements = size(obj.X,1);
            obj.output_text = 'Cycle Life';
            obj.x_label_text = 'Voltage (V)';
            obj.y_label_text = '\DeltaQ_{100-10}';
            obj.x_values = linspace(2.0,3.5,1000);
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
        end
    end
end

