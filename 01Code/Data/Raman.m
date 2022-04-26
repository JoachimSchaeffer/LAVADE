classdef Raman < Data_class
    % Summary of this class goes here
    
    properties
        name; 
        group;
        train_groups;
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
        function obj = Raman(~, meta)
            % Construct an instance of this class.
            % This is a subclass of the dataclass.
            % Load data:
            data_raman = readmatrix('RamanRaw.csv', 'NumHeaderLines', 1);
            groups = data_raman(:,1);
            X = data_raman(:,2:end-4);
            x_vals = linspace(100,100-1+size(X,2), size(X,2));
            y = data_raman(:, end-3:end);     
            
            switch meta
                case 'Gluc'
                    idx = 1; 
                case 'Lac'
                    idx = 2;
                case 'Gln'
                    idx = 3;
                case 'NH4'
                    idx = 4;
            end
            
            % Restrict the Raman spectra range for the analysis. 
            % 400 - 1800 cm-1 (index 1 corresponds to 100cm-1)
            X_ = X(:, 300:1700);
            x_vals_ = x_vals(1, 300:1700);

            obj = obj@Data_class(X_, y(:,idx));
            obj.x_values = x_vals_;
            obj.group = groups;

            obj.measurements = size(obj.X, 1);
            
            obj.name = 'Raman Spectra';
            obj.output_text = 'Concentration';
            obj.x_label_text = 'Raman Shift (cm^{-1})';
            obj.y_label_text = 'Counts';
            % Train-Test Ratio Limits
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
                obj.train_groups = obj.idx_para(1:ceil(fraction_train*groups));
                obj.idx_test = sum(obj.group==obj.train_groups,2)==0;
                obj.idx_train = sum(obj.group==obj.train_groups,2)==1;
            end
            train_test_split@Data_class(obj, fraction_train, new, group)
        end

    end
end