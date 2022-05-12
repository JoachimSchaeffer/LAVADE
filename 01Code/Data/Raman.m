classdef Raman < Data_class
    %Raman datset class
    
    properties
        % Local properties exclusively for this dataset go in here
    end
    
    methods
        function obj = Raman(filter, meta)
            % Construct an instance of this class.
            % This is a subclass of the dataclass.
            % Load data:
            switch filter
                case 'No Background Removed'
                    data_raman = readmatrix('RamanRaw.csv', 'NumHeaderLines', 1);
                case 'LMJ5'
                    data_raman = readmatrix('RamanLMJBackSubOrder5.csv', 'NumHeaderLines', 1);
                case 'LMJ6'
                    data_raman = readmatrix('RamanLMJBackSubOrder6.csv', 'NumHeaderLines', 1);
                case 'Matlab msback'
                    data_raman = readmatrix('RamanMsback.csv', 'NumHeaderLines', 1);
            end
             
            
            groups = data_raman(:,1);
            X = data_raman(:,2:end-4);
            x_vals = linspace(100,100-1+size(X,2), size(X,2));
            y = data_raman(:, end-3:end);     
            
            switch meta
                case 'Gluc'
                    idx = 1; 
                    y_unit = '(g/L)';
                    output_text = 'Glucose Concentration';
                case 'Lac'
                    idx = 2;
                    y_unit = '(g/L)';
                    output_text = 'Lactate Concentration';
                case 'Gln'
                    idx = 3;
                    y_unit = '(mmol/L)';
                    output_text = 'Glutamine Concentration';
                case 'NH4'
                    idx = 4;
                    y_unit = '(mmol/L)';
                    output_text = 'NH_4 Concentration';
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
            obj.y_unit = y_unit;
            obj.output_text = output_text; 
            obj.x_label_text = 'Raman Shift (cm^{-1})';
            obj.y_label_text = 'Counts';
            % Train-Test Ratio Limits
            obj.ttr_limits = [0.1 0.9];
            obj.ttr_majorticks = [0.1 0.3 0.5 0.7 0.9];
            % Grouping check box
            obj.vis_cb = 'on';
            % Raman specific drop downs.
            obj.vis_dd_raman = 'on';
        end
        

        function train_test_split(obj, fraction_train, new, group)
            % splits the dataset into training and test dataset, accroding
            % to the fraction. Respects the grouping structure if requested
            % to do so
            if group
                % Only two groups
                groups = 2;
                obj.idx_para = randperm(groups);
                if new
                    obj.idx_para = flip(obj.idx_para);
                end
                train_groups = obj.idx_para(1);
                obj.idx_test = sum(obj.group==train_groups,2)==0;
                obj.idx_train = sum(obj.group==train_groups,2)==1;
            end
            train_test_split@Data_class(obj, fraction_train, new, group)
        end

    end
end