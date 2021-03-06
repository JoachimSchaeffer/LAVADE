classdef Data_class < handle
    % Data superclass. Each dataset subclass inherits from it.
    properties
        % general properties of the class
        name;           % Name of the object
        group;          % Grouped splitting or not?
        X;              % X data (might be modified with noise)
        y;              % modeling objective (might be modified with noise)
        y_unit;         % Unit of the modelling objective (will be plotted)
        X_true;         % True X data wo noise
        y_true;         % True modeling objective wo noise
        X_train;
        X_test;
        y_train;
        y_test;
        idx_para;
        u_lim;
        idx_train;
        idx_test;
        vis;
        vis_dd_raman;
        vis_cb;         % Visibility details in LAVADE file
        ttr_limits;     % Train test Ratio Limits
        ttr_majorticks; % Train test Ratio Major Ticks
        x_label_text;   % x-axis label of the data X
        y_label_text;   % y-xis label of the data X
        output_text;    %
        measurements;  
        x_values;
    end

    properties (Constant)
    end
    
    methods
        function obj = Data_class(X,y)
            % Very simple constructor
            obj.X = X; 
            obj.y = y;
            obj.X_true = X; 
            obj.y_true = y;
            % Only for the articficial data generation method
            obj.vis = 'off';
            % Only for the Raman data generation method
            obj.vis_dd_raman = 'off';
        end

        function add_noise(obj, noise_X, noise_y, noise)
            % Adding noise to the data matrix X as well as y  
            if noise
                obj.X = addwgn(obj.X_true, noise_X);
                obj.y = addwgn(obj.y_true, noise_y);
            else
                obj.X = obj.X_true; 
                obj.y = obj.y_true;
            end
        end 
        
        function train_test_split(obj, fraction_train, new, group)
            % splits the dataset into training and test dataset, according
            % to the fraction. Respects the grouping structure if requested
            % to do so for the 'Paracetamol' and 'LFP' dataset 
            if ~group
                if new
                    obj.idx_para = randperm(obj.measurements);
                end
                cut_idx = round(fraction_train*obj.measurements);
                obj.idx_train = obj.idx_para(1:cut_idx);
                obj.idx_test = obj.idx_para(cut_idx+1:end);
            end 
            obj.X_train = obj.X(obj.idx_train,:);
            obj.X_test = obj.X(obj.idx_test,:);
            obj.y_train = obj.y(obj.idx_train);
            obj.y_test = obj.y(obj.idx_test);
            obj.u_lim = cast(size(obj.X_train,1)-1, 'double');
        end 
     end 
end 