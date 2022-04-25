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
        function obj = Raman(run, meta)
            % Construct an instance of this class.
            % This is a subclass of the dataclass.
            % Load data:
            
            switch run
                case 'fb_r3' 
                    X = readmatrix('20200820_fb_r3_X_nbFlex.csv');
                    y = readmatrix('20200820_fb_r3_y_nbFlex.csv');
                    % Frequency 
                    x_vals = readmatrix('20200820_fb_r3spectra_X_nbFlex.csv');
                case 'perf_r1' 
                    X = readmatrix('20200911_perf_r1_X_nbFlex.csv');
                    y = readmatrix('20200911_perf_r1_y_nbFlex.csv');
                    % Frequency 
                    x_vals = readmatrix('20200911_perf_r1spectra_X_nbFlex.csv');
                case 'perf_r4'
                    X = readmatrix('20210312_perf_r4_X_nbFlex.csv');
                    y = readmatrix('20210312_perf_r4_y_nbFlex.csv');
                    % Frequency 
                    x_vals = readmatrix('20210312_perf_r4spectra_X_nbFlex.csv');
                case 'perf_R1'
                    X = readmatrix('20210507_perf_R1_X_nbFlex.csv');
                    y = readmatrix('20210507_perf_R1_y_nbFlex.csv');
                    % Frequency 
                    x_vals = readmatrix('20210507_perf_R1spectra_X_nbFlex.csv');
                case 'perf_R2'
                    X = readmatrix('20210709_perf_R2_X_nbFlex.csv');
                    y = readmatrix('20210709_perf_R2_y_nbFlex.csv');
                    % Frequency 
                    x_vals = readmatrix('20210709_perf_R2spectra_X_nbFlex.csv');
                case 'perf_R3'
                    X = readmatrix('20211119_perf_R3_X_nbFlex.csv');
                    y = readmatrix('20211119_perf_R3_y_nbFlex.csv');
                    % Frequency 
                    x_vals = readmatrix('20211119_perf_R3spectra_X_nbFlex.csv');
                case 'all'
                    % Instead Load all the data and have the runs as groups (; 
                    X1 = readmatrix('20200820_fb_r3_X_nbFlex.csv');
                    y1 = readmatrix('20200820_fb_r3_y_nbFlex.csv');
                    X2 = readmatrix('20200911_perf_r1_X_nbFlex.csv');
                    y2 = readmatrix('20200911_perf_r1_y_nbFlex.csv');
                    X3 = readmatrix('20210312_perf_r4_X_nbFlex.csv');
                    y3 = readmatrix('20210312_perf_r4_y_nbFlex.csv');
                    X4 = readmatrix('20210709_perf_R2_X_nbFlex.csv');
                    y4 = readmatrix('20210709_perf_R2_y_nbFlex.csv');
                    X5 = readmatrix('20211119_perf_R3_X_nbFlex.csv');
                    y5 = readmatrix('20211119_perf_R3_y_nbFlex.csv');
                    % Frequency 
                    x_vals = readmatrix('20200820_fb_r3spectra_X_nbFlex.csv');
                    X = [X1; X2; X3; X4; X5];
                    groups = [...
                        0*ones(size(X1,1), 1); ones(size(X2,1), 1); 2*ones(size(X3,1), 1);
                        3*ones(size(X4,1), 1); 4*ones(size(X5,1), 1)];
            end
            
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
            
            y = [y1(:,1:4); y2(:,1:4); y3(:,1:4); y4(:,1:4); y5(:,1:4)];

            obj = obj@Data_class(X(:,:), y(:,idx));
            obj.x_values = x_vals(1, :);
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