classdef Raman < Data_class
    % Summary of this class goes here
    
    properties
        name; 
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
                    % X = X(1:86,:);
                    % y = y(:,2);
                    % Frequency 
                    x_vals = readmatrix('20200820_fb_r3spectra_X_nbFlex.csv');
                case 'perf_r1' 
                    X = readmatrix('20200911_perf_r1_X_nbFlex.csv');
                    y = readmatrix('20200911_perf_r1_y_nbFlex.csv');
                    % X = X(1:86,:);
                    % y = y(:,2);
                    % Frequency 
                    x_vals = readmatrix('20200911_perf_r1spectra_X_nbFlex.csv');
                case 'perf_r4'
                    X = readmatrix('20210312_perf_r4_X_nbFlex.csv');
                    y = readmatrix('20210312_perf_r4_y_nbFlex.csv');
                    % X = X(1:86,:);
                    % y = y(:,2);
                    % Frequency 
                    x_vals = readmatrix('20210312_perf_r4spectra_X_nbFlex.csv');
                case 'perf_R1'
                    X = readmatrix('20210507_perf_R1_X_nbFlex.csv');
                    y = readmatrix('20210507_perf_R1_y_nbFlex.csv');
                    % X = X(1:86,:);
                    % y = y(:,2);
                    % Frequency 
                    x_vals = readmatrix('20210507_perf_R1spectra_X_nbFlex.csv');
                case 'perf_R2'
                    X = readmatrix('20210709_perf_R2_X_nbFlex.csv');
                    y = readmatrix('20210709_perf_R2_y_nbFlex.csv');
                    % X = X(1:86,:);
                    % y = y(:,2);
                    % Frequency 
                    x_vals = readmatrix('20210709_perf_R2spectra_X_nbFlex.csv');
                case 'perf_R3'
                    X = readmatrix('20211119_perf_R3_X_nbFlex.csv');
                    y = readmatrix('20211119_perf_R3_y_nbFlex.csv');
                    % X = X(1:86,:);
                    % y = y(:,2);
                    % Frequency 
                    x_vals = readmatrix('20211119_perf_R3spectra_X_nbFlex.csv');
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
            obj = obj@Data_class(X(:,1:3000), y(:,idx));
            obj.x_values = x_vals(1,1:3000);
           
            obj.measurements = size(obj.X, 1);
            
            obj.name = 'Raman Spectra';
            obj.output_text = 'Concentration';
            obj.x_label_text = 'Raman Shift (cm^-1)';
            obj.y_label_text = 'Counts';
            % Train-Test Ratio Limits
            obj.ttr_limits = [0.3 0.7];
            obj.ttr_majorticks = [0.3 0.5 0.7];
            obj.vis = 'off';
            obj.vis_cb = 'off';
        end
        
        function train_test_split(obj, fraction_train, new, ~)
            % splits the dataset into training and test dataset, randomly
            train_test_split@Data_class(obj, fraction_train, new, 0)
        end
    end
end