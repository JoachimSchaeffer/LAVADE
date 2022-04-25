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
            A = importdata('Paracetamol.txt');
            
            X = A(1:size(A,1)-2,2:end)';
            c = A(size(A,1), 2:end)';
            y = c./(1+c);
            obj = obj@Data_class(X, y);
            
            % Temperature is unused
            % T = A(size(A,1)-1,2:end)';
            
            % Frequency 
            obj.x_values = A(1:size(A,1)-2,1)';
            
            [~,~,obj.group] = unique(y);
            % Setting the groupnumber of the biased group to zero
            obj.group(obj.group==2) = 0;
            obj.group(obj.group>2) = obj.group(obj.group>2)-1;
                
            obj.measurements = size(obj.X, 1);
            
            obj.name = 'Paracetamol Spectra';
            obj.output_text = 'Concentration';
            obj.x_label_text = 'Wavenumber [cm^{-1}]';
            obj.y_label_text = 'Absorbance';
            obj.ttr_limits = [0.3 0.7];
            obj.ttr_majorticks = [0.3 0.5 0.7];
            obj.vis = 'off';
            obj.vis_cb = 'on';
        end
        
        function train_test_split(obj, fraction_train, new, group)
            % splits the dataset into training and test dataset, according
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