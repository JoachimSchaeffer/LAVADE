%% Test

app = lavade_exported;
max_nloop = 250;

app.SNRLeftSlider.Value = 20;
app.RightSlider.Value = 20;   
app.SignalSlider.Value = 20;
app.SigEEditField.Value = 2;
app.SigSEditField.Value = 2;
app.NoiseCheckBox.Value = true;
app.MethodDropDown.Value = "EN";

for i=1:4
    r2train_ = [];
    r2test_ = [];
    for j= 1:max_nloop
        init(app)
        r2test_(j) = app.r2_test; 
        r2train_(j) = app.r2_train; 
    end
    r2_train_en(i, :) = r2train_; 
    r2_test_en(i,:) =  r2test_;
end

%% 
results.en.r2train = reshape(r2_train_en,1,numel(r2_train_en));
results.en.r2test = reshape(r2_test_en,1,numel(r2_test_en)); 


[a,b] = stats(results.en.r2train);
[c,d] = stats(results.en.r2test);


function [mean_,std_] = stats(vector)
        mean_ = mean(vector); 
        std_ = std(vector);
end
%%
    