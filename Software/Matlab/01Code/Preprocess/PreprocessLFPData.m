%% Preprocess LFP Data 

%%
rng(3)
protocol = [""];

for i=1:length(batch_comb)
    protocol(i) = batch_comb(i).policy_readable;
end

for i=1:length(batch8)
    protocol(i+length(batch_comb)) = extractBefore(batch8(i).policy_readable,"-newstructure");
end

groups = unique(protocol);

for i=1:length(protocol)
    group(i) = find(strcmp(protocol(i),groups));
end 

%%
group = group';
%%
DQ_100_10 = [QDiff; QD_test2];
y = [batch_comb_label; batch8_label];

%%
save LFP group DQ_100_10 y


