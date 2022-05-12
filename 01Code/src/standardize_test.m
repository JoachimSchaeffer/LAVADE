function Z = standardize_test(X, mu_train, std_train)
    % Helper function to normalize the test data based 
    % on the train data stats.
    idx = find(abs(std_train) < eps); 
    if any(idx)
      std_train(idx) = 1;
    end
    Z = (X-mu_train)./std_train;
end