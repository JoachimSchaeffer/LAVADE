function b = ridge_regression(y, X, k)
    % Custom implementation of the ridge regression 
    % Reason: The matlab ridge function ALWAYS standardized the
    % data. For this demonstrator we want to able to compare RR
    % also for the non-standardized case.

    Z = X-mean(X);
    y = y-mean(y);
    % Use closed form solution for this demonstrator, because speed doesnt
    % play an important role (small datasets)
    n_col = size(X,2);
    b = (Z'*Z+k*eye(n_col))\(Z'*y);
end