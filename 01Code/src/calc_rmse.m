function rmse = calc_rmse(y, yfit)
    % Helper function to calculate Root Mean Squared Error
    y_resid = y - yfit;
    rmse = sqrt(mean(y_resid.^2)); 
end