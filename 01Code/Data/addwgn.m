function Y = addwgn(X, SNR_db)
    % Add White gaussian noise, with SNR in dB to a matrix mxn with m 
    % signals over n timesteps, where each signal is assumed to have an
    % individual signal to noise ratio
    % Logic from: 
    % https://www.gaussianwaves.com/2015/06/how-to-generate-awgn-noise-in-matlaboctave-without-using-in-built-awgn-function/
    % The signal strength is measured to match the 'noise' in SNR
    % Variable naming and logic following 
    % https://en.wikipedia.org/wiki/Additive_white_Gaussian_noise
    % Determine the energy of the signal
    % https://en.wikipedia.org/wiki/Energy_(signal_processing)
    
    Y = zeros(size(X));
    n = size(X,1);
    l = size(X,2);
    for i=1:n
        P_s = sum(X(i,:).^2)/l;         % Power of the signal
        N_0 = P_s/(10^(SNR_db/10));     % req. power spectral dens. noise
        Z = sqrt(N_0)*randn(1,l);
        Y(i,:) = X(i,:) + Z;
    end