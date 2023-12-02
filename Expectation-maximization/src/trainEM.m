function [] = trainEM(numComponents, dataBG, dataFG, iterationLabel)
    numIterations = 2000; % Number of iterations for the EM algorithm
    numFeatures = size(dataBG, 2); % Assuming the number of features is the number of columns in dataBG

    % Initialize parameters for background and foreground
    [weightsBG, meansBG, covariancesBG] = initializeGMMParameters(numComponents, dataBG);
    [weightsFG, meansFG, covariancesFG] = initializeGMMParameters(numComponents, dataFG);

    % Train the GMM for background data
    fprintf('Starting EM for BG with %d components, iteration %d\n', numComponents, iterationLabel);
    [weightsBG, meansBG, covariancesBG] = runEM(dataBG, numComponents, numIterations, weightsBG, meansBG, covariancesBG);

    % Train the GMM for foreground data
    fprintf('Starting EM for FG with %d components, iteration %d\n', numComponents, iterationLabel);
    [weightsFG, meansFG, covariancesFG] = runEM(dataFG, numComponents, numIterations, weightsFG, meansFG, covariancesFG);

    % Save the trained model parameters
    saveFileName = sprintf('trainedGMM_Comp=%d_Label=%d.mat', numComponents, iterationLabel);
    save(saveFileName, 'weightsBG', 'meansBG', 'covariancesBG', 'weightsFG', 'meansFG', 'covariancesFG');
end

function [weights, means, covariances] = initializeGMMParameters(numComponents, data)
    % Initialize parameters for the GMM
    [n, d] = size(data);
    weights = rand (numComponents ,1) ;
    weights = weights / sum (weights) ;
    means = data(randperm(n, numComponents), :);
    covariances = repmat(diag(diag(rand(d))), [1, 1, numComponents]);
end

function [weights, means, covariances] = runEM(data, numComponents, numIterations, weights, means, covariances)
    % Run the EM algorithm for GMM
    n = size(data, 1);
    numFeatures = size(data, 2);
    
    for iteration = 1:numIterations
        % E-step: compute responsibilities
        responsibilities = zeros(n, numComponents);
        for j = 1:numComponents
            % Calculate the covariance matrix for the j-th component
            covarianceMatrix = reshape(covariances(:, :, j), numFeatures, numFeatures);
            % Compute the probability density function for each data point
            responsibilities(:, j) = weights(j) * mvnpdf(data, means(j, :), covarianceMatrix);
        end
        responsibilities = responsibilities ./ sum(responsibilities, 2);

        % M-step: update weights, means, and covariances
        for j = 1:numComponents
            weight = sum(responsibilities(:, j)) / n;
            mean = (responsibilities(:, j)' * data) / sum(responsibilities(:, j));
            centeredData = data - mean;
            covariance = (centeredData' * (centeredData .* responsibilities(:, j))) / sum(responsibilities(:, j));
            
            weights(j) = weight;
            means(j, :) = mean;
            covariances(:, :, j) = diag(diag(covariance));
        end
        
    end
end
