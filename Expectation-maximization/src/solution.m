%% Load the data
load('TrainingSamplesDCT_8_new.mat'); % Assume TrainingSamplesDCT_new_8.mat data is loaded into the workspace

%% Train EM models
% Train a set of five GMMs with C=8 components
for i = 1:5
    % Train and save GMM parameters for the i-th model
    trainEM(8, TrainsampleDCT_BG, TrainsampleDCT_FG, i); % Labels 1-5
end

% Train a set of GMMs for different values of C components
C_values = [1, 2, 4, 8, 16, 32];
for C = C_values
    % Train and save GMM parameters for each C value, labeled 0
    trainEM(C, TrainsampleDCT_BG, TrainsampleDCT_FG, 0); 
end

%% (a) 25 random classifiers
C = 8; % Number of components in the mixture model
dimensions = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]; % Array of dimensions to be considered
numMixtures = 5; % Number of GMMs to be trained for each class
errorRates = zeros(numMixtures^2, length(dimensions)); % Matrix to store error rates
mixturePairs = combvec(1:numMixtures, 1:numMixtures)'; % Generate all possible mixture pairs

% Calculate error rates for each dimension
for d = 1:length(dimensions)
    dimension = dimensions(d);
    disp(['Starting dimension: ', num2str(dimension)]);

    % Loop through all possible pairs of mixtures
    for mixPair = 1:size(mixturePairs, 1)
        mixBG = mixturePairs(mixPair, 1); % Background mixture index
        mixFG = mixturePairs(mixPair, 2); % Foreground mixture index

        % Load GMM parameters for background and foreground
        load(sprintf('trainedGMM_Comp=%d_Label=%d.mat', C, mixBG), 'weightsBG', 'meansBG', 'covariancesBG');
        load(sprintf('trainedGMM_Comp=%d_Label=%d.mat', C, mixFG), 'weightsFG', 'meansFG', 'covariancesFG');

        % Compute error rates using the EM_BDR function
        error = EM_BDR(TrainsampleDCT_BG, TrainsampleDCT_FG, weightsBG, weightsFG, meansBG(:, 1:dimension), meansFG(:, 1:dimension), covariancesBG(1:dimension, 1:dimension, :), covariancesFG(1:dimension, 1:dimension, :), dimension, C);

        % Store the error rate
        errorRates(mixPair, d) = error;
    end
end

%% Plot the error rates
figure;
for mixPair = 1:size(mixturePairs, 1)
    plot(dimensions, errorRates(mixPair, :), '-o'); % Plot a line for each mixture pair
    hold on; % Hold the figure for multiple plots
end
hold off;
xlabel('Dimensions');
ylabel('Error Rate');
title('Error Rate vs. Dimension for different GMM mixture pairs');
legendCell = cellstr(num2str(mixturePairs, 'BG%d-FG%d')); % Create a legend
legend(legendCell);

%% (b) Classifiers with different component numbers
% Initialize parameters
dimensions = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]; % Array of dimensions to be considered
C_values = [1, 2, 4, 8, 16, 32]; % Array of numbers of mixture components
errorRatesC = zeros(length(C_values), length(dimensions)); % Matrix to store error rates for different C values

% Loop to calculate error rates for each value of C
for idx = 1:length(C_values)
    C = C_values(idx); % Current number of mixture components
    disp(['Starting component num: ', num2str(C)]);

    % Loop through different dimensions
    for d = 1:length(dimensions)
        dimension = dimensions(d);

        % Load the model parameters, assuming models are saved with iteration label 0
        load(sprintf('trainedGMM_Comp=%d_Label=0.mat', C), 'weightsBG', 'meansBG', 'covariancesBG');
        load(sprintf('trainedGMM_Comp=%d_Label=0.mat', C), 'weightsFG', 'meansFG', 'covariancesFG');

        % Compute and store the error rate
        errorRatesC(idx, d) = EM_BDR(TrainsampleDCT_BG, TrainsampleDCT_FG, weightsBG, weightsFG, meansBG(:, 1:dimension), meansFG(:, 1:dimension), covariancesBG(1:dimension, 1:dimension, :), covariancesFG(1:dimension, 1:dimension, :), dimension, C);
    end
end

%% Plot the error rates
figure;
for idx = 1:length(C_values)
    plot(dimensions, errorRatesC(idx, :), '-o', 'DisplayName', sprintf('C=%d', C_values(idx)));
    hold on; % Hold on to plot multiple lines on the same figure
end
hold off; % Release the hold to finish plotting
xlabel('Dimension');
ylabel('Error Rate');
title('Error Rate vs. Dimension for Different Numbers of Mixture Components');
legend show;

