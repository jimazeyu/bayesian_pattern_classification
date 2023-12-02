%% Calculate the error rates of different algorithms/strategies/datasets
% Load the alpha values and the datasets
load('Alpha.mat');
load('TrainingSamplesDCT_subsets_8.mat'); % Load the datasets

% Define the datasets for background (BG) and foreground (FG) for each D1, D2, D3, D4
datasets_BG = {D1_BG, D2_BG, D3_BG, D4_BG};
datasets_FG = {D1_FG, D2_FG, D3_FG, D4_FG};
prior_files = {'Prior_1.mat', 'Prior_2.mat'};

% Initialize an array to store error rates for 4 datasets, 2 strategies, each alpha value, and 3 error types
errors = zeros(4, 2, length(alpha), 3); 

% Loop over each dataset
for i = 1:length(datasets_BG)
    D_BG = datasets_BG{i};
    D_FG = datasets_FG{i};

    % Loop over each strategy
    for j = 1:length(prior_files)
        prior_file = prior_files{j};
        k = 1;

        % Loop over each alpha value
        for a = alpha
            % Compute error rates for MAP, Bayes, and ML decision rules
            errors(i, j, k, 1) = MAP_BDR(D_BG, D_FG, a, prior_file);
            errors(i, j, k, 2) = Bayes_BDR(D_BG, D_FG, a, prior_file);
            k = k + 1;
        end
        errors(i, j, :, 3) = ML_BDR(D_BG, D_FG);
    end
end

%% Plotting the results
for i = 1:length(datasets_BG)
    figure('Position', [100, 100, 1200, 400]); % Adjust the size of the figure

    % Plot results for each strategy
    for j = 1:length(prior_files)
        subplot(1, 2, j);
        plot(alpha, squeeze(errors(i, j, :, 1)), 'r-'); hold on; % MAP error rate
        plot(alpha, squeeze(errors(i, j, :, 2)), 'g-'); hold on; % Bayes error rate
        plot(alpha, squeeze(errors(i, j, :, 3)), 'b-'); hold on; % ML error rate
        set(gca, 'XScale', 'log'); % Set x-axis to logarithmic scale
        xticks(alpha); % Set x-axis ticks to each alpha value
        title(sprintf('Dataset %d, Strategy %d', i, j)); % Title for each subplot
        xlabel('Alpha'); % X-axis label
        ylabel('Error Rate'); % Y-axis label
        legend('MAP', 'Bayes', 'ML'); % Legend
    end
end

