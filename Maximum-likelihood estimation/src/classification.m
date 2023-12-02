%% (a)Estimate the prior probabilities
load('TrainingSamplesDCT_8_new.mat')
[rowFG, columnFG] = size(TrainsampleDCT_FG);
[rowBG, columnBG] = size(TrainsampleDCT_BG);
priorFG = rowFG / (rowBG + rowFG);
priorBG = rowBG / (rowBG + rowFG);
% Plot the number of samples for each class
figure;
subplot(2,1,1);
bar([rowFG, rowBG]);
set(gca, 'XTickLabel', {'Cheetah (Foreground)', 'Grass (Background)'});
ylabel('Number of Samples');
title('Number of Samples for Each Class');
grid on;
% Plot the prior probability for each class
subplot(2,1,2);
bar([priorFG, priorBG]);
set(gca, 'XTickLabel', {'Cheetah (Foreground)', 'Grass (Background)'});
ylabel('Prior Probability');
title('Prior Probability for Each Class');
ylim([0 1]);
grid on;


%% (b)Plot the marginal densities for the two classes
% Compute the MLE for parameters (mean and covariance) for both classes
meanFG = mean(TrainsampleDCT_FG);
meanBG = mean(TrainsampleDCT_BG);
covFG = cov(TrainsampleDCT_FG);
covBG = cov(TrainsampleDCT_BG);

% Create 64 plots for the marginal densities
figure;

for k = 1:64
    subplot(8,8,k);

    % Determine the range for x centered at 0
    absMax = max(abs([TrainsampleDCT_FG(:, k); TrainsampleDCT_BG(:, k)]));
    % Gaussian PDF for the k-th DCT coefficient for Cheetah and Grass
    x = linspace(-absMax, absMax, 100);
    yFG = my_normpdf(x, meanFG(k), sqrt(covFG(k,k)));
    yBG = my_normpdf(x, meanBG(k), sqrt(covBG(k,k)));
    
    plot(x, yFG, 'r-', x, yBG, 'b-');
    title(['Feature ' num2str(k)]);
    % legend('Cheetah','Grass');
    xlim([-absMax, absMax]);
end

sgtitle('Marginal Densities for each DCT Coefficient'); 

% Seletecting the features
bestFeatures = [1,14,17,21,32,40,41,45];
worstFeatures = [3,4,5,59,60,62,63,64];

% Plotting best features
figure;
for i = 1:8
    k = bestFeatures(i);
    subplot(2,4,i);
    
    % Determine the range for x centered at 0
    absMax = max(abs([TrainsampleDCT_FG(:, k); TrainsampleDCT_BG(:, k)]));
    x = linspace(-absMax, absMax, 100);
    
    yFG = my_normpdf(x, meanFG(k), sqrt(covFG(k,k)));
    yBG = my_normpdf(x, meanBG(k), sqrt(covBG(k,k)));
    
    plot(x, yFG, 'r-', x, yBG, 'b-');
    title(['Feature ' num2str(k)]);
    % legend('Cheetah','Grass');
    xlim([-absMax, absMax]);
end
sgtitle('Best 8 Features');

% Plotting worst features
figure;
for i = 1:8
    k = worstFeatures(i);
    subplot(2,4,i);
    
    % Determine the range for x centered at 0
    absMax = max(abs([TrainsampleDCT_FG(:, k); TrainsampleDCT_BG(:, k)]));
    x = linspace(-absMax, absMax, 100);
    
    yFG = my_normpdf(x, meanFG(k), sqrt(covFG(k,k)));
    yBG = my_normpdf(x, meanBG(k), sqrt(covBG(k,k)));
    
    plot(x, yFG, 'r-', x, yBG, 'b-');
    title(['Feature ' num2str(k)]);
    % legend('Cheetah','Grass');
    xlim([-absMax, absMax]);
end
sgtitle('Worst 8 Features');


%% (c)Separate the foreground and background with 64 and 8 features
% Read and preprocess the image
img = imread('cheetah.bmp');
imgDouble = im2double(img);
[height, width] = size(imgDouble);

% Get pattern index
pattern = readmatrix('Zig-Zag Pattern.txt') + 1;

% Calculate the threshold
thresStar = priorBG / priorFG;

% Compute P(x|cheetah) and P(x|grass) and make a decision
mask64 = zeros(height, width);
mask8 = zeros(height, width);

% Loop over the image
for i = 1:height-7
    for j = 1:width-7
        block = imgDouble(i:i+7, j:j+7); % 8x8 block
        dctBlock = dct2(block);
        zigzag = zeros(1,64);
        for m = 1:8
            for n = 1:8
                zigzag(pattern(m,n)) = dctBlock(m,n);
            end
        end

        % 64-dimensional classification
        Px_yFG64 = my_mvnpdf(zigzag, meanFG, covFG);
        Px_yBG64 = my_mvnpdf(zigzag, meanBG, covBG);
        if Px_yFG64 / Px_yBG64 > thresStar
            mask64(i, j) = 1;
        end

        % 8-dimensional classification
        Px_yFG8 = my_mvnpdf(zigzag(bestFeatures), meanFG(bestFeatures), covFG(bestFeatures,bestFeatures));
        Px_yBG8 = my_mvnpdf(zigzag(bestFeatures), meanBG(bestFeatures), covBG(bestFeatures,bestFeatures));
        if Px_yFG8 / Px_yBG8 > thresStar
            mask8(i, j) = 1;
        end
    end
end

% Display the masks
figure;
subplot(1,2,1);
imshow(mask64);
title('Classification using 64 Features');

subplot(1,2,2);
imshow(mask8);
title('Classification using Best 8 Features');

%% Compute error
maskGT = imread('cheetah_mask.bmp');
maskGT = im2double(maskGT);
error64 = sum(sum(mask64 ~= maskGT)) / (height * width);
error8 = sum(sum(mask8 ~= maskGT)) / (height * width);

Pfg_fg64 = sum(sum(mask64 == 1 & maskGT==1))/sum(sum(maskGT==1));
Pfg_bg64 = sum(sum(mask64 == 1 & maskGT==0))/sum(sum(maskGT==0));
e64 = Pfg_bg64*priorBG + (1-Pfg_fg64)*priorFG;

Pfg_fg8 = sum(sum(mask8 == 1 & maskGT==1))/sum(sum(maskGT==1));
Pfg_bg8 = sum(sum(mask8 == 1 & maskGT==0))/sum(sum(maskGT==0));
e8 = Pfg_bg8*priorBG + (1-Pfg_fg8)*priorFG;

%% Define functions to calculate the PDF for Normal Distribution
function single_pdf = my_normpdf(x, mu, sigma)
    single_pdf = 1 / (sqrt(2 * pi) * sigma) * exp(-(x - mu).^2 / (2 * sigma^2));
end

function multi_pdf = my_mvnpdf(x, mu, Sigma)
    k = length(mu);
    multi_pdf = 1 / ((2 * pi)^(k/2) * sqrt(det(Sigma))) * exp(-0.5 * (x - mu) * inv(Sigma) * (x - mu)');
end
