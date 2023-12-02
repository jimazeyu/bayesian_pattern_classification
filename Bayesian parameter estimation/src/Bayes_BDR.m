function [error] = Bayes_BDR(trainBG, trainFG, alpha, prior)
    load('TrainingSamplesDCT_subsets_8.mat');
    % Estimate the prior probabilities
    [rowFG, columnFG] = size(trainFG);
    [rowBG, columnBG] = size(trainBG);
    priorFG = rowFG / (rowBG + rowFG);
    priorBG = rowBG / (rowBG + rowFG);
    
    % Compute the parameters for Bayes
    load(prior);
    covFG = cov(trainFG);
    covBG = cov(trainBG);
    
    covPrior = diag(alpha * W0);
    
    meanFG = mean(trainFG);
    meanFG = covPrior * inv(covPrior + covFG/size(trainFG, 1)) * ...
        meanFG' + covFG/size(trainFG, 1) * inv(covPrior + covFG/size(trainFG, 1)) * mu0_FG';
    covFG = covFG + covPrior * inv(covPrior + covFG/size(trainFG,1)) * covFG/size(trainFG,1);

    meanBG = mean(trainBG);
    meanBG = covPrior * inv(covPrior + covBG/size(trainBG, 1)) * ...
        meanBG' + covBG/size(trainBG, 1) * inv(covPrior + covBG/size(trainBG, 1)) * mu0_BG';
    covBG = covBG + covPrior * inv(covPrior + covBG/size(trainBG,1)) * covBG/size(trainBG,1);
    
    % Read and preprocess the image
    img = imread('cheetah.bmp');
    imgDouble = im2double(img);
    [height, width] = size(imgDouble);
    
    % Get pattern index
    pattern = readmatrix('Zig-Zag Pattern.txt') + 1;
    
    % Calculate the threshold
    thresStar = priorBG / priorFG;
    
    % Loop over the image and make a decision
    maskRes = zeros(height, width);
    for i = 1:height-7
        for j = 1:width-7
            block = imgDouble(i:i+7, j:j+7);
            dctBlock = dct2(block);
            zigzag = zeros(1, 64);
            for m = 1:8
                for n = 1:8
                    zigzag(pattern(m,n)) = dctBlock(m,n);
                end
            end
    
            Px_yFG = my_mvnpdf(zigzag, meanFG', covFG);
            Px_yBG = my_mvnpdf(zigzag, meanBG', covBG);
            if Px_yFG / Px_yBG > thresStar
                maskRes(i, j) = 1;
            end
    
        end
    end
    
    % % Display the masks
    % figure;
    % imshow(maskRes);
    % title('Classification Results');
    
    % Compute error
    maskGT = imread('cheetah_mask.bmp');
    maskGT = im2double(maskGT);
    error = sum(sum(maskRes ~= maskGT)) / (height * width);
end
%% Define the function to calculate the PDF for Normal Distribution
function multi_pdf = my_mvnpdf(x, mu, Sigma)
    k = length(mu);
    multi_pdf = 1 / ((2 * pi)^(k/2) * sqrt(det(Sigma))) * exp(-0.5 * (x - mu) * inv(Sigma) * (x - mu)');
end
