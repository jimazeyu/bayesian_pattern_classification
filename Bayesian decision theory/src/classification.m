%% Estimate the prior probabilities
load('TrainingSamplesDCT_8.mat')
[rowFG, columnFG] = size(TrainsampleDCT_FG);
[rowBG, columnBG] = size(TrainsampleDCT_BG);
priorFG = rowFG / (rowBG + rowFG);
priorBG = rowBG / (rowBG + rowFG);
%% Compute and plot index histograms
% Compute the conditional probabilities
indexBinFG = zeros(columnFG,1);
for i = 1:rowFG
    row = TrainsampleDCT_FG(i, :);
    [~, sortedIndices] = sort(row, 'descend');
    indexBinFG(sortedIndices(2)) = indexBinFG(sortedIndices(2)) + 1;
end
conditionalProbsFG = indexBinFG / sum(indexBinFG(:));

indexBinBG = zeros(columnBG,1);
for i = 1:rowBG
    row = TrainsampleDCT_BG(i, :);
    [~, sortedIndices] = sort(row, 'descend');
    indexBinBG(sortedIndices(2)) = indexBinBG(sortedIndices(2)) + 1;
end
conditionalProbsBG = indexBinBG / sum(indexBinBG(:));

% Plot the histograms
figure;

subplot(2, 1, 1);
indicesFG = 1:length(conditionalProbsFG);
bar(indicesFG, conditionalProbsFG);
xlabel('x(index)');
ylabel('P(x|cheetah)');
title('Conditional Probobilities of FG');
xlim([min(indicesFG)-1, max(indicesFG)+1]);

subplot(2, 1, 2);
indicesBG = 1:length(conditionalProbsBG);
bar(indicesBG, conditionalProbsBG);
xlabel('x(index)');
ylabel('P(x|grass)');
title('Conditional Probobilities of BG');
xlim([min(indicesBG)-1, max(indicesBG)+1]);

%% Compute DCT coefficient and separate the foreground and background
% Read and preprocess the image
img = imread('cheetah.bmp');
imgDouble = im2double(img);
[height, width] = size(imgDouble);
extendedImg = zeros(height + 7, width + 7);
extendedImg(1:end-7, 1:end-7) = imgDouble;

% Classify BG and FG
pattern = readmatrix('Zig-Zag Pattern.txt') + 1;
indexMap = zeros(64, 1);% Map index for accelerated algorithm speed
for i = 1:8
    for j = 1:8
        indexMap(j + (i-1)*8) = pattern(i, j); 
    end
end

predictedMask = zeros(height, width);
thresStar = priorBG / priorFG;
for i = 1:height
    for j = 1:width
        block = extendedImg(i:i+7, j:j+7);
        dctBlock = abs(dct2(block));
        % Find the index of the second largest value
        dctBlock = dctBlock';
        dctLine = dctBlock(:); % Flatten the matrix row by row
        [~, index] = sort(dctLine, 'descend');
        indexMapped = indexMap(index(2));
        % Calculate the probability
        thres =conditionalProbsFG(indexMapped) / conditionalProbsBG(indexMapped);
        if thres > thresStar
            predictedMask(i, j) = 1;
        end
    end
end

figure;
imagesc(predictedMask);
colormap(gray(255));

%% Compute error
maskGT = imread('cheetah_mask.bmp');
maskGT = im2double(maskGT);
error = sum(sum(predictedMask ~= maskGT)) / (height * width);