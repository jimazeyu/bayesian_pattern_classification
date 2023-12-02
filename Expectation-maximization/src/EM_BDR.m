function [error] = EM_BDR(trainBG, trainFG, weightBG, weightFG, muBG, muFG, sigmaBG, sigmaFG, dimension, numComponents)
    % Estimate the prior probabilities
    [rowFG, ~] = size(trainFG);
    [rowBG, ~] = size(trainBG);
    priorFG = rowFG / (rowBG + rowFG);
    priorBG = rowBG / (rowBG + rowFG);

    % Read and preprocess the image
    img = im2double(imread('cheetah.bmp'));
    [height, width] = size(img);

    % Get zig-zag pattern
    pattern = readmatrix('Zig-Zag Pattern.txt') + 1;

    % Initialize result mask
    maskResult = zeros(height-7, width-7);

    % Loop over the image
    for i = 1:(height - 7)
        for j = 1:(width - 7)
            block = img(i:(i+7), j:(j+7));
            dctBlock = dct2(block);
            zigzagBlock = zeros(1, 64);
            for m = 1:8
                for n = 1:8
                    zigzagBlock(pattern(m,n)) = dctBlock(m,n);
                end
            end

            % Select the corresponding parameters for FG and BG
            muFgD = muFG(:, 1:dimension); 
            muBgD = muBG(:, 1:dimension);
            sigmaFgD = sigmaFG(1:dimension, 1:dimension, 1:numComponents);
            sigmaBgD = sigmaBG(1:dimension, 1:dimension, 1:numComponents);
            
            % Compute the likelihoods for FG and BG
            pxYFg = myGmmPdf(zigzagBlock(1:dimension), muFgD, sigmaFgD, weightFG);
            pxYBg = myGmmPdf(zigzagBlock(1:dimension), muBgD, sigmaBgD, weightBG);

            % Apply BDR
            if pxYFg * priorFG > pxYBg * priorBG
                maskResult(i, j) = 1;
            end
        end
    end

    % % Display the masks
    % figure;
    % imshow(maskResult);
    % title('Classification Results');

    % Compute classification error against ground truth
    maskGT = im2double(imread('cheetah_mask.bmp'));
    error = sum(sum(maskResult ~= maskGT(1:end-7, 1:end-7))) / numel(maskResult);
end

% My own GMM function
function pdfValue = myGmmPdf(x, mu, sigma, weight)
    numComponents = size(mu, 1);
    pdfValue = 0;
    for i = 1:numComponents
        diff = x - mu(i,:);
        exponent = -0.5 * (diff / sigma(:,:,i)) * diff';
        coefficient = 1 / sqrt(((2 * pi)^length(x)) * det(sigma(:,:,i)));
        pdfValue = pdfValue + weight(i) * coefficient * exp(exponent);
    end
end