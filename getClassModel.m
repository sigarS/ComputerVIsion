function X = getClassModel(trainSet)
    % This function returns a class model from a training set of images. It
    % assumes the set of images is stored in a 1 x trainSize cell, and that
    % all images are of the same dimensions.

    % get size of training set
    trainSize = size(trainSet);
    trainSize = trainSize(2);
    
    % define the sampling rate for downsampling along both axes
    sRateX = 2;
    sRateY = 2;
    
    % get size of images in training set (assumes all images are the same
    % size)
    [nrows, ncols] = size(trainSet{1});

    % get number of sampled rows/cols and size of image vector
    nSampledRows = floor(nrows/sRateX);
    nSampledCols = floor(ncols/sRateY);
    vectorSize = nSampledRows * nSampledCols;
    
    % initialise class model
    X = zeros(vectorSize, trainSize);
    
    % get image vector for each training image
    for m=1:trainSize
        maxVal = 0; % needed to normalise image vectors
        for j=1:vectorSize
            row = (mod(j-1,nSampledRows) + 1)*sRateY;
            col = ceil(j/nSampledRows)*sRateX;
            X(j,m) = trainSet{m}(row,col);
            % check if max val needs updating
            if X(j,m) > maxVal
                maxVal = X(j,m);
            end
        end
        % normalise image vector
        X(:,m) = X(:,m) ./ maxVal;
    end
end