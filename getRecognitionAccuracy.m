%This script is essentially the same as main.m, but repeats the process for
%a given number of trials, where the data is randomly split each time
%making the testing similar to a cross-folds validation strategy.

clear all; close all; clc

fpath = 'C:\Users\jerem\Documents\MATLAB\Project\FaceDataset';     %set folder path
train_ratio = 0.5;                                              %set train ratio


T = 6; % number of trials 

accuracies = zeros(T,1);

for t = 1:T
    
    correct=0;                                  %set counter for monitoring accuracy
    [train_models, test_ims, test_labels] = getTrainTest(fpath, train_ratio);  %get training models, test images, and test labels

    fnames = fieldnames(train_models);          %get field names of train_model structure
    %loop through each test image in test_ims
    % then compute the distance between test image and each training class model
    % , take the min distance field name as our predicted class, evaluate
    % against ground truth.
    for i = 1:length(test_ims)
        y = getClassModel(test_ims(i));
        for j = 1:numel(fnames)
            X = train_models.(string(fnames(j)));
            y_hat = X*inv((transpose(X)*X))*transpose(X)*y;
            distance(j) = sqrt(sum((y(:) - y_hat(:)) .^ 2));
        end
        pred = fnames(find(distance == min(distance)));
        fprintf('For test image of class: %s\n', string(test_labels(i)));       %ground truth label
        fprintf('Predicted class is: %s\n', string(pred(1)));                   %our predicted class

        %success/failure reporting
        if string(pred(1)) == string(test_labels(i))
            fprintf('Success! Correct class found.\n');
            correct = correct + 1;
        else
            fprintf('Failed! Incorrect class predicted.\n');
        end
    end

    % simple accuracy reporting
    fprintf('Classification accuracy: %g out of %g correct.\n', correct, length(test_labels));
    
    % calculate percentage accuracy
    accuracy = correct/length(test_labels)*100;
    accuracies(t) = accuracy;

end

avgAcc = mean(accuracies);

disp('\nRecognition Accuracies\n');
for t = 1:T
    fprintf('Trial %d: %.2f\n', t, accuracies(t));
end
fprintf('Avg: %.2f\n', avgAcc);