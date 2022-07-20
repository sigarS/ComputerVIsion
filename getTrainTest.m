function [train_models, test_ims, test_labels, class_rep] = getTrainTest(path, train_ratio)
       
    %{
        This script takes in the folder path containing the sub-folders for
        each image set, and generates training and test data sets according
        to a desired ratio. The training sets are then passed to
        getClassModel() to generate the class models.

        The class models and labels, along with test images and labels, are
        then retured as variables for use in test scripts or detection app.
    %}

    %INPUTS
    % path <- image root folder path
    
    % train_ratio <- percentage of images to use for training, rest are
    % used in the test set (must be between (0,1))
    
    % OUTPUTS
    % train_models <- structure containing each class model
    % class_rep <- class model label
    % test_ims <- cell containing each test image
    % test_labels <- cell containing each test image label
    
    %Create the image datastore containing all images, using the folder
    %names as the labels
    ds = datastore(path, 'IncludeSubfolders', true, 'Type', 'image',...
    'LabelSource', 'foldernames');
    
    %Split the datastore into separate datastores by the desired ratio
    %using splitEachLabel(). Setting 'randomized' enables random sampling.
    if (0<train_ratio) && (train_ratio<1)
        [train_ds, test_ds] = splitEachLabel(ds, train_ratio, 'randomized');
    else
        return
    end
    
    %Store the training images to a vector
    train_ims = transpose(readall(train_ds));
    
    %Store the training labels to a vector
    train_labels = transpose(train_ds.Labels);
    
    %Get total (unique) classes
    classes = unique(train_labels);
    
    %Shuffle the test datastore for randomisation
    test_ds = shuffle(test_ds);
    
    %Read the test images into vector
    test_ims = transpose(readall(test_ds));
    
    %Get the test labels
    test_labels = transpose(test_ds.Labels);
    
    %The toal number of available classes
    n_classes = length(classes);
    
    
    %This loop iterates over each class, and generates the class models for
    %each.
    for i=1:n_classes
        temp_class = train_ims(train_labels==classes(i));
        train_models.(string(classes(i))) = getClassModel(temp_class);
        class_rep.(string(classes(i))) = temp_class{1};
    end
    
    
    
    
    


