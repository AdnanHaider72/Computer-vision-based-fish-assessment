
close all
clear all

load 'PIFSnet_lgraph'
lgraph=lgraph_2;

Folder ='';
 
train_img_dir = fullfile(Folder,'Train');
imds = imageDatastore(train_img_dir); 
% %  
Val_img_dir = fullfile(Folder,'Val');%Training image directory
imdsVal = imageDatastore(Val_img_dir); 

classes = ["FISH","BG"]; %% Class names
labelIDs   = [1,0]; % Class id % Class id


train_label_dir = fullfile(Folder,'Train_GT');  %% Training label directory
pxds = pixelLabelDatastore(train_label_dir,classes,labelIDs);

Val_label_dir = fullfile(Folder,'Val_GT');  %% Training label directory
pxdsVal = pixelLabelDatastore(Val_label_dir,classes,labelIDs);

tbl = countEachLabel(pxds); 


frequency = tbl.PixelCount/sum(tbl.PixelCount); % frequency of each class

imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq;    % frequency balancing median 
%%% Training options %%%%%

dsVal = combine(imdsVal,pxdsVal);

options = trainingOptions('adam', ...
    'SquaredGradientDecayFactor',0.95, ...
    'GradientThreshold',5, ...
    'GradientThresholdMethod','global-l2norm', ...
    'Epsilon',1e-5, ...
    'InitialLearnRate',1e-3, ...
    'L2Regularization',0.0005, ...
    'ValidationData',dsVal,...
    'MaxEpochs',130, ...  
    'MiniBatchSize',6, ...
    'CheckpointPath',tempdir, ...
    'Shuffle','every-epoch', ...
    'VerboseFrequency',2, ...
    'OutputFcn',@(info)stopIfAccuracyNotImproving(info,25));
    

augment_data = imageDataAugmenter('RandXReflection',true,...
    'RandXTranslation',[-4 4],'RandYTranslation',[-4 4]); % optional data augmentation


training_data = pixelLabelImageDatastore(imds,pxds,...
    'DataAugmentation',augment_data); %% complete image+label data

[net, info] = trainNetwork(training_data,lgraph,options);% Train the network


function stop = stopIfAccuracyNotImproving(info,N)

stop = false;

% Keep track of the best validation accuracy and the number of validations for which
% there has not been an improvement of the accuracy.
persistent bestValAccuracy
persistent valLag

% Clear the variables when training starts.
if info.State == "start"
    bestValAccuracy = 0;
    valLag = 0;
    
elseif ~isempty(info.ValidationLoss)
    
    % Compare the current validation accuracy to the best accuracy so far,
    % and either set the best accuracy to the current accuracy, or increase
    % the number of validations for which there has not been an improvement.
    if info.ValidationAccuracy > bestValAccuracy
        valLag = 0;
        bestValAccuracy = info.ValidationAccuracy;
    else
        valLag = valLag + 1;
    end
    
   
    if valLag >= N
        stop = true;
    end
    
end

end

  

