
close all
clear all

load 'PFFSnet_lgraph'
lgraph=lgraph_2;
Folder ='';
train_img_dir = fullfile(Folder,'Train');
imds = imageDatastore(train_img_dir); 
% %  
Val_img_dir = fullfile(Folder,'Val');
imdsVal = imageDatastore(Val_img_dir); 

classes = ["FISH","BG"]; 
labelIDs   = [255,0]; 

train_label_dir = fullfile(Folder,'Train_GT');  
pxds = pixelLabelDatastore(train_label_dir,classes,labelIDs);

Val_label_dir = fullfile(Folder,'Val_GT');  
pxdsVal = pixelLabelDatastore(Val_label_dir,classes,labelIDs);

tbl = countEachLabel(pxds); 


frequency = tbl.PixelCount/sum(tbl.PixelCount); 

imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq; 

%%% Training options %%%%%

dsVal = combine(imdsVal,pxdsVal);

options = trainingOptions('adam', ...
    'SquaredGradientDecayFactor',0.95, ...
    'GradientThreshold',5, ...
    'GradientThresholdMethod','global-l2norm', ...
    'Epsilon',1e-6, ...
    'InitialLearnRate',1e-3, ...
    'L2Regularization',0.0005, ...
    'ValidationData',dsVal,...
    'MaxEpochs',130, ...  
    'MiniBatchSize',5, ...
    'CheckpointPath',tempdir, ...
    'Shuffle','every-epoch', ...
    'VerboseFrequency',2, ...
    'OutputFcn',@(info)stopIfAccuracyNotImproving(info,25));
    

augment_data = imageDataAugmenter('RandXReflection',true,...
    'RandXTranslation',[-2 2],'RandYTranslation',[-2 2]); % optional data augmentation


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
    
    % If the validation lag is at least N, that is, the validation accuracy
    % has not improved for at least N validations, then return true and
    % stop training.
    if valLag >= N
        stop = true;
    end
    
end

end

  
