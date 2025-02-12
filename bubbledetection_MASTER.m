close all; 
clear; 
clc

% ===========================================================
%                VIDEO PROCESSING AND FRAME SPLITTING
% ===========================================================



% ================================================
%                READ VIDEO FILE
% ================================================

% Supply input video file
vidReader = VideoReader('purple_video.mp4');

% Create output folder for frames - supply the folder name for storage
outputFolder = 'purple_video_frames';
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Initialize frame index and preallocate cell array for frames
frameIndex = 0;
allFrames = {}; % Cell array to store frames

% Read and process each frame
while hasFrame(vidReader)
    % Read video frame
    vidFrame = readFrame(vidReader);
    
    % Increment frame index
    frameIndex = frameIndex + 1;
    
    % Save the frame as an image
    frameFileName = fullfile(outputFolder, sprintf('frame_%04d.png', frameIndex));
    imwrite(vidFrame, frameFileName);
    
    % Store the frame in the cell array
    allFrames{frameIndex} = vidFrame; % Store the raw RGB data
end

% =========================================================
%                SAVE FRAME DIRECTORY TO .mat FILE
% =========================================================
% Specify the target directory
targetDir = 'purple_video_frames';

% Get ONLY image files from directory (exclude folders)
f = dir(fullfile(targetDir, '*.png'));
files = {f.name};

% Preallocate cell array
Im = cell(1, numel(files));

for k = 1:numel(files)
    % Build full file path
    fullPath = fullfile(targetDir, files{k});
    
    % Read and process image
    Im{k} = im2double(rgb2gray(imread(fullPath)));
end

save('purple_video_frames.mat', 'Im', '-v7.3');


%%
% =======================================================
%              BUBBLE BINARIZATION AND DETECTION
% =======================================================
close all;
clear;
clc;

% Adapted and modified from Mike X Cohen's Ca+ Imaging tutorial
% https://sincxpress.com/neuroscience/
% =======================================================
%              LOADING AND POPULATING DATA MATRIX
% =======================================================

load("purple_video_frames.mat")
numFrames = numel(Im)

npnts = numFrames;

data = zeros([ size(Im{1}) npnts]);

% populate the matrix with data, one slice at a time
for i=1:npnts
    data(:,:,i) = Im{i};    
end

% ========================================================
%                    REMOVING BACKGROUND SIGNAL
% ========================================================

for frameIdx = 1:3
    currentFrame = Im{frameIdx};

    % Computing average map
    avemap = currentFrame;
    %avemap = mean(data, 3);
    
    % normalize to range of [0 1] using MinMax Scaling
    %avemap = avemap - min(avemap(:)); % to vectorize the avemap
    %avemap = avemap / max(avemap(:));
    % here values in avemap are already normalized
    
    % apply MATLAB's local adaptive histogram equalization
    avemap = adapthisteq(avemap);
    % estimate the background as a fuzzy version of the image
    background = imgaussfilt(avemap,10);
    % create the boosted-SNR map by subtracting the 'background'
    foreground = avemap - background;
    % create the binarized image
    set(gca,'clim', [0.15 0.2])
    threshval = 0.2; % pick a threshold here based on inspecting the foreground map
    threshimg = foreground > threshval; % create a boolean map of all pixels brighter than the threshold
    islands = bwconncomp(threshimg);
    
    % ========================================================
    %               PER-FRAME VISUALIZATION
    % ========================================================
    figure('Visible', 'off', 'Position', [100, 100, 800, 600]);
    colormap hot
    
    % Calculate bubble properties
    props = regionprops(threshimg, 'MajorAxisLength', 'Eccentricity');
    eccentricity = [props.Eccentricity];
    majoraxis = [props.MajorAxisLength];
    ecc = mean(eccentricity);
    ma = mean(majoraxis);
    dia = ma * 0.2645833333;
    num_bubbles = islands.NumObjects;
    
    % Create subplots with added text
    for i = 1:4
        subplot(2,2,i)
        switch i
            case 1
                imagesc(avemap)
                title(['Frame ' num2str(frameIdx) ' - Mean'])
            case 2
                imagesc(background)
                title('Background Signal')
            case 3
                imagesc(foreground)
                title('Isolated Features')
                set(gca, 'clim', [0.15 0.2])
            case 4
                imagesc(threshimg)
                title('Binarized Detection')
                text(0.5, -0.1, sprintf('Mean Eccentricity: %.2f\nMean Diameter: %.2f\nNumber of Bubbles: %d', ecc, dia, num_bubbles), ...
                'Units', 'normalized', ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'top', ...
                'FontSize', 8)
        end
        axis square
    end
    
    % Save visualization in the new subdirectory
    frameFolder = 'purple_video_frames/analysis';
    if ~exist(frameFolder, 'dir')
        mkdir(frameFolder);
    end
    saveas(gcf, fullfile(frameFolder, sprintf('frame_%04d_analysis.png', frameIdx)));
    close gcf
    
    % Display progress
    fprintf('Processed frame %d/%d\n', frameIdx, numFrames);

%     % Use only if required (for extremely noisy background)
%     % %% 
%     % % ========================================================
%     % %        CONTIGUITY BASED FILTERING OF BINARIZED IMAGE
%     % % ========================================================
%     % 
%     % % get cluster information
%     % islands = bwconncomp(threshimg);
%     % 
%     % % identify the cluster sizes
%     % cellsizes = cellfun(@length,islands.PixelIdxList);
%     % 
%     % % find small and large cells
%     % cells2cut = cellsizes < 4 | cellsizes>20;
%     % 
%     % % and remove those cells
%     % islands.PixelIdxList(cells2cut) = [];
%     % 
%     % % update the number of remaining clusters
%     % islands.NumObjects = numel(islands.PixelIdxList);
%     % 
%     % 
%     % % finally, recreate the threshold image without rejected clusters 
%     % threshimgFilt = false(size(avemap));
%     % for i=1:islands.NumObjects
%     %     threshimgFilt(islands.PixelIdxList{i}) = true;
%     % end
%     % 
%     % % visualize
%     % figure(4), clf
%     % subplot(121)
%     % imagesc(threshimg)
%     % axis square
%     % title('binarized (original)')
%     % colormap gray
%     % 
%     % 
%     % 
%     % % show again for comparison
%     % subplot(122)
%     % imagesc(threshimgFilt)
%     % axis square
%     % title('binarized (filtered)')
%     
%     % Save visualization in the new subdirectory
    % filterFolder = 'filtered_images';
    % if ~exist(filterFolder, 'dir')
    %     mkdir(filterFolder);
    % end
    % saveas(gcf, fullfile(filterFolder, sprintf('frame_%04d_filter.png', frameIdx)));
    % close gcf

end

disp('Done! Open each frame in the analysis folder to view bubble parameters')