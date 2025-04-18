%% ===========================================================
%                VIDEO PROCESSING AND FRAME SPLITTING
% ===========================================================
close all; clear; clc;

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

for frameIdx = 1:numFrames
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

% --------- PARAMETERS ---------
inputVideoFile = 'purple_video.mp4';
outputFolder = 'purple_video_frames';
frameMatFile = 'purple_video_frames.mat';
frameRate = 960;         % frames per second
pixelsPerMm = 10;        % adjust as needed for your scale
maxInvisibleFrames = 30; % frames a track can be invisible before deletion
minVisibleCount = 5;     % minimum visible frames before a track is valid
distanceThreshold = 50;  % max distance (pixels) for assignment

%% --------- READ VIDEO AND SAVE FRAMES ---------
% vidReader = VideoReader(inputVideoFile);
% if ~exist(outputFolder, 'dir'), mkdir(outputFolder); end
% 
% frameIndex = 0;
% Im = {};
% while hasFrame(vidReader)
%     vidFrame = readFrame(vidReader);
%     frameIndex = frameIndex + 1;
%     frameFileName = fullfile(outputFolder, sprintf('frame_%04d.png', frameIndex));
%     imwrite(vidFrame, frameFileName);
%     Im{frameIndex} = im2double(rgb2gray(vidFrame));
% end
% numFrames = frameIndex;
% save(frameMatFile, 'Im', '-v7.3');

%% =======================================================
%              BUBBLE TRACKING ALGORITHM
% =======================================================
% Load frames if not already in workspace
if ~exist('Im','var')
    load(frameMatFile, 'Im');
    numFrames = numel(Im);
end

% Initialize tracking structures
tracks = struct('id', {}, 'kalmanFilter', {}, 'age', {}, 'totalVisibleCount', {}, ...
    'consecutiveInvisibleCount', {}, 'centroids', {}, 'trajectory', {}, 'velocity', {});
nextId = 1;

trackingFolder = fullfile(outputFolder, 'tracking');
if ~exist(trackingFolder, 'dir'), mkdir(trackingFolder); end

for frameIdx = 1:numFrames
    frame = Im{frameIdx};
    [centroids, bboxes, mask] = detectBubbles(frame);

    % Predict new locations of tracks
    for i = 1:length(tracks)
        predictedCentroid = predict(tracks(i).kalmanFilter);
        tracks(i).centroids = [tracks(i).centroids; predictedCentroid];
    end

    % Assign detections to tracks using Hungarian algorithm
    nTracks = length(tracks);
    nDetections = size(centroids,1);
    cost = zeros(nTracks, nDetections);
    for i = 1:nTracks
        for j = 1:nDetections
            cost(i,j) = norm(tracks(i).centroids(end,:) - centroids(j,:));
        end
    end
    cost(cost > distanceThreshold) = Inf; % ignore distant assignments
    [assignments, unassignedTracks, unassignedDetections] = assignDetectionsToTracks(cost);

% Update assigned tracks
for k = 1:size(assignments,1)
    trackIdx = assignments(k,1);
    detectionIdx = assignments(k,2);
    correct(tracks(trackIdx).kalmanFilter, centroids(detectionIdx,:));
    tracks(trackIdx).centroids = [tracks(trackIdx).centroids; centroids(detectionIdx,:)];
    tracks(trackIdx).trajectory = [tracks(trackIdx).trajectory; centroids(detectionIdx,:)];
    tracks(trackIdx).totalVisibleCount = tracks(trackIdx).totalVisibleCount + 1;
    tracks(trackIdx).consecutiveInvisibleCount = 0;
    % --- Velocity calculation in mm/s ---
    if size(tracks(trackIdx).centroids,1) > 1
        dt = 1/frameRate; % seconds between frames
        % Displacement in pixels between last two centroids
        dpix = norm(tracks(trackIdx).centroids(end,:) - tracks(trackIdx).centroids(end-1,:));
        % Convert pixels to mm
        dmm = dpix / pixelsPerMm;
        % Velocity in mm/s
        v_mmps = dmm / dt;
        tracks(trackIdx).velocity = v_mmps;
    else
        tracks(trackIdx).velocity = 0;
    end
end


    % Update unassigned tracks
    for i = 1:length(unassignedTracks)
        ind = unassignedTracks(i);
        tracks(ind).consecutiveInvisibleCount = tracks(ind).consecutiveInvisibleCount + 1;
        tracks(ind).centroids = [tracks(ind).centroids; tracks(ind).centroids(end,:)];
        tracks(ind).trajectory = [tracks(ind).trajectory; tracks(ind).centroids(end,:)];
    end

    % Delete lost tracks
    lost = [];
    for i = 1:length(tracks)
        if tracks(i).consecutiveInvisibleCount >= maxInvisibleFrames
            lost = [lost i];
        end
    end
    tracks(lost) = [];

    % Create new tracks for unassigned detections
    for i = 1:length(unassignedDetections)
        centroid = centroids(unassignedDetections(i),:);
        kf = configureKalmanFilter('ConstantVelocity', centroid, [1 1], [1 1], 1);
        newTrack = struct(...
            'id', nextId, ...
            'kalmanFilter', kf, ...
            'age', 1, ...
            'totalVisibleCount', 1, ...
            'consecutiveInvisibleCount', 0, ...
            'centroids', centroid, ...
            'trajectory', centroid, ...
            'velocity', 0);
        tracks(end+1) = newTrack;
        nextId = nextId + 1;
    end

    % Display and save results
    % Display and save results
    displayTrackingResults(frame, tracks, bboxes, mask, frameIdx, minVisibleCount);
    saveas(gcf, fullfile(trackingFolder, sprintf('frame_%04d_tracking.png', frameIdx)));
    close gcf
    fprintf('Tracking: Processed frame %d/%d\n', frameIdx, numFrames);
end

%% ========================= FUNCTIONS =========================

function [centroids, bboxes, mask] = detectBubbles(frame)
    % Enhance contrast and remove background
    frame = adapthisteq(frame);
    background = imgaussfilt(frame, 10);
    foreground = frame - background;
    % Threshold and clean up
    thresh = graythresh(foreground);
    mask = imbinarize(foreground, thresh*0.8);
    mask = bwareaopen(mask, 10);
    mask = imclose(mask, strel('disk',3));
    mask = imfill(mask, 'holes');
    % Get properties
    stats = regionprops(mask, 'Centroid', 'BoundingBox');
    if isempty(stats)
        centroids = [];
        bboxes = [];
    else
        centroids = cat(1, stats.Centroid);
        bboxes = cat(1, stats.BoundingBox);
    end
end

function [assignments, unassignedTracks, unassignedDetections] = assignDetectionsToTracks(cost)
    [assignments, unassignedTracks, unassignedDetections] = deal([]);
    if isempty(cost)
        unassignedTracks = 1:size(cost,1);
        unassignedDetections = 1:size(cost,2);
        return;
    end
    [assignment,~] = munkres(cost);
    for i = 1:length(assignment)
        if assignment(i) > 0 && cost(i,assignment(i)) < Inf
            assignments = [assignments; i, assignment(i)];
        else
            unassignedTracks = [unassignedTracks; i];
        end
    end
    for j = 1:size(cost,2)
        if ~ismember(j, assignment(assignment>0))
            unassignedDetections = [unassignedDetections; j];
        end
    end
end

% Download munkres.m from:
% https://www.mathworks.com/matlabcentral/fileexchange/20652-munkres-assignment-algorithm

function displayTrackingResults(frame, tracks, bboxes, mask, frameIdx, minVisibleCount)
    figure('Visible','off','Position',[100 100 1280 720]);
    subplot(1,2,1);
    imshow(frame,[]);
    hold on;
    for i = 1:length(tracks)
        if tracks(i).totalVisibleCount < minVisibleCount, continue; end
        plot(tracks(i).trajectory(:,1), tracks(i).trajectory(:,2), 'g-', 'LineWidth', 2);
        plot(tracks(i).centroids(end,1), tracks(i).centroids(end,2), 'ro', 'MarkerSize', 8, 'LineWidth',2);
        text(tracks(i).centroids(end,1)+10, tracks(i).centroids(end,2), ...
            sprintf('ID:%d V:%.1f mm/s', tracks(i).id, tracks(i).velocity), ...
            'Color','yellow','FontSize',10,'FontWeight','bold');
    end
    for k = 1:size(bboxes,1)
        rectangle('Position',bboxes(k,:),'EdgeColor','cyan','LineWidth',1);
    end
    title(sprintf('Tracking Results - Frame %d', frameIdx));
    hold off;

    subplot(1,2,2);
    imshow(mask,[]);
    title('Bubble Mask');
end

