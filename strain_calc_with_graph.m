% Read the video file
videoReader = VideoReader('reactorvid.mov');

% Create a figure with two subplots
fig = figure('Position', [100, 100, 1200, 500]);
videoAxes = subplot(1, 2, 1);
plotAxes = subplot(1, 2, 2);

% Initialize variables
initialLength = 100; % Set this to your initial gauge length (in mm)
frameInterval = round(videoReader.FrameRate); % Update strain every second
frameCount = 0;
currentStrain = 0;

% Set up the animated line
hold(plotAxes, 'on');
strainLine = animatedline(plotAxes, 'Color', 'b', 'LineWidth', 2);
xlabel(plotAxes, 'Time (s)');
ylabel(plotAxes, 'Strain Magnitude');
title(plotAxes, 'Absolute Strain vs Time');
grid(plotAxes, 'on');
yline(plotAxes, 0, 'k--', 'LineWidth', 1); % Add only once

% Initialize min and max strain for dynamic y-axis scaling
minStrain = 0;
maxStrain = 0;

% Display first frame for initialization
if hasFrame(videoReader)
    frame = readFrame(videoReader);
    imHandle = imshow(frame, 'Parent', videoAxes);
    title(videoAxes, 'Video');
    frameCount = 1;
end

% Reset video to start
videoReader.CurrentTime = 0;

% Process the video
while hasFrame(videoReader)
    frame = readFrame(videoReader);
    frameCount = frameCount + 1;
    currentTime = frameCount / videoReader.FrameRate;
    
    if mod(frameCount, frameInterval) == 0
        % Calculate current length
        currentLength = calculateCurrentLength(frame);
        
        % Calculate strain
        currentStrain = abs((currentLength - initialLength) / initialLength);
        
        % Update min and max strain
        minStrain = min(minStrain, currentStrain);
        maxStrain = max(maxStrain, currentStrain);
        
        % Add point to the animated line
        addpoints(strainLine, currentTime, currentStrain);

        % Update plot axes limits (ensure a reasonable range)
        xlim(plotAxes, [0, currentTime]);
        yPad = 0.01; % Minimum padding for y-axis
        yMin = min(-yPad, minStrain*1.1);
        yMax = max(yPad, maxStrain*1.1);
        ylim(plotAxes, [yMin, yMax]);
        drawnow limitrate;
    end
    
    % Display strain on the frame
    strainText = sprintf('Strain: %.4f', currentStrain);
    frame = insertText(frame, [10 10], strainText, 'FontSize', 18, 'BoxColor', 'white', 'BoxOpacity', 0.4, 'TextColor', 'black');
    
    % Update the frame in the video axes
    if exist('imHandle', 'var') && ishandle(imHandle)
        set(imHandle, 'CData', frame);
    else
        imHandle = imshow(frame, 'Parent', videoAxes);
    end
    title(videoAxes, 'Video');
    
    % Force MATLAB to draw the updated figure
    drawnow limitrate
    
    % Pause to control playback speed (adjust as needed)
    pause(1/videoReader.FrameRate);
end

function length = calculateCurrentLength(frame)
    % Convert frame to grayscale and normalize
    grayFrame = im2double(rgb2gray(frame));
    
    % Apply local adaptive histogram equalization
    enhancedFrame = adapthisteq(grayFrame);
    
    % Estimate background
    background = imgaussfilt(enhancedFrame, 10);
    
    % Create foreground by subtracting background
    foreground = enhancedFrame - background;
    
    % Binarize the image
    threshval = 0.2; % Adjust this threshold as needed
    threshimg = foreground > threshval;
    
    % Detect connected components (bubbles)
    islands = bwconncomp(threshimg);
    
    % Calculate bubble properties
    props = regionprops(threshimg, 'MajorAxisLength');
    
    % Calculate mean diameter in mm (assuming 1 pixel = 0.2645833333 mm)
    if ~isempty(props)
        meanDiameter = mean([props.MajorAxisLength]) * 0.2645833333;
    else
        meanDiameter = 0;
    end
    
    % Return the mean diameter as the current length
    length = meanDiameter;
end







