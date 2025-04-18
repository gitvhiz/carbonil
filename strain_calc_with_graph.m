% Read the video file
videoReader = VideoReader('reactorvid.mov');

% Create a figure with two subplots
fig = figure('Position', [100, 100, 1200, 500]);
videoAxes = subplot(1, 2, 1);
plotAxes = subplot(1, 2, 2);

% Initialize variables
initialLength = 100; % Set this to your initial gauge length
frameInterval = round(videoReader.FrameRate); % Update strain every second
frameCount = 0;
currentStrain = 0;

% Set up the animated line
hold(plotAxes, 'on');
strainLine = animatedline(plotAxes, 'Color', 'b', 'LineWidth', 2);
xlabel(plotAxes, 'Time (s)');
ylabel(plotAxes, 'Strain');
title(plotAxes, 'Strain vs Time');
grid(plotAxes, 'on');

% Initialize min and max strain for dynamic y-axis scaling
minStrain = 0;
maxStrain = 0;

% Process the video
while hasFrame(videoReader)
    frame = readFrame(videoReader);
    frameCount = frameCount + 1;
    currentTime = frameCount / videoReader.FrameRate;
    
    if mod(frameCount, frameInterval) == 0
        % Calculate current length
        currentLength = calculateCurrentLength(frame);
        
        % Calculate strain
        currentStrain = (currentLength - initialLength) / initialLength;
        
        % Update min and max strain
        minStrain = min(minStrain, currentStrain);
        maxStrain = max(maxStrain, currentStrain);
        
        % Add point to the animated line
        addpoints(strainLine, currentTime, currentStrain);
        

        % Update plot axes limits
        xlim(plotAxes, [0, currentTime]);
        yRange = max(abs(minStrain), abs(maxStrain));
        ylim(plotAxes, [min(minStrain*1.1, -0.001), max(maxStrain*1.1, 0.001)]); % Ensure proper y-axis limits for both positive and negative strains
        drawnow;
        
        % Ensure the y-axis includes zero
        yline(plotAxes, 0, 'k--', 'LineWidth', 1);
    end
    
    % Display strain on the frame
    strainText = sprintf('Strain: %.4f', currentStrain);
    frame = insertText(frame, [10 10], strainText, 'FontSize', 18, 'BoxColor', 'white', 'BoxOpacity', 0.4, 'TextColor', 'black');
    
    % Display the frame
    imshow(frame, 'Parent', videoAxes);
    title(videoAxes, 'Video');
    
    % Force MATLAB to draw the updated figure
    drawnow
    
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



