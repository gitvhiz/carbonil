% Step 1: Load and display the image
rgb = imread('greenimage.jpeg');
figure;
subplot(2,2,1);
imshow(rgb);
title('Original Image');

% Step 2: Color segmentation to isolate green objects
hsv_img = rgb2hsv(rgb);

% Define green color range in HSV
hue_min = 0.2; hue_max = 0.4;  % Green hue range
sat_min = 0.3;                 % Minimum saturation
val_min = 0.3;                 % Minimum brightness

% Create mask for green objects
green_mask = (hsv_img(:,:,1) >= hue_min & hsv_img(:,:,1) <= hue_max) & ...
             (hsv_img(:,:,2) >= sat_min) & ...
             (hsv_img(:,:,3) >= val_min);

% Morphological cleanup
se = strel('disk', 2);
green_mask = imopen(green_mask, se);
green_mask = imclose(green_mask, se);
green_mask = imfill(green_mask, 'holes');
green_mask = bwareaopen(green_mask, 50);

subplot(2,2,2);
imshow(green_mask);
title('Green Object Mask');

% Step 3: Convert to grayscale for circle detection
gray = rgb2gray(rgb);
masked_gray = gray;
masked_gray(~green_mask) = 255; % Set non-green areas to white

subplot(2,2,3);
imshow(masked_gray);
title('Masked Grayscale');

% Step 4: Circle detection parameters
min_radius = 5;
max_radius = 25;

% Step 5: Detect circles
[centers, radii, metric] = imfindcircles(masked_gray, [min_radius max_radius], ...
    'ObjectPolarity', 'dark', ...
    'Sensitivity', 0.85, ...
    'EdgeThreshold', 0.05);

% Step 6: Filter circles by green mask
valid_circles = false(length(centers), 1);
for i = 1:length(centers)
    x = round(centers(i,1));
    y = round(centers(i,2));
    if x >= 1 && x <= size(green_mask,2) && y >= 1 && y <= size(green_mask,1)
        if green_mask(y, x)
            valid_circles(i) = true;
        end
    end
end

centers_filtered = centers(valid_circles, :);
radii_filtered = radii(valid_circles);
metric_filtered = metric(valid_circles);

% Step 7: Filter circles inside circular field of view
center_img = [size(rgb,2)/2, size(rgb,1)/2];
radius_img = min(size(rgb,1), size(rgb,2))/2 * 0.98;
dist_from_center = sqrt((centers_filtered(:,1)-center_img(1)).^2 + ...
                        (centers_filtered(:,2)-center_img(2)).^2);

valid_idx = dist_from_center + radii_filtered < radius_img;
centers_filtered = centers_filtered(valid_idx,:);
radii_filtered = radii_filtered(valid_idx);
metric_filtered = metric_filtered(valid_idx);

% Step 8: Display results
subplot(2,2,4);
imshow(rgb);
hold on;
viscircles(centers_filtered, radii_filtered, 'EdgeColor', 'r', 'LineWidth', 2);
plot(centers_filtered(:,1), centers_filtered(:,2), 'r+', 'MarkerSize', 5, 'LineWidth', 2);
title(sprintf('Detected Green Circles: %d', length(centers_filtered)));
hold off;

% Step 9: Output statistics
fprintf('Detection Results:\n');
fprintf('Total green circles detected: %d\n', length(centers_filtered));
if ~isempty(radii_filtered)
    fprintf('Average radius: %.2f pixels\n', mean(radii_filtered));
    fprintf('Radius range: %.2f - %.2f pixels\n', min(radii_filtered), max(radii_filtered));
else
    fprintf('No circles detected.\n');
end