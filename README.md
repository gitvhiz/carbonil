# CarboNil

This GitHub repository contains the code used for monitoring bubble dynamics in a photobioreactor. This is part of a collaborative, interdisciplinary project at the Birla Institute of Technology and Science, Pilani. This code repository is a supplement to the patent "System and Method for Analysing Bubble Dynamics in a Photobioreactor"; Rupesh Kumar, Vehaan Handa, S.V Sumanth, Nishanth Lanka, Anirban Roy, Snehanshu Saha, Santonu Sarkar; Indian Patent App. No. 202411088778 [Filed Nov. 16, 2024]

A video file has to be supplied in the initial section of the code titled "BUBBLE_DETECTION_MASTER.m", along with a target directory. This code performs the task of reading the video file, performs frame splitting and stores the resulting frames in the target directory. Then, morphological operations and image processing techniques are applied to an arbitrary number of frames (as decided by the user) and the binarized results are stored as separate images within the subdirectory titled "Analysis". The results provide information about the number of bubbles, their mean diameter and their mean eccentricity.

The above code also includes a function wherein bubbles are tracked frame by frame, and a bounding box is drawn around each identified bubble, with the resulting frames stored in the "Tracking" subdirectory. The Hungarian algorithm is used for this purpose.

A separate code file "strain_calc_with_graph.m" deals with calculating the average strain of bubbles in each frame. A strain-time graph is plotted to showcase this variation as well.

In certain videos where the photobioreactor takes up a relatively small amount of screen real estate as compared to the background, a commented-out section which deals with contiguity based filtering has also been included, wherein additional processing is performed on the binarized image.

Sample video files (slowed down to 960 fps) have been provided. These video samples are what we validated the code on.
