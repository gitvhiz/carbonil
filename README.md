# CarboNil

This GitHub repository contains digital image correlation (DIC) code written in MATLAB, and is primarily utilised for monitoring bubble dynamics in a photobioreactor. This is part of a collaborative, interdisciplinary project conducted at the Birla Institute of Technology and Science, Pilani. The primary end goal of this project is to regulate energy expenditure in the aeration assembly, and more importantly, to be able to automate this process by adjusting gas holdup and mass transport parameters through an entirely data-driven procedure.

A video file has to be supplied in the initial section of the code titled `BUBBLE_DETECTION_MASTER.m`, along with a target directory. This code performs the task of reading the video file, performs frame splitting and stores the resulting frames in the target directory. Then, morphological operations and image processing techniques are applied to an arbitrary number of frames (as decided by the user) and the binarized results are stored as separate images within the subdirectory titled `Analysis`. The results provide information about the number of bubbles, their mean diameter and their mean eccentricity.

The above code also includes a function wherein bubbles are tracked frame by frame, and a bounding box is drawn around each identified bubble, with the resulting frames stored in the `Tracking` subdirectory. The Hungarian algorithm (defined in `munkres.m`) is used for this purpose.

A separate code file `strain_calc_with_graph.m` deals with calculating the average strain of bubbles in each frame. A strain-time graph is plotted to showcase this variation as well.

In certain videos where the photobioreactor takes up a relatively small amount of screen real estate as compared to the background, a commented-out section which deals with contiguity based filtering has also been included, wherein additional processing is performed on the binarized image.

Sample video files (slowed down to 960 fps) have been provided. These video samples are what we validated the code on.

To track bubbles under green light, we experimented with identifying cell boundaries in a microscopic image of green algae first. `greencircle.m` defines a simplified logic employing masks and contiguity-based filters to detect over 95% of all cell boundaries, including, to a large extent, algae at the image boundary.
