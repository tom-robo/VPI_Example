# VPI_Example
An example to demonstrate differences in number of points tracked by either reusing the status array or creating and destroying the array every time using the LK optical flow function in the VPI library.

Follow along in the forum exchange: https://forums.developer.nvidia.com/t/vpi-unable-to-write-to-vpi-array-type-u8-on-jetson-device/242282

## Dependencies:
-the VPI v2.1 (or greater) library with header files (install nvidia-vpi2-dev on jetson with apt I think or vpi2-dev on x86-64). 
-Also required are opencv-core, opencv-imgproc and opencv-imgcodecs for image loading.

## To build: 

In repository root: 
 - mkdir build && cd build 
 - cmake .. 
 - make -j6
 ./VPI_Example 
 
 ## Explanation
 The example iterates through the 20 test images seen in the repository root. For each image a gaussian image pyramid is generated and keypoints are computed using vpiSubmitHarrisCornerDetector and stored in the prevFeatures buffer. 
 
The vpiSubmitOpticalFlowPyrLK function is used to track the generated keypoints from its corresponding image pyramid (pyrPrevFrame) to incoming frame's pyramid (pyrCurFrame). 

Once the function is complete, the number of tracked points are recorded by reading from the featStatus buffer (0 if tracked, 1 if not) and stored in a std::vector in setOutputPoints(). 

In this same function, if the featStatus buffer is being reused, all positions read from are written to with "0" to indicate that all points should be attempted to be tracked for the next frame. This is considered "resusing the buffer" and executed in the reuseStatusArray() function. 

If the featStatus buffer is not reused (i.e. in the createDestroyStatusArray() function) then it is created before vpiSubmitOpticalFlowPyrLK and subsequently destroyed after all processing is complete for that image frame. 

The number of keypoints computed and tracked to the next frame are printed for each image. 

## Results
If the featStatus buffer is zeroed correctly when trying to reuse, the number of keypoints found and points tracked should be identical for the createDestroyStatusArray() function and reuseStatusArray(). 

This is the case when testing this code using VPI 2.2.4 on an x86-64 laptop running inside a nvcr.io/nvidia/deepstream:6.1-devel image (results found in x86 .txt file above) as well as natively. This is running ubuntu 18.04 natively and 20.04 inside the container. 

Sadly this is not the case (fewer points tracked when attempting to resuse the buffer) on a jetson NX running 35.1 either natively or in an l4t-jetpack:35.1 container using VPI 2.1.6. I have attached both print outs of the results in this folder. I have also anecdotally seen this behaviour on my other NX board running jetpack 5.1 running the same 35.1 container. 
 
Overall, it is pleasing to see that the jetson createDestroy results mirror exactly compared to the x86 results. 
 
## Not zeroing featStatus buffer
 
I have also added in the results file number of points tracked if the featStatus array is not zeroed after use to try and simulate the behaviour on the jetson devices. You can see I demonstrate similar behaviour on x86-64, however the number of points tracked does not exactly match the jetson results.

## Any problems, do not hesistate to make an issue or contact me in the forum thread


