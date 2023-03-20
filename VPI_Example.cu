#include <iostream> 
#include <vector>
#include <sstream>
#include <iomanip>

#include <cuda_runtime.h> 

#include <vpi/VPI.h>
#include <vpi/algo/OpticalFlowPyrLK.h>
#include <vpi/algo/GaussianPyramid.h>
#include <vpi/algo/HarrisCorners.h>
#include <vpi/OpenCVInterop.hpp>
#include <vpi/algo/HarrisCorners.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/Image.h>

#include <opencv2/core/mat.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp> 

inline void checkVPIError(VPIStatus stmt, const char *file, int line)
{
    VPIStatus status__ = (stmt);                            
    if (status__ != VPI_SUCCESS)                            
    {                                                       
        char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH];         
        vpiGetLastStatusMessage(buffer, sizeof(buffer));    
        std::ostringstream ss;                              
        ss << vpiStatusGetName(status__) << ": " << buffer 
                << " file: " << file << ":" << line << "\n"; 
        throw std::runtime_error(ss.str());                 
    }                                                       
}

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        std::stringstream err; 
        err << "ERROR: GPUassert: " << cudaGetErrorString(code) 
                << " " << file << ":" << line << std::endl;
        throw std::runtime_error(err.str());
    }
}

#define gpuErrchk(ans) { \
            gpuAssert((ans), __FILE__, __LINE__); \
        }

#define CHECK_VPI(STMT) do  {  \
        STMT; \
        checkVPIError(STMT, __FILE__, __LINE__); \
    } while (0);


class tracker
{
public: 

    tracker(int w, int h, float s, int l, int max_corners) : 
        width(w), 
        height(h), 
        scale(s), 
        levels(l), 
        maxCorners(max_corners)
    {
        currMat = cv::Mat(height, width,  CV_8UC3); 
        grayMat = cv::Mat(height, width,  CV_8UC1); 
        equalHist = cv::Mat(height, width,  CV_8UC1); 
        harrisMat = cv::Mat(height, width, CV_8UC3); 

        CHECK_VPI(vpiImageCreate(width, height, format, 0,&prevImage)); 
        CHECK_VPI(vpiImageCreate(width, height, format, 0,&inputImage));
        CHECK_VPI(vpiImageCreateWrapperOpenCVMat(currMat, 0, &currImage));

        CHECK_VPI(vpiImageCreateWrapperOpenCVMat(harrisMat, 0, &wrappedHarris));
        CHECK_VPI(vpiImageCreate(width, height, VPI_IMAGE_FORMAT_S16, 0, &inputHarris));

        CHECK_VPI(vpiPyramidCreate(width, height, format, levels, scale, 0, &pyrPrevFrame));
        CHECK_VPI(vpiPyramidCreate(width, height, format, levels, scale, 0, &pyrCurFrame));

        CHECK_VPI(vpiArrayCreate(maxCorners, VPI_ARRAY_TYPE_KEYPOINT_F32, 0, &prevFeatures));
        CHECK_VPI(vpiArrayCreate(maxCorners, VPI_ARRAY_TYPE_KEYPOINT_F32, 0, &currFeatures));
        CHECK_VPI(vpiArrayCreate(maxCorners, VPI_ARRAY_TYPE_U32, 0, &scores));
    
        CHECK_VPI(vpiCreateOpticalFlowPyrLK(backend,
                                        width, height,
                                        format,
                                        levels,scale, 
                                        &optflow));

        CHECK_VPI(vpiInitOpticalFlowPyrLKParams(&lkParams));

        CHECK_VPI(vpiCreateHarrisCornerDetector(backend, width, height, &harris));
        CHECK_VPI(vpiInitHarrisCornerDetectorParams(&harrisParams));

        CHECK_VPI(vpiStreamCreate(0, &stream));

    }

    ~tracker()
    {

    }

    void createStatusArray()
    {
        // CHECK_VPI(vpiArrayCreate(maxCorners, VPI_ARRAY_TYPE_U8, 0, &featStatus));
        
        featStatusData.bufferType = VPI_ARRAY_BUFFER_CUDA_AOS;
        featStatusData.buffer.aos.sizePointer = &numStatusPoints; 
        featStatusData.buffer.aos.capacity = maxCorners; 
        featStatusData.buffer.aos.strideBytes = maxCorners; 

        gpuErrchk(cudaMalloc(&featStatusData.buffer.aos.data, maxCorners)); 

        featStatusData.buffer.aos.type = VPI_ARRAY_TYPE_U8;

        CHECK_VPI(vpiArrayCreateWrapper(&featStatusData, VPI_BACKEND_CUDA, &featStatus)); 
        
        std::cout << "Created array wrapper" << std::endl;;
    }


    void destroyStatusArray()
    {
        if (featStatus != NULL) 
        {
            vpiArrayDestroy(featStatus);
            featStatus = NULL; 
        }

        gpuErrchk(cudaFree(featStatusData.buffer.aos.data));
    }

    void makeCurrGaussPyramid()
    {
        // Adding image to input for LKTracker
        CHECK_VPI(vpiImageSetWrappedOpenCVMat(currImage, currMat));
        CHECK_VPI(vpiSubmitConvertImageFormat(stream, backend, currImage, inputImage, NULL));

        // Making pyrCurFrame gaussian pyramid; 
        CHECK_VPI(vpiSubmitGaussianPyramidGenerator(stream, backend, inputImage, pyrCurFrame, VPI_BORDER_CLAMP));
    }

// Convert colour image to grayscale and add to vpi
    void setInputImage(std::string filename)
    {
        cv::Mat image; 

        try
        {
            image = cv::imread(filename);

        }
        catch (const cv::Exception& e)
        {
            std::cerr << "ERROR: Cannot load image: " << filename << " Error: " << e.what() << std::endl; 
        }

//        std::cout << "INFO: Loaded image: " << filename << " Width: " << image.cols << " Height: " << image.rows << std::endl;

        try
        {
            cv::cvtColor(image, grayMat, cv::COLOR_RGB2GRAY);
        }
        catch (cv::Exception& e)
        {
            std::cerr << "ERROR: Cannot convert colour input image to grayscale. Error: " << e.what() << std::endl;
        }

        try
        {
            cv::equalizeHist(grayMat, equalHist);
            cv::cvtColor(equalHist, currMat, cv::COLOR_GRAY2RGB);
        }
        catch (cv::Exception& e)
        {
            std::cerr << "ERROR: Canot convert grayscale mat to RGB (3-channels). Errror: " << e.what() << std::endl; 
        }

        harrisMat = currMat.clone(); 
    }

    int trackPoints(bool zeroStatusBuffer)
    {
        numTrackedPoints=0; 

        makeCurrGaussPyramid(); 

        if (!first)
        {

            // debugCheckStatusHasBeenSetToZero(numFeatures2Track); 

            CHECK_VPI(vpiSubmitOpticalFlowPyrLK(stream, 0, optflow, pyrPrevFrame, pyrCurFrame, 
                                                    prevFeatures, currFeatures, featStatus, &lkParams));

            CHECK_VPI(vpiStreamSync(stream));
            setOutputPoints(zeroStatusBuffer);
        
        }
        else
        { 
            first = false; 
            return 0; 
        }

        computePointsToTrack();
        swapPyramids(); 

        return numTrackedPoints; 
    }

    int getNumTrackedPoints()
    {
        return numTrackedPoints; 
    }

    int getNumFeatures2Track()
    {
        return numFeatures2Track;
    }

private: 

    void debugCheckStatusHasBeenSetToZero(int num_points)
    {
        VPIArrayData currStatusBuff; 

        int zeroedElements = 0;
        int nonZeroedElements = 0;  

        CHECK_VPI(vpiArrayLockData(featStatus, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &currStatusBuff));

        unsigned char* currStatusArray = (unsigned char*)currStatusBuff.buffer.aos.data;

        for (int i = 0; i<num_points; i++)
        {
            unsigned char status = currStatusArray[i];

            if (status == 0)
            {
                zeroedElements++;
            }
            else
            {
                nonZeroedElements++;
            }
        }

        CHECK_VPI(vpiArrayUnlock(featStatus));

        if (zeroedElements == num_points)
        {
            std::cout << "SUCCESS: Point to be checked: " << num_points 
                        << "number of non-zero points: " << nonZeroedElements
                        << " number of zeroed elements: " << zeroedElements << std::endl; 

        }
        else
        {
            std::cout << "FAILURE: Point to be checked: " << num_points 
                        << "number of non-zero points: " << nonZeroedElements
                        << " number of zeroed elements: " << zeroedElements
                        << " Not all points have been zeroed! " << std::endl; 
        }
    }

    void setOutputPoints(bool zeroStatusBuffer)
    {
        
        VPIArrayData currFeaturesBuff; 
        VPIArrayData prevFeaturesBuff; 
        VPIArrayData currStatusBuff; 

        CHECK_VPI(vpiArrayLockData(currFeatures, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &currFeaturesBuff));
        CHECK_VPI(vpiArrayLockData(prevFeatures, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &prevFeaturesBuff));
        CHECK_VPI(vpiArrayLockData(featStatus, VPI_LOCK_READ_WRITE, VPI_ARRAY_BUFFER_HOST_AOS, &currStatusBuff));

        VPIKeypointF32* currKeypointsArray = (VPIKeypointF32 *)currFeaturesBuff.buffer.aos.data;
        VPIKeypointF32* prevKeypointsArray = (VPIKeypointF32 *)prevFeaturesBuff.buffer.aos.data;

        unsigned char* currStatusArray = (unsigned char*)currStatusBuff.buffer.aos.data;

        currTrackedPoints.clear(); 
        prevTrackedPoints.clear(); 

        for (int i = 0; i<numFeatures2Track; i++)
        {
            unsigned char status = currStatusArray[i];
            VPIKeypointF32 currPoint = currKeypointsArray[i]; 
            VPIKeypointF32 prevPoint = prevKeypointsArray[i]; 

            if (status == 0)
            {
                currTrackedPoints.push_back({prevPoint.x, prevPoint.y}); 
                prevTrackedPoints.push_back({prevPoint.x, prevPoint.y});

                numTrackedPoints++;
            }

            if (zeroStatusBuffer)
            {
                currStatusArray[i] = 0; 
            }
        }

/// Setting statusPtr to zero via cudaMemset; 
        gpuErrchk(cudaMemset(statusPtr, 0, maxCorners)); 

        CHECK_VPI(vpiArrayUnlock(currFeatures));
        CHECK_VPI(vpiArrayUnlock(prevFeatures));
        CHECK_VPI(vpiArrayUnlock(featStatus));
    }

    void swapPyramids()
    {
        std::swap(pyrPrevFrame, pyrCurFrame);
    }

    void computePointsToTrack()
    {
        // Adding image to input for Harris Corners (needs to be S16 format unlike LKTracker)
        CHECK_VPI(vpiImageSetWrappedOpenCVMat(wrappedHarris, harrisMat));
        CHECK_VPI(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, 
                                                    wrappedHarris, inputHarris, NULL));

        CHECK_VPI(vpiArraySetSize(prevFeatures, maxCorners)); 

        CHECK_VPI(vpiSubmitHarrisCornerDetector(stream, backend, harris, inputHarris,
                                                prevFeatures, scores, &harrisParams));
        CHECK_VPI(vpiStreamSync(stream));

        VPIArrayData prevPointsBuff; 
        CHECK_VPI(vpiArrayLockData(prevFeatures, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &prevPointsBuff));
        numFeatures2Track = *prevPointsBuff.buffer.aos.sizePointer;
        
        numStatusPoints = numFeatures2Track; 

        CHECK_VPI(vpiArrayUnlock(prevFeatures));
//        std::cout << " Number of detected Points: " << numFeatures2Track << std::endl;

    }


private: 
    int width; 
    int height; 
    float scale; 
    int levels; 
    int maxCorners; 

    bool first = true; 

    int32_t numFeatures2Track=0;
    int32_t numTrackedPoints=0;

    std::vector<cv::Point2f> prevTrackedPoints; 
    std::vector<cv::Point2f> currTrackedPoints; 

    VPIStream stream;

    cv::Mat currMat; 
    cv::Mat grayMat; 
    cv::Mat equalHist; 

    VPIImage inputHarris; 
    VPIImage wrappedHarris; 
    cv::Mat harrisMat; 

    VPIImage currImage; 
    VPIImage inputImage; 
    VPIImage prevImage;

    VPIBackend backend = VPI_BACKEND_CUDA; 
    VPIImageFormat format = VPI_IMAGE_FORMAT_U8;

    unsigned char* statusPtr; 
    int32_t numStatusPoints=0;

    VPIPyramid pyrPrevFrame=NULL;
    VPIPyramid pyrCurFrame=NULL; 

    VPIArray prevFeatures=NULL;
    VPIArray currFeatures=NULL;
    VPIArray scores=NULL;

    VPIArray featStatus=NULL;
    VPIArrayData featStatusData; 

    VPIOpticalFlowPyrLKParams lkParams;
    VPIPayload optflow=NULL;

    VPIPayload harris=NULL;
    VPIHarrisCornerDetectorParams harrisParams;

}; 

void reuseStatusArray(bool zeroStatusBuffer, int width, int height, float scale, int levels, int max_corners, int n)
{
    std::cout << "_____REUSING STATUS ARRAY. "
                << " ZEROING STATUS BUFFER: " << zeroStatusBuffer << "_____" << std::endl;

    tracker track(width, height, scale, levels, max_corners); 

    track.createStatusArray();

    for (int i=1; i<n; ++i)
    {
        std::stringstream filename; 
        filename << "./../dashcam-" << std::setw(3) << std::setfill('0') << i << ".jpg"; 

        std::cout << " Into tracking loop"; 

        track.setInputImage(filename.str());

        std::cout << " Set input image"; 

        int numPointsTracked = track.trackPoints(zeroStatusBuffer);

        std::cout << " Tracked points"; 

        std::cout << i << ": Point in prev array: " << track.getNumFeatures2Track() << " number of points tracked: " << track.getNumTrackedPoints() << std::endl; 
    }

    track.destroyStatusArray(); 
}

void createDestroyStatusArray(bool zeroStatusBuffer, int width, int height, float scale, int levels, int max_corners, int n)
{
    std::cout << "____CREATING/ DESTROYING STATUS ARRAY EVERY O.F. CYCLE. "
                << " ZEROING STATUS BUFFER: " << zeroStatusBuffer << "_____" << std::endl;

    tracker track(width, height, scale, levels, max_corners); 

    for (int i=1; i<n; i++)
    {
        std::stringstream filename; 
        filename << "./../dashcam-" << std::setw(3) << std::setfill('0') << i << ".jpg"; 

        track.setInputImage(filename.str()); 

        track.createStatusArray();

        int numPointsTracked = track.trackPoints(zeroStatusBuffer);
    
        track.destroyStatusArray(); 

        std::cout << i << ": Point in prev array: " << track.getNumFeatures2Track() << " number of points tracked: " << track.getNumTrackedPoints() << std::endl; 

    }
}

int main(int argc, char **argv) 
{
    int width = 1280; 
    int height = 720;
    float scale = 0.5; 
    int maxCorners = 8192;
    int levels = 4; 

    int n = 20; 

    reuseStatusArray(true, width, height, scale, levels, maxCorners, n); 
    createDestroyStatusArray(true, width, height, scale, levels, maxCorners, n);

    reuseStatusArray(false, width, height, scale, levels, maxCorners, n);
    createDestroyStatusArray(false, width, height, scale, levels, maxCorners, n);
 
}
