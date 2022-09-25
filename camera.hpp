#pragma once

#include <thread>
#include <vector>

#include <tbb/concurrent_queue.h>
#include <opencv2/opencv.hpp>

extern "C" {
#include "apriltag/apriltag.h"
#include "apriltag/tag36h11.h"
#include "apriltag/apriltag_pose.h"
}

using namespace std;
using namespace cv;
using namespace tbb;

class CameraPose {
    public:
        int cam_idx;
        int tag_id;
        double err;
        apriltag_pose_t pose;
        chrono::time_point<chrono::high_resolution_clock> frame_time;
        CameraPose() {
        }
        CameraPose(int cam_idx, int tag_id, double err, apriltag_pose_t pose, chrono::time_point<chrono::high_resolution_clock> frame_time)
            : cam_idx{cam_idx}, tag_id{tag_id}, err{err}, pose{pose}, frame_time{frame_time} {
            
        }
};

class CameraFrame {
    public:
        int cam_idx;
        Mat frame;
        chrono::time_point<chrono::high_resolution_clock> frame_time;
};

class Camera {
    public:
        vector<concurrent_queue<CameraFrame>*> frame_queue;

        Camera(vector<int> cam_ids);
        void init_and_start();

    private:
        vector<VideoCapture*> cam_caps;
        vector<int> camera_indexes;
        vector<thread*> camera_threads;

        void cam_thread(int idx);

};
