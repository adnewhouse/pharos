#include <opencv2/opencv.hpp>
#include <thread>

extern "C" {
#include "apriltag/apriltag.h"
#include "apriltag/tag36h11.h"
#include "apriltag/apriltag_pose.h"
}

#include "camera.hpp"

using namespace cv;
using namespace std;

void process_thread(Camera& cam, int idx, concurrent_queue<CameraPose*>& pose_queue) {
    apriltag_family_t *tf = tag36h11_create();
    apriltag_detector_t *td = apriltag_detector_create();
    apriltag_detector_add_family(td, tf);
    td->quad_decimate = 2;
    td->nthreads = 6;

    int fontface = FONT_HERSHEY_SCRIPT_SIMPLEX;
    double fontscale = 1.0;

    CameraFrame cf;
    Mat img, gray;
    auto prev = chrono::high_resolution_clock::now();

    while(1) {
        //Pop frame from queue and check if the frame is valid
        if (cam.frame_queue[idx]->try_pop(cf)) {
            img = cf.frame;
            cvtColor(img, gray, COLOR_BGR2GRAY);

            image_u8_t im = { .width = gray.cols,
                .height = gray.rows,
                .stride = gray.cols,
                .buf = gray.data
            };

            zarray_t *detections = apriltag_detector_detect(td, &im);
            for (int i = 0; i < zarray_size(detections); i++) {
                apriltag_detection_t *det;
                zarray_get(detections, i, &det);

                stringstream ss;
                ss << det->id;
                String text = ss.str();
                int baseline;
                Size textsize = getTextSize(text, fontface, fontscale, 2,
                                                &baseline);
                cv::putText(img, text, Point(det->c[0]-textsize.width/2,
                                        det->c[1]+textsize.height/2),
                        fontface, fontscale, Scalar(0xff, 0x99, 0), 2);

                apriltag_detection_info_t info;
                info.det = det;
                info.tagsize = 0.2;
                info.fx = 1075;
                info.fy = 1075;
                info.cx = 631;
                info.cy = 430;

                apriltag_pose_t pose;
                double err = estimate_tag_pose(&info, &pose);
                pose_queue.push(new CameraPose(idx, det->id, err, pose, cf.frame_time));
            }
            apriltag_detections_destroy(detections);

            auto now = chrono::high_resolution_clock::now();
            auto elapsed = now - prev;
            prev = now;
            int f_time = chrono::duration_cast<chrono::microseconds>(elapsed).count();

            cv::putText(img, to_string(1000000/f_time), Point(40, 40),
                fontface, fontscale, Scalar(0xff, 0x99, 0), 2);

            cv::imshow("Disp " + to_string(idx), img);
            pollKey();
        }
    }

}

int main() {
   
    vector<int> cam_ids = { 0, 2 };
    Camera cam(cam_ids);
    cam.init_and_start();

    vector<thread*> worker_threads;
    concurrent_queue<CameraPose*> pose_queue;
    thread* t;
    for (int i = 0; i < cam_ids.size(); i++) {
        t = new thread(&process_thread, ref(cam), i, ref(pose_queue));
        worker_threads.push_back(t);
    }

    while(1) {
        CameraPose* cf;
        if (pose_queue.try_pop(cf)) {
            auto now = chrono::high_resolution_clock::now();
            auto pose_latency = chrono::duration_cast<chrono::milliseconds>(now - cf->frame_time);
            cout << "Tag id: " << cf->tag_id << ", \tTag latency (ms):  " << pose_latency.count() << "\t Tag error: " << cf->err << endl;
        }
    }


    return 0;
}
