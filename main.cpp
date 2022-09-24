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

void process_thread(Camera cam, int idx) {
    apriltag_family_t *tf = tag36h11_create();
    apriltag_detector_t *td = apriltag_detector_create();
    apriltag_detector_add_family(td, tf);
    td->quad_decimate = 2;
    td->nthreads = 6;

    int fontface = FONT_HERSHEY_SCRIPT_SIMPLEX;
    double fontscale = 1.0;

    Mat img, gray;
    auto prev = chrono::high_resolution_clock::now();

    while(1) {
        //Pop frame from queue and check if the frame is valid
        if (cam.frame_queue[idx]->try_pop(img)) {
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
            }
            apriltag_detections_destroy(detections);

            auto now = chrono::high_resolution_clock::now();
            auto elapsed = now - prev;
            prev = now;
            int f_time = chrono::duration_cast<chrono::microseconds>(elapsed).count();

            cv::putText(img, to_string(1000000/f_time), Point(40, 40),
                fontface, fontscale, Scalar(0xff, 0x99, 0), 2);

            cv::imshow("Disp " + to_string(idx), img);
            if (waitKey(1) >= 0)
                break;
        }
    }

}

int main() {
   
    vector<int> capture_index = { 0, 2 };
    Camera cam(capture_index);
    cam.init_and_start();

    thread* t0 = new thread(&process_thread, cam, 0);
    thread* t1 = new thread(&process_thread, cam, 1);

    while(1) {
        sleep(1);
    }

    return 0;
}
