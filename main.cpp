#include <opencv2/opencv.hpp>

extern "C" {
#include "apriltag/apriltag.h"
#include "apriltag/tag36h11.h"
#include "apriltag/apriltag_pose.h"
}

using namespace cv;
using namespace std;

int main() {
    apriltag_family_t *tf = tag36h11_create();
    apriltag_detector_t *td = apriltag_detector_create();
    apriltag_detector_add_family(td, tf);
    td->quad_decimate = 1;

    Mat img, gray;
    int fontface = FONT_HERSHEY_SCRIPT_SIMPLEX;
    double fontscale = 1.0;
    namedWindow("Disp");
    
    VideoCapture cap(0);
    cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(CAP_PROP_FRAME_HEIGHT, 720);
    cap.set(CAP_PROP_FPS, 60);

    if (!cap.isOpened()) {
	    cerr << "Couldn't open video capture device" << endl;
	    return -1;
    }

    while(1) {
        auto start = chrono::high_resolution_clock::now();
        cap >> img;
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
            line(img, Point(det->p[0][0], det->p[0][1]),
                     Point(det->p[1][0], det->p[1][1]),
                     Scalar(0, 0xff, 0), 2);
            line(img, Point(det->p[0][0], det->p[0][1]),
                     Point(det->p[3][0], det->p[3][1]),
                     Scalar(0, 0, 0xff), 2);
            line(img, Point(det->p[1][0], det->p[1][1]),
                     Point(det->p[2][0], det->p[2][1]),
                     Scalar(0xff, 0, 0), 2);
            line(img, Point(det->p[2][0], det->p[2][1]),
                     Point(det->p[3][0], det->p[3][1]),
                     Scalar(0xff, 0, 0), 2);

            stringstream ss;
            ss << det->id;
            String text = ss.str();
            int baseline;
            Size textsize = getTextSize(text, fontface, fontscale, 2,
                                            &baseline);
            putText(img, text, Point(det->c[0]-textsize.width/2,
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
            char* fmt = "%f  ";
            matd_print_transpose(pose.t, fmt);
            printf("\n");
        }
        apriltag_detections_destroy(detections);

        auto elapsed = chrono::high_resolution_clock::now() - start;
        long long millis = chrono::duration_cast<chrono::microseconds>(elapsed).count() / 1000;
        int fps = (1/(millis/1000.0));

        putText(img, to_string(fps), Point(40, 40),
            fontface, fontscale, Scalar(0xff, 0x99, 0), 2);

        imshow("Disp", img);
        if (waitKey(1) >= 0)
            break;
    }

    return 0;
}
