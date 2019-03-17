#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "demo.h"
#ifdef WIN32
#include <time.h>
#include <winsock.h>
#include "gettimeofday.h"
#else
#include <sys/time.h>
#endif

#define FRAMES 3

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/version.hpp"
#ifndef CV_VERSION_EPOCH
#include "opencv2/videoio/videoio_c.h"
#endif
#include "http_stream.h"
image get_image_from_stream(CvCapture *cap);

static char **demo_names;
static image **demo_alphabet;
static int demo_classes;

static float **probs;
static box *boxes;
static network net;
static image in_s ;
static image det_s;
static CvCapture * cap;
static int cpp_video_capture = 0;
static float fps = 0;
static float demo_thresh = 0;
static int demo_ext_output = 0;

static float *predictions[FRAMES];
static int demo_index = 0;
static image images[FRAMES];
static IplImage* ipl_images[FRAMES];
static float *avg;



detection* videoDetec;
int score1 = 0;
char scorestr1[50] = "SCORE : ";
char labelstr1[50] = "Click on the car";
char greatstr1[50] = "Great";
char loststr1[50] = "too bad";
char **global_names1;
char* namess1[12];
char* object_to_choose1;
image **alphabet1, original_image1;
int clicked1 = 0;

void write_detections_to_file();
void write_to_file();
void on_mouse1(int event, int x, int y, int flags, void* userdata);
void draw_detections_cv(IplImage* show_img, int num, float thresh, box *boxes, float **probs, char **names, image **alphabet, int classes);
void draw_detections_cv_v3(IplImage* show_img, detection *dets, int num, float thresh, char **names, image **alphabet, int classes, int ext_output);
void show_image_cv_ipl(IplImage *disp, const char *name);
image get_image_from_stream_resize(CvCapture *cap, int w, int h, int c, IplImage** in_img, int cpp_video_capture, int dont_close);
image get_image_from_stream_letterbox(CvCapture *cap, int w, int h, int c, IplImage** in_img, int cpp_video_capture, int dont_close);
int get_stream_fps(CvCapture *cap, int cpp_video_capture);
IplImage* in_img;
IplImage* det_img;
IplImage* show_img;

static int flag_exit;
static int letter_box = 0;

void *fetch_in_thread(void *ptr)
{
    //in = get_image_from_stream(cap);
    int dont_close_stream = 0;    // set 1 if your IP-camera periodically turns off and turns on video-stream
    if(letter_box)
        in_s = get_image_from_stream_letterbox(cap, net.w, net.h, net.c, &in_img, cpp_video_capture, dont_close_stream);
    else
        in_s = get_image_from_stream_resize(cap, net.w, net.h, net.c, &in_img, cpp_video_capture, dont_close_stream);
    if(!in_s.data){
        //error("Stream closed.");
        printf("Stream closed.\n");
        flag_exit = 1;
        return EXIT_FAILURE;
    }
    //in_s = resize_image(in, net.w, net.h);

    return 0;
}

detection *detect_in_thread(void *ptr)
{
    float nms = .45;    // 0.4F

    layer l = net.layers[net.n-1];
    float *X = det_s.data;
    float *prediction = network_predict(net, X);

    memcpy(predictions[demo_index], prediction, l.outputs*sizeof(float));
    mean_arrays(predictions, FRAMES, l.outputs, avg);
    l.output = avg;

    free_image(det_s);

    int nboxes = 0;
    detection *dets = NULL;
    if (letter_box)
        dets = get_network_boxes(&net, in_img->width, in_img->height, demo_thresh, demo_thresh, 0, 1, &nboxes, 1); // letter box
    else
        dets = get_network_boxes(&net, det_s.w, det_s.h, demo_thresh, demo_thresh, 0, 1, &nboxes, 0); // resized
    //if (nms) do_nms_obj(dets, nboxes, l.classes, nms);    // bad results
    if (nms) do_nms_sort(dets, nboxes, l.classes, nms);


    printf("\033[2J");
    printf("\033[1;1H");
    printf("\nFPS:%.1f\n",fps);
    printf("Objects:\n\n");

    ipl_images[demo_index] = det_img;
    det_img = ipl_images[(demo_index + FRAMES / 2 + 1) % FRAMES];
    demo_index = (demo_index + 1)%FRAMES;

    draw_detections_cv_v3(det_img, dets, nboxes, demo_thresh, demo_names, demo_alphabet, demo_classes, demo_ext_output);
//    free_detections(dets, nboxes);
	
    return dets;
}

double get_wall_time()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

void demo(char *cfgfile, char *weightfile, float thresh, float hier_thresh, int cam_index, const char *filename, char **names, int classes,
    int frame_skip, char *prefix, char *out_filename, int http_stream_port, int dont_show, int ext_output)
{
    //skip = frame_skip;
    image **alphabet = load_alphabet();
    int delay = frame_skip;
	//list *options = read_data_cfg(cfgfile);
	//char *name_list = option_find_str(options, "names", "data/names.list");
	//int names_size = 0;
	//global_names1 = get_labels_custom(name_list, &names_size);
    demo_names = names;
    demo_alphabet = alphabet;
    demo_classes = classes;
    demo_thresh = thresh;
    demo_ext_output = ext_output;
    printf("Demo\n");
    net = parse_network_cfg_custom(cfgfile, 1);    // set batch=1
    if(weightfile){
        load_weights(&net, weightfile);
    }
    //set_batch_network(&net, 1);
    fuse_conv_batchnorm(net);
    calculate_binary_weights(net);
    srand(2222222);

    if(filename){
        printf("video file: %s\n", filename);
//#ifdef CV_VERSION_EPOCH    // OpenCV 2.x
//        cap = cvCaptureFromFile(filename);
//#else                    // OpenCV 3.x
        cpp_video_capture = 1;
        cap = get_capture_video_stream(filename);
//#endif
    }else{
        printf("Webcam index: %d\n", cam_index);
//#ifdef CV_VERSION_EPOCH    // OpenCV 2.x
//        cap = cvCaptureFromCAM(cam_index);
//#else                    // OpenCV 3.x
        cpp_video_capture = 1;
        cap = get_capture_webcam(cam_index);
//#endif
    }

    if (!cap) {
#ifdef WIN32
        printf("Check that you have copied file opencv_ffmpeg340_64.dll to the same directory where is darknet.exe \n");
#endif
        error("Couldn't connect to webcam.\n");
    }

    layer l = net.layers[net.n-1];
    int j;

    avg = (float *) calloc(l.outputs, sizeof(float));
    for(j = 0; j < FRAMES; ++j) predictions[j] = (float *) calloc(l.outputs, sizeof(float));
    for(j = 0; j < FRAMES; ++j) images[j] = make_image(1,1,3);

    boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
    probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
    for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes, sizeof(float *));

    flag_exit = 0;

    pthread_t fetch_thread;
    pthread_t detect_thread;

    fetch_in_thread(0);
    det_img = in_img;
    det_s = in_s;

    fetch_in_thread(0);
	videoDetec = detect_in_thread(0);
    det_img = in_img;
    det_s = in_s;

    for(j = 0; j < FRAMES/2; ++j){
        fetch_in_thread(0);
		videoDetec = detect_in_thread(0);
        det_img = in_img;
        det_s = in_s;
    }

    int count = 0;
    if(!prefix && !dont_show){
        cvNamedWindow("Demo", CV_WINDOW_NORMAL);
    //    cvMoveWindow("Demo", 0, 0);
  //      cvResizeWindow("Demo", 1352, 1013);
    }

    CvVideoWriter* output_video_writer = NULL;    // cv::VideoWriter output_video;
    if (out_filename && !flag_exit)
    {
        CvSize size;
        size.width = det_img->width, size.height = det_img->height;
        int src_fps = 25;
        src_fps = get_stream_fps(cap, cpp_video_capture);

        //const char* output_name = "test_dnn_out.avi";
        //output_video_writer = cvCreateVideoWriter(out_filename, CV_FOURCC('H', '2', '6', '4'), src_fps, size, 1);
        output_video_writer = cvCreateVideoWriter(out_filename, CV_FOURCC('D', 'I', 'V', 'X'), src_fps, size, 1);
        //output_video_writer = cvCreateVideoWriter(out_filename, CV_FOURCC('M', 'J', 'P', 'G'), src_fps, size, 1);
        //output_video_writer = cvCreateVideoWriter(out_filename, CV_FOURCC('M', 'P', '4', 'V'), src_fps, size, 1);
        //output_video_writer = cvCreateVideoWriter(out_filename, CV_FOURCC('M', 'P', '4', '2'), src_fps, size, 1);
        //output_video_writer = cvCreateVideoWriter(out_filename, CV_FOURCC('X', 'V', 'I', 'D'), src_fps, size, 1);
        //output_video_writer = cvCreateVideoWriter(out_filename, CV_FOURCC('W', 'M', 'V', '2'), src_fps, size, 1);
    }

    double before = get_wall_time();

    while(1){
        ++count;
		if (show_img) {
			CvPoint pt_text;
			pt_text.x = show_img->width - (show_img->width / 2 + show_img->width / 6);
			pt_text.y = 15; 
			CvScalar black_color;
			black_color.val[0] = 100;
			CvFont font;
			float const font_size = show_img->height / 1000.F;
	//		strcpy(labelstr1, "click on the car");
			cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, font_size, font_size, 0, font_size * 3, 8);
			cvPutText(show_img, labelstr1, pt_text, &font, black_color);

		}
        if(1){
            if(pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
            if(pthread_create(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed");

            if(!prefix){
                if (!dont_show) {
                    show_image_cv_ipl(show_img, "Demo");
                    int c = cvWaitKey(1);
                    if (c == 10) {
                        if (frame_skip == 0) frame_skip = 60;
                        else if (frame_skip == 4) frame_skip = 0;
                        else if (frame_skip == 60) frame_skip = 4;
                        else frame_skip = 0;
                    }
                    else if (c == 27 || c == 1048603) // ESC - exit (OpenCV 2.x / 3.x)
                    {
                        flag_exit = 1;
                    }
                }
            }else{
                char buff[256];
                sprintf(buff, "%s_%08d.jpg", prefix, count);
                cvSaveImage(buff, show_img, 0);
                //save_image(disp, buff);
            }

            // if you run it with param -http_port 8090  then open URL in your web-browser: http://localhost:8090
            if (http_stream_port > 0 && show_img) {
                //int port = 8090;
                int port = http_stream_port;
                int timeout = 200;
                int jpeg_quality = 30;    // 1 - 100
                send_mjpeg(show_img, port, timeout, jpeg_quality);
            }

            // save video file
            if (output_video_writer && show_img) {
                cvWriteFrame(output_video_writer, show_img);
				cvSetMouseCallback("Demo", on_mouse1, 0);
                printf("\n cvWriteFrame \n");
            }

            cvReleaseImage(&show_img);

            pthread_join(fetch_thread, 0);
            pthread_join(detect_thread, 0);

            if (flag_exit == 1) break;

            if(delay == 0){
                show_img = det_img;
            }
            det_img = in_img;
            det_s = in_s;
        }else {
            fetch_in_thread(0);
            det_img = in_img;
            det_s = in_s;
            detect_in_thread(0);

            show_img = det_img;
            if (!dont_show) {
                show_image_cv_ipl(show_img, "Demo");
                cvWaitKey(1);
            }
            cvReleaseImage(&show_img);
        }
        --delay;
        if(delay < 0){
            delay = frame_skip;

            double after = get_wall_time();
            float curr = 1./(after - before);
            fps = curr;
            before = after;
        }
    }
    printf("input video stream closed. \n");
    if (output_video_writer) {
        cvReleaseVideoWriter(&output_video_writer);
        printf("output_video_writer closed. \n");
		write_detections_to_file();
    }

    // free memory
    cvReleaseImage(&show_img);
    cvReleaseImage(&in_img);
    free_image(in_s);

    free(avg);
    for (j = 0; j < FRAMES; ++j) free(predictions[j]);
    for (j = 0; j < FRAMES; ++j) free_image(images[j]);

    for (j = 0; j < l.w*l.h*l.n; ++j) free(probs[j]);
    free(boxes);
    free(probs);

    free_ptrs(names, net.layers[net.n - 1].classes);

    int i;
    const int nsize = 8;
    for (j = 0; j < nsize; ++j) {
        for (i = 32; i < 127; ++i) {
            free_image(alphabet[j][i]);
        }
        free(alphabet[j]);
    }
    free(alphabet);

    free_network(net);
}


void write_detections_to_file() {
	FILE *f = fopen("C:\\Users\\shahar\\Documents\\Visual Studio 2015\\Projects\\opencvtestt\\opencvtestt\\detections.txt", "w");
	if (f == NULL)
	{
		printf("Error opening file!\n");
		exit(1);
	}

	/* print some text */
	char str[12];
	sprintf(str, "%d", frame_id);
	fprintf(f, str);
	fprintf(f, "\n");
	for (int i = 1; i < frame_id; i++) {
		fprintf(f, arrayOfDetections[i]);
		fprintf(f, "\n");
	}

	fclose(f);
}

void on_mouse1(int event, int x, int y, int flags, void* userdata)
{
	int x1, y1, h1, w1, tempnum;
	int clicked = 0;
	char tempstr[20];
	float rgb[3] = { 100, 150, 200 };
	char* new_object_to_choose = "";
//	image temp = copy_image(original_image1);

	if (event == CV_EVENT_LBUTTONDOWN)
	{
		printf("%d", frame_id);

		printf("mouse position : %d, %d", x, y);
		printf("\n");
		//printf("detection positions : %d, %d", dets1[0].bbox.x, dets1[i].bbox.x + dets1[0].bbox.w);
		clicked = 0;
		int i;
		for ( i = 0; i < 3; i++)		// dets_num1
		{
			x1 = videoDetec[i].bbox.x;
			y1 = videoDetec[i].bbox.y;
			h1 = videoDetec[i].bbox.h;
			w1 = videoDetec[i].bbox.w;
			tempnum = global_selected_detections[i].best_class;
			strcpy(tempstr, demo_names[global_selected_detections[i].best_class]);
			printf("\n");
			printf("detection positions X : %d, %d", x1, w1);
			printf("\n");
			printf("detection positions Y: %d, %d", y1, h1);
			printf("\n");
			printf("i: %d", i);
			printf("\n");
			printf("object name: %s", tempstr);
			printf("\n");
			if (x1 < x && x < w1 && y1 < y && y < h1 )//&& (strcmp(tempstr, object_to_choose1) == 0))
			{
				clicked = 1;
			}
		}

		char str[10];

		if (clicked == 1) {
			score1 += 10;
			printf("Clicked");
			strcpy(labelstr1, "GREAT");
//			write_text_image(alphabet1, greatstr1, (temp.h*.03), temp, rgb, 15, temp.w - (temp.w / 2 + temp.w / 4));
		}
		else {
			printf("Not Clicked");
			strcpy(labelstr1, "OH NO");
			if (score1 > 0)
				score1 -= 5;
//			printf("Clicked");
//			write_text_image(alphabet1, loststr1, (temp.h*.03), temp, rgb, 15, temp.w - (temp.w / 2 + temp.w / 4));
		}
/*
		char score_str[20] = "SCORE : ";
		sprintf(str, "%d", score1);
		strcat(score_str, str);
		write_text_image(alphabet1, score_str, (temp.h*.03), temp, rgb, temp.h - 10, temp.w - temp.w / 4);
		//	show_image(temp, "Camera1");

		new_object_to_choose = get_object_to_choose();
		object_to_choose1 = new_object_to_choose;
		temp = copy_image(original_image1);
		strcpy(labelstr1, "Click on the ");
		strcat(labelstr1, new_object_to_choose);
		write_text_image(alphabet1, labelstr1, (temp.h*.03), temp, rgb, 15, temp.w - (temp.w / 2 + temp.w / 4));
		write_text_image(alphabet1, score_str, (temp.h*.03), temp, rgb, temp.h - 10, temp.w - temp.w / 4);
		clock_t start_time = clock();
		while (clock() < (start_time + 1000));
		show_image(temp, "Camera");
		cvSetMouseCallback("Camera", on_mouse1, 0);*/
	}
}


#else
void demo(char *cfgfile, char *weightfile, float thresh, float hier_thresh, int cam_index, const char *filename, char **names, int classes,
    int frame_skip, char *prefix, char *out_filename, int http_stream_port, int dont_show, int ext_output)
{
    fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}
#endif

