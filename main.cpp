#include <iostream>
#include <string>
#include <chrono>
#include <memory>
#include <array>
#include <vector>

// LibMP Header
#include "libmp.h"

// Compiled protobuf headers for MediaPipe types used
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/image_format.pb.h"

// OpenCV
#include <opencv2/opencv.hpp>

////////////////////////////////////////
//           Helper Function          //
////////////////////////////////////////

// returns landmark XYZ data for all detected faces (or empty vector if no detections)
// dimensions = (# faces) x (# landmarks/face) x 3
// i.e., each landmark is a 3-float array (X,Y,Z), so the middle vector contains 468 or 478 of these
// and the outermost vector is for each detected face in the frame
static std::vector<std::vector<std::array<float, 3>>> get_landmarks(const std::shared_ptr<mediapipe::LibMP>& face_mesh) {
    std::vector<std::vector<std::array<float, 3>>> normalized_landmarks;

    // I use a unique_ptr for convenience, so that DeletePacket is called automatically
    // You could also manage deletion yourself, manually:
    // const void* packet = face_mesh->GetOutputPacket("landmarks");
    // mediapipe::LibMP::DeletePacket(packet);
    std::unique_ptr<const void, decltype(&mediapipe::LibMP::DeletePacket)> lm_packet_ptr(nullptr, mediapipe::LibMP::DeletePacket);

    // Keep getting packets from queue until empty
    while (face_mesh->GetOutputQueueSize("landmarks") > 0) {
        lm_packet_ptr.reset(face_mesh->GetOutputPacket("landmarks"));
    }
    if (lm_packet_ptr.get() == nullptr || mediapipe::LibMP::PacketIsEmpty(lm_packet_ptr.get())) {
        return normalized_landmarks; // return empty vector if no output packets or packet is invalid
    }

    // Create landmarks from packet's protobuf data
    size_t num_faces = mediapipe::LibMP::GetPacketProtoMsgVecSize(lm_packet_ptr.get());
    for (int face_num = 0; face_num < num_faces; face_num++) {
        // Get reference to protobuf message for face
        const void* lm_list_proto = mediapipe::LibMP::GetPacketProtoMsgAt(lm_packet_ptr.get(), face_num);
        // Get byte size of protobuf message
        size_t lm_list_proto_size = mediapipe::LibMP::GetProtoMsgByteSize(lm_list_proto);

        // Create buffer to hold protobuf message data; copy data to buffer
        std::shared_ptr<uint8_t[]> proto_data(new uint8_t[lm_list_proto_size]);
        mediapipe::LibMP::WriteProtoMsgData(proto_data.get(), lm_list_proto, static_cast<int>(lm_list_proto_size));

        // Initialize a mediapipe::NormalizedLandmarkList object from the buffer
        mediapipe::NormalizedLandmarkList face_landmarks;
        face_landmarks.ParseFromArray(proto_data.get(), static_cast<int>(lm_list_proto_size));

        // Copy the landmark data to our custom data structure
        normalized_landmarks.emplace_back();
        for (const mediapipe::NormalizedLandmark& lm : face_landmarks.landmark()) {
            normalized_landmarks[face_num].push_back({ lm.x(), lm.y(), lm.z() });
        }
    }

    return normalized_landmarks;
}

////////////////////////////////////////
//            Main Function           //
////////////////////////////////////////

int main(int argc, char* argv[]) {
    // adapted from https://github.com/google/mediapipe/blob/master/mediapipe/graphs/face_mesh/face_mesh_desktop_live.pbtxt
    // runs face mesh for up to 1 face with both attention and previous landmark usage enabled
    const char* graph = R"(
       # MediaPipe graph that performs hands tracking on desktop with TensorFlow
# Lite on CPU.
# Used in the example in
# mediapipe/examples/desktop/hand_tracking:hand_tracking_cpu.

# CPU image. (ImageFrame)
input_stream: "input_video"

# CPU image. (ImageFrame)
output_stream: "output_video"

output_stream: "landmarks"

# Generates side packet cotaining max number of hands to detect/track.
node {
  calculator: "ConstantSidePacketCalculator"
  output_side_packet: "PACKET:num_hands"
  node_options: {
    [type.googleapis.com/mediapipe.ConstantSidePacketCalculatorOptions]: {
      packet { int_value: 2 }
    }
  }
}

# Detects/tracks hand landmarks.
node {
  calculator: "HandLandmarkTrackingCpu"
  input_stream: "IMAGE:input_video"
  input_side_packet: "NUM_HANDS:num_hands"
  output_stream: "LANDMARKS:landmarks"
  output_stream: "HANDEDNESS:handedness"
  output_stream: "PALM_DETECTIONS:multi_palm_detections"
  output_stream: "HAND_ROIS_FROM_LANDMARKS:multi_hand_rects"
  output_stream: "HAND_ROIS_FROM_PALM_DETECTIONS:multi_palm_rects"
}

# Subgraph that renders annotations and overlays them on top of the input
# images (see hand_renderer_cpu.pbtxt).
node {
  calculator: "HandRendererSubgraph"
  input_stream: "IMAGE:input_video"
  input_stream: "DETECTIONS:multi_palm_detections"
  input_stream: "LANDMARKS:landmarks"
  input_stream: "HANDEDNESS:handedness"
  input_stream: "NORM_RECTS:0:multi_palm_rects"
  input_stream: "NORM_RECTS:1:multi_hand_rects"
  output_stream: "IMAGE:output_video"
}

    )";

    // Create MP face mesh graph
    std::shared_ptr<mediapipe::LibMP> face_mesh(mediapipe::LibMP::Create(graph, "input_video"));

    // MP-rendered output stream of FaceRendererCpu subgraph
    // NOTE: only enable if needed; otherwise, output image packets will queue up & consume memory
    // face_mesh->AddOutputStream("output_video");

    // Landmark XYZ data output stream
    face_mesh->AddOutputStream("landmarks");

    // Start MP graph
    face_mesh->Start();

    // Stream from webcam (device #0)
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Could not open device #0. Is a camera/webcam attached?" << std::endl;
        return EXIT_FAILURE;
    }

    cv::Mat frame_bgr;
    while (cap.read(frame_bgr)) {
        // Convert frame from BGR to RGB
        cv::Mat frame_rgb;
        cv::cvtColor(frame_bgr, frame_rgb, cv::COLOR_BGR2RGB);

        // Start inference clock
        auto t0 = std::chrono::high_resolution_clock::now();

        // Feed RGB frame into MP face mesh graph (image data is COPIED internally by LibMP)
        if (!face_mesh->Process(frame_rgb.data, frame_rgb.cols, frame_rgb.rows, mediapipe::ImageFormat::SRGB)) {
            std::cerr << "Process() failed!" << std::endl;
            break;
        }
        face_mesh->WaitUntilIdle();

        // Stop inference clock
        auto t1 = std::chrono::high_resolution_clock::now();
        int inference_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();

        // Get landmark coordinates in custom data structure using helper function (see above)
        std::vector<std::vector<std::array<float, 3>>> normalized_landmarks = get_landmarks(face_mesh);

        // For each face, draw a circle at each landmark's position
        size_t num_faces = normalized_landmarks.size();
        for (int face_num = 0; face_num < num_faces; face_num++) {
            for (const std::array<float, 3>& norm_xyz : normalized_landmarks[face_num]) {
                int x = static_cast<int>(norm_xyz[0] * frame_bgr.cols);
                int y = static_cast<int>(norm_xyz[1] * frame_bgr.rows);
                cv::circle(frame_bgr, cv::Point(x, y), 1, cv::Scalar(0, 255, 0), -1);
            }
        }

        // Write some info on frame
        cv::putText(frame_bgr, "Press any key to exit", cv::Point(10, 20), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 255, 0));
        cv::putText(frame_bgr, "# Faces Detected: " + std::to_string(num_faces), cv::Point(10, 40), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 255, 0));
        cv::putText(frame_bgr, "Inference time: " + std::to_string(inference_time_ms) + " ms", cv::Point(10, 60), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 255, 0));

        // Display frame
        cv::imshow("LibMP Example", frame_bgr);

        // Close on any keypress
        if (cv::waitKey(1) >= 0) {
            break;
        }
    }

    cv::destroyAllWindows();
    return EXIT_SUCCESS;
}
