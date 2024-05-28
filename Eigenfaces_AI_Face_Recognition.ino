#define CAMERA_MODEL_XIAO_ESP32S3 // Has PSRAM

// Libraries included
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "pca_mean_vector.h"
#include <string.h>
#include "pca_transformation_matrix.h"
#include "class_centroids.h"
#include "esp_timer.h"
#include "esp_camera.h"
#include "fb_gfx.h"
#include "camera_pins.h"
#include "img_converters.h"
#include "human_face_detect_msr01.hpp"
#include "human_face_detect_mnp01.hpp"
#include <WiFi.h>
#include <WiFiClient.h>
#include <esp_http_client.h>
#include <Base64.h>
#include "FS.h"
#include "SD.h"
#include "SPI.h"
#include "allam_CAM.h"
#include "pca_eigenvalues.h"
#include <time.h>

#define WIFI_SSID "Fritz!Box 7430 JN"
#define WIFI_PASSWORD "69453963022239318486"
#define SERVER_URL "http://192.168.188.171:3500/api/face_recognition"



// Predefined Attributes
#define BASE_PRIORITY 1
#define NUM_COMPONENTS 18
#define VECTOR_SIZE 4096
#define STACK_SIZE 10000
#define HALF_VECTOR_SIZE 2048
#define FACE_COLOR_YELLOW 0xFFFF00
#define TWO_STAGE 1
#define Normalized_Detected_Face_IMG_SIZE 4096
#define RGB_BRIGHTNESS 255

HumanFaceDetectMSR01 stage1(0.1F, 0.5F, 10, 0.2F);
HumanFaceDetectMNP01 stage2(0.5F, 0.3F, 5);

uint64_t start_Face_Recognition_time = 0;
uint64_t end_Face_Recognition_time = 0;

unsigned long lastCaptureTime = 0; // Last shooting time
int imageCount = 1;                // File Counter
bool camera_sign = false;          // Check camera status
int Centering_Data_Counter = 0;

// Semaphores
SemaphoreHandle_t sync_PCA_Loop_Semaphore;
SemaphoreHandle_t pca_Task_Core_0_Semaphore;
SemaphoreHandle_t pca_Task_Core_1_Semaphore;
SemaphoreHandle_t centering_CAM_Face_DATA_Semaphore;
SemaphoreHandle_t execute_PCA_Grp_Semaphore;
SemaphoreHandle_t execute_Eigen_Faces_Task_Semaphore;
SemaphoreHandle_t image_Capture_Semaphore;

// DataStructures
float PCA_Output_Vector[NUM_COMPONENTS];
float Centered_CAM_Face_Data[VECTOR_SIZE];
float Normalized_Detected_Face_IMG[Normalized_Detected_Face_IMG_SIZE];

// Methods and Functions Definitions
void apply_PCA(int idx, int vectorStart, int vectorEnd);
void pca_Task(void *params);
void execute_Eigen_Faces_Projection(void *params);
float compute_Euclidean_Distance(const float* a, const float* b, int length);
void centering_CAM_Face_Data(const float *Normalized_Detected_Face_IMG, float *centered);
void crop_CAM_Face_IMG(uint8_t *src, uint8_t *dest, int src_width, int crop_x, int crop_y, int crop_width, int crop_height);
void convert_RGB565_to_Gray_Scale(uint8_t *src, uint8_t *dest, int width, int height);
void resize_CAM_Face_IMG(uint8_t *src, int src_width, int src_height, uint8_t *dest, int dest_width, int dest_height);
void process_CAM_Face_IMG(camera_fb_t *fb, const dl::detect::result_t& result);
void detect_Face(camera_fb_t *fb);
void execute_PCA_Task_Grp();
void apply_Centering_CAM_Face_Data(void *params);
void pcaTask_Core_0(void *params);
void pcaTask_Core_1(void *params);

// Methods Implementations


// Save pictures to SD card
void photo_save( uint8_t *jpeg_buf ,size_t jpeg_len ,const char * fileName) {
    writeFile(SD, fileName, jpeg_buf, jpeg_len);  
    Serial.println("Photo saved to file");
}

// SD card write file
void writeFile(fs::FS &fs, const char * path, uint8_t * data, size_t len){
    Serial.printf("Writing file: %s\n", path);

    File file = fs.open(path, FILE_WRITE);
    if(!file){
        Serial.println("Failed to open file for writing");
        return;
    }
    if(file.write(data, len) == len){
        Serial.println("File written");
    } else {
        Serial.println("Write failed");
    }
    file.close();
}

void centering_CAM_Face_Data(const float *Normalized_Detected_Face_IMG, float* Centered_CAM_Face_Data) {
    static int params_core_0[2];
    params_core_0[0] = 0;
    params_core_0[1] = HALF_VECTOR_SIZE;

    static int params_core_1[2];
    params_core_1[0] = HALF_VECTOR_SIZE;
    params_core_1[1] = VECTOR_SIZE;

    xTaskCreatePinnedToCore(apply_Centering_CAM_Face_Data, "Centering_CAM_Face_Data_Core_0", STACK_SIZE, params_core_0, BASE_PRIORITY, NULL, 0);
    xTaskCreatePinnedToCore(apply_Centering_CAM_Face_Data, "Centering_CAM_Face_Data_Core_1", STACK_SIZE, params_core_1, BASE_PRIORITY, NULL, 1);
}

void apply_Centering_CAM_Face_Data(void *params) {
    int VECTOR_START = ((int*)params)[0];
    int VECTOR_END = ((int*)params)[1];
    for (int i = VECTOR_START; i < VECTOR_END; i += 64) {
        for (int j = 0; j < 64; j++) {
            Centered_CAM_Face_Data[i + j] = Normalized_Detected_Face_IMG[i + j] - (pca_mean_vector[i + j]);
        }
    }
    xSemaphoreGive(centering_CAM_Face_DATA_Semaphore);
    vTaskDelete(NULL);
}

void applyPCA(int i, int vectorStart, int vectorEnd) {
    float sum = 0;
    for (int j = vectorStart; j < vectorEnd; j += 64) {
        for (int k = 0; k < 64; k++) {
            sum += (Centered_CAM_Face_Data[j + k] * (pca_transformation_matrix[i][j + k]));
        }
    }
      
      PCA_Output_Vector[i] += (sum / sqrt(pca_eigenvalues[i]));  // Apply whitening
  }

void pcaTask_Core_0(void *params) {
    int idx = (int)params;
    applyPCA(idx, 0, HALF_VECTOR_SIZE);
    delay(5);
    xSemaphoreGive(pca_Task_Core_0_Semaphore);
    vTaskDelete(NULL);
}

void pcaTask_Core_1(void *params) {
    int idx = (int)params;
    applyPCA(idx, HALF_VECTOR_SIZE, VECTOR_SIZE);
    delay(5);
    xSemaphoreGive(pca_Task_Core_1_Semaphore);
    vTaskDelete(NULL);
}

void createPCATasks(int idx) {
    xTaskCreatePinnedToCore(pcaTask_Core_0, "PCA_Task_Core0", STACK_SIZE, (void *)idx, BASE_PRIORITY, NULL, 0);
    xTaskCreatePinnedToCore(pcaTask_Core_1, "PCA_Task_Core1", STACK_SIZE, (void *)idx, BASE_PRIORITY, NULL, 1);
}

void execute_PCA_Task_Grp() {
    if (xSemaphoreTake(execute_PCA_Grp_Semaphore, portMAX_DELAY)) {
        createPCATasks(0);
        for (int i = 1; i < NUM_COMPONENTS; i++) {
            if(xSemaphoreTake(pca_Task_Core_0_Semaphore, portMAX_DELAY)  && xSemaphoreTake(pca_Task_Core_1_Semaphore , portMAX_DELAY )) {
                createPCATasks(i);
            }
        }
        xSemaphoreGive(execute_PCA_Grp_Semaphore);
        if(xSemaphoreTake(pca_Task_Core_0_Semaphore, portMAX_DELAY)  && xSemaphoreTake(pca_Task_Core_1_Semaphore , portMAX_DELAY )) {
            xTaskCreate(execute_Eigen_Faces_Projection, "Eigenfaces", STACK_SIZE, NULL, BASE_PRIORITY, NULL);
            xSemaphoreGive(execute_Eigen_Faces_Task_Semaphore);
            xSemaphoreGive(pca_Task_Core_0_Semaphore);
            xSemaphoreGive(pca_Task_Core_1_Semaphore);
        }
    }
}


void execute_Eigen_Faces_Projection(void *params) {
    if (xSemaphoreTake(execute_Eigen_Faces_Task_Semaphore, portMAX_DELAY) && xSemaphoreTake(pca_Task_Core_0_Semaphore, portMAX_DELAY) && xSemaphoreTake(pca_Task_Core_1_Semaphore, portMAX_DELAY)) {
        float min_distance = 4.0;
        int class_label = -1;
        // Iterate through each class centroid and calculate Euclidean distance to the output vector
        for (int i = 0; i < 3; i++) {
          float distance = compute_Euclidean_Distance(PCA_Output_Vector, class_centroids[i], NUM_COMPONENTS);
          
          if (distance <= min_distance) {
              min_distance = distance;
              class_label = i;
          }
        }
        end_Face_Recognition_time = esp_timer_get_time(); // Capture end time
        Serial.print("Execution Time (ms): ");
        Serial.println((end_Face_Recognition_time - start_Face_Recognition_time) / 1000); // Print execution time in milliseconds
        //  if (class_label != -1) {
        //     blink_TWICE_LED(); // Blink twice for Face Recognition
        // } else {
        //     blink_ONCE_LED(); // Blink once for Face Recognition
        // }
        Serial.print("Predicted class :");
        Serial.println(class_label);
        Serial.println();
        Serial.print("Min_Distance :");
        Serial.println(min_distance);
        sendResultsToServer(class_label, min_distance);
        reset_PCA_Output_Vector();
        xSemaphoreGive(image_Capture_Semaphore);  // Allow frame capturing to resume after processing
        vTaskDelete(NULL);
    }
}


float compute_Euclidean_Distance(const float* a, const float* b, int length) {
    float distance = 0;
    for (int i = 0; i < length; i++) {
        float diff = a[i] - b[i];
        distance += (diff * diff);
    }
    return sqrt(distance);
}

// Function to crop the camera face image
void crop_CAM_Face_IMG(uint8_t *src, uint8_t *dest, int src_width, int crop_x, int crop_y, int crop_width, int crop_height) {
    for (int y = 0; y < crop_height; y++) {
        memcpy(dest + y * crop_width * 2, src + ((crop_y + y) * src_width + crop_x) * 2, crop_width * 2);
    }
}

// Function to convert RGB565 to grayscale
void convert_RGB565_to_Gray_Scale(uint8_t *src, uint8_t *dest, int width, int height) {
    for (int i = 0; i < width * height; i++) {
        uint16_t pixel = (src[2 * i] << 8) | src[2 * i + 1];
        uint8_t r = (pixel >> 11) & 0x1F;
        uint8_t g = (pixel >> 5) & 0x3F;
        uint8_t b = pixel & 0x1F;
        // Convert to grayscale using luminosity method
        uint8_t gray = (uint8_t)((r * 0.299 * 255.0 / 31) + (g * 0.587 * 255.0 / 63) + (b * 0.114 * 255.0 / 31));
        dest[i] = gray;
    }
}

// Function to resize the camera face image
void resize_CAM_Face_IMG(uint8_t *src, int src_width, int src_height, uint8_t *dest, int dest_width, int dest_height) {
    float x_scale = src_width / (float) dest_width;
    float y_scale = src_height / (float) dest_height;

    for (int y = 0; y < dest_height; y++) {
        for (int x = 0; x < dest_width; x++) {
            int src_x = (int)(x * x_scale);
            int src_y = (int)(y * y_scale);
            src_x = src_x < src_width ? src_x : src_width - 1;   // Clamping to ensure we don't go out of bounds
            src_y = src_y < src_height ? src_y : src_height - 1; // Clamping to ensure we don't go out of bounds
            dest[y * dest_width + x] = src[src_y * src_width + src_x];
        }
    }
}

// Function to process the camera face image
void process_CAM_Face_IMG(camera_fb_t *fb, const dl::detect::result_t& result) {
    static int imageCount = 0;

    // Convert the original image to JPEG
    uint8_t *jpeg_buf = NULL;
    size_t jpeg_len = 0;
    bool converted = fmt2jpg(fb->buf, fb->len, fb->width, fb->height, PIXFORMAT_RGB565, 100, &jpeg_buf, &jpeg_len);
    if (!converted) {
        ESP_LOGE("JPEG Conversion", "Conversion to JPEG failed");
        return;
    }

    // Create the filename for the original JPEG image
    char filename[32];
    sprintf(filename, "/image%d.jpg", imageCount);
    if (jpeg_buf != NULL) {
        photo_save(jpeg_buf, jpeg_len, filename);
        free(jpeg_buf);
        jpeg_len = 0;
    }

    imageCount++;

    int x = result.box[0];
    int y = result.box[1];
    int w = result.box[2] - result.box[0];
    int h = result.box[3] - result.box[1];

    uint8_t *cropped_img = (uint8_t *)malloc(w * h * 2);
    if (!cropped_img) {
        Serial.println("Failed to allocate memory for cropped image");
        return;
    }
    crop_CAM_Face_IMG(fb->buf, cropped_img, fb->width, x, y, w, h);

    // Convert the cropped image to JPEG and save it
    // converted = fmt2jpg(cropped_img, w * h * 2, w, h, PIXFORMAT_RGB565, 90, &jpeg_buf, &jpeg_len);
    // if (converted) {
    //     sprintf(filename, "/cropped_image%d.jpg", imageCount);
    //     photo_save(jpeg_buf, jpeg_len, filename);
    //     free(jpeg_buf);
    //     jpeg_len = 0;
    // } else {
    //     ESP_LOGE("JPEG Conversion", "Conversion of cropped image to JPEG failed");
    // }

    uint8_t *gray_img = (uint8_t *)malloc(w * h);
    if (!gray_img) {
        Serial.println("Failed to allocate memory for grayscale image");
        // free(cropped_img);
        return;
    }
    convert_RGB565_to_Gray_Scale(cropped_img, gray_img, w, h);

    // Convert the grayscale image to JPEG and save it
    // converted = fmt2jpg(gray_img, w * h, w, h, PIXFORMAT_GRAYSCALE, 90, &jpeg_buf, &jpeg_len);
    // if (converted) {
    //     sprintf(filename, "/gray_image%d.jpg", imageCount);
    //     photo_save(jpeg_buf, jpeg_len, filename);
    //     free(jpeg_buf);
    //     jpeg_len = 0;
    // } else {
    //     ESP_LOGE("JPEG Conversion", "Conversion of grayscale image to JPEG failed");
    // }

    uint8_t *resized_img = (uint8_t *)malloc(64 * 64);
    if (!resized_img) {
        Serial.println("Failed to allocate memory for resized image");
        // free(cropped_img);
        free(gray_img);
        return;
    }
    resize_CAM_Face_IMG(gray_img, w, h, resized_img, 64, 64);

    // Convert the resized image to JPEG and save it
    // converted = fmt2jpg(resized_img, 64 * 64, 64, 64, PIXFORMAT_GRAYSCALE, 90, &jpeg_buf, &jpeg_len);
    // if (converted) {
    //     sprintf(filename, "/resized_image%d.jpg", imageCount);
    //     photo_save(jpeg_buf, jpeg_len, filename);
    //     free(jpeg_buf);
    //     jpeg_len = 0;
    // } else {
    //     ESP_LOGE("JPEG Conversion", "Conversion of resized image to JPEG failed");
    // }

    // Normalize the image data
    for (int i = 0; i < 64 * 64; i++) {
        Normalized_Detected_Face_IMG[i] = resized_img[i] / 255.0f;
    }
    centering_CAM_Face_Data(Normalized_Detected_Face_IMG, Centered_CAM_Face_Data);

    delay(5);
    if (uxSemaphoreGetCount(centering_CAM_Face_DATA_Semaphore) == 2) {
        xSemaphoreTake(centering_CAM_Face_DATA_Semaphore, portMAX_DELAY);
        xSemaphoreTake(centering_CAM_Face_DATA_Semaphore, portMAX_DELAY);
        xSemaphoreGive(execute_PCA_Grp_Semaphore);
        execute_PCA_Task_Grp();
    }

    // Cleanup
    // free(cropped_img);
    free(gray_img);
    free(resized_img);
}

void reset_PCA_Output_Vector() {
    for (int i = 0; i < NUM_COMPONENTS; i++) {
        PCA_Output_Vector[i] = 0.0f;
    }
}

void detect_Face(camera_fb_t *fb) {
    if (!fb) return;
    std::list<dl::detect::result_t> results;

    if (TWO_STAGE) {
        auto candidates = stage1.infer((uint16_t *)fb->buf, {fb->height, fb->width, 3});
        results = stage2.infer((uint16_t *)fb->buf, {fb->height, fb->width, 3}, candidates);
    } else {
        results = stage1.infer((uint16_t *)fb->buf, {fb->height, fb->width, 3});
    }

    if (!results.empty()) {
        Serial.println("Face Detected Successfully !!!!!");
        start_Face_Recognition_time = esp_timer_get_time(); // Capture start time
        for (const auto& result : results) {
            process_CAM_Face_IMG(fb, result);
        }
    } else {
        Serial.println("no face detected !");
        xSemaphoreGive(image_Capture_Semaphore);
    }
    esp_camera_fb_return(fb);
}
void sendResultsToServer(int class_label, float min_distance) {
    esp_http_client_config_t config = {
        .url = SERVER_URL,
        .method = HTTP_METHOD_POST
    };
    esp_http_client_handle_t client = esp_http_client_init(&config);
    if (client == NULL) {
        Serial.println("Failed to initialize HTTP client");
        return;
    }
    Serial.println("HTTP client initialized");

    char jsonPayload[100];
    snprintf(jsonPayload, sizeof(jsonPayload), "{\"class_label\": %d, \"min_distance\": %.2f}", class_label, min_distance);
    Serial.print("JSON Payload: ");
    Serial.println(jsonPayload);

    esp_http_client_set_header(client, "Content-Type", "application/json");
    esp_http_client_set_post_field(client, jsonPayload, strlen(jsonPayload));

    esp_err_t err = esp_http_client_perform(client);
    if (err == ESP_OK) {
        Serial.printf("HTTP POST Status = %d, content_length = %d\n",
                      esp_http_client_get_status_code(client),
                      esp_http_client_get_content_length(client));
    } else {
        Serial.printf("HTTP POST request failed: %s\n", esp_err_to_name(err));
    }

    esp_http_client_cleanup(client);
}

void setup() {
    Serial.begin(115200);
    while (!Serial);

    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    config.xclk_freq_hz = 2000000;
    config.frame_size = FRAMESIZE_240X240;
    config.pixel_format = PIXFORMAT_RGB565; // for face detection/recognition
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.jpeg_quality = 12;
    config.fb_count = 1;

    if(config.pixel_format == PIXFORMAT_JPEG) {
        if(psramFound()) {
            config.jpeg_quality = 10;
            config.fb_count = 2;
            config.grab_mode = CAMERA_GRAB_LATEST;
        } else {
            // Limit the frame size when PSRAM is not available
            config.frame_size = FRAMESIZE_SVGA;
            config.fb_location = CAMERA_FB_IN_DRAM;
        }
    } else {
        #if CONFIG_IDF_TARGET_ESP32S3
            config.fb_count = 2;
        #endif
    }

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Camera init failed with error 0x%x", err);
        return;
    }
    if(!SD.begin(21)){
    Serial.println("Card Mount Failed");
    return;
  }
  uint8_t cardType = SD.cardType();

  // Determine if the type of SD card is available
  if(cardType == CARD_NONE){
    Serial.println("No SD card attached");
    return;
  }

  Serial.print("SD Card Type: ");
  if(cardType == CARD_MMC){
    Serial.println("MMC");
  } else if(cardType == CARD_SD){
    Serial.println("SDSC");
  } else if(cardType == CARD_SDHC){
    Serial.println("SDHC");
  } else {
    Serial.println("UNKNOWN");
  }


    execute_PCA_Grp_Semaphore = xSemaphoreCreateBinary();
    execute_Eigen_Faces_Task_Semaphore = xSemaphoreCreateBinary();
    sync_PCA_Loop_Semaphore = xSemaphoreCreateMutex();
    pca_Task_Core_0_Semaphore = xSemaphoreCreateBinary();
    pca_Task_Core_1_Semaphore = xSemaphoreCreateBinary();
    centering_CAM_Face_DATA_Semaphore = xSemaphoreCreateCounting(2, 0);
    image_Capture_Semaphore = xSemaphoreCreateBinary();

    if (!sync_PCA_Loop_Semaphore  || !pca_Task_Core_0_Semaphore || !image_Capture_Semaphore 
    || !Centered_CAM_Face_Data || !execute_PCA_Grp_Semaphore   || !execute_Eigen_Faces_Task_Semaphore 
    || !centering_CAM_Face_DATA_Semaphore) {
        Serial.println("Failed to create task Semaphore or MUTEX!");
        return;
    }
    
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("");
    Serial.println("WiFi connected");
    Serial.println("IP address: ");
    Serial.println(WiFi.localIP());
    xSemaphoreGive(image_Capture_Semaphore);
}


void loop() {

    if ( xSemaphoreTake(image_Capture_Semaphore, portMAX_DELAY)) {
        camera_fb_t *fb = esp_camera_fb_get();
        if (!fb) {
            Serial.println("Camera capture failed");
        } else {
            detect_Face(fb);
          
        }
    }
}





// without sd card
// #define CAMERA_MODEL_XIAO_ESP32S3 // Has PSRAM

// // Libraries included
// #include "freertos/FreeRTOS.h"
// #include "freertos/task.h"
// #include "freertos/semphr.h"
// #include "pca_mean_vector.h"
// #include <string.h>
// #include "pca_transformation_matrix.h"
// #include "class_centroids.h"
// #include "esp_timer.h"
// #include "esp_camera.h"
// #include "fb_gfx.h"
// #include "camera_pins.h"
// #include "img_converters.h"
// #include "human_face_detect_msr01.hpp"
// #include "human_face_detect_mnp01.hpp"
// #include <WiFi.h>
// #include <WiFiClient.h>
// #include <esp_http_client.h>
// #include <Base64.h>  // Include the Base64 library
// // #include "bilieEillish_p.h"



// #define WIFI_SSID "Fritz!Box 7430 JN"
// #define WIFI_PASSWORD "69453963022239318486"
// #define SERVER_URL "http://192.168.188.171:3500/api/face_recognition"


// // Predefined Attributes
// #define NUM_TASKS 20
// #define BASE_PRIORITY 1
// #define NUM_COMPONENTS 20
// #define VECTOR_SIZE 4096
// #define STACK_SIZE 10000
// #define HALF_VECTOR_SIZE 2048
// #define FIRST_HALF_COMPONENTS 12
// #define SECOND_HALF_COMPONENTS 13
// #define FACE_COLOR_YELLOW 0xFFFF00
// #define TWO_STAGE 1
// #define Normalized_Detected_Face_IMG_SIZE 4096
// #define RGB_BRIGHTNESS 255

// uint64_t start_Face_Recognition_time = 0;
// uint64_t end_Face_Recognition_time = 0;

// HumanFaceDetectMSR01 stage1(0.1F, 0.5F, 10, 0.2F);
// HumanFaceDetectMNP01 stage2(0.5F, 0.3F, 5);

// int Centering_Data_Counter = 0;

// // Semaphores
// SemaphoreHandle_t sync_PCA_Loop_Semaphore;
// SemaphoreHandle_t pca_Task_Core_0_Semaphore;
// SemaphoreHandle_t pca_Task_Core_1_Semaphore;
// SemaphoreHandle_t centering_CAM_Face_DATA_Semaphore;
// SemaphoreHandle_t execute_PCA_Grp_1_Semaphore;
// SemaphoreHandle_t execute_PCA_Grp_2_Semaphore;
// SemaphoreHandle_t execute_PCA_Grp_3_Semaphore;
// SemaphoreHandle_t execute_PCA_Grp_4_Semaphore;
// SemaphoreHandle_t execute_PCA_Grp_5_Semaphore;
// SemaphoreHandle_t execute_Eigen_Faces_Task_Semaphore;
// SemaphoreHandle_t image_Capture_Semaphore;

// // DataStructures
// float PCA_Output_Vector[NUM_COMPONENTS];
// float Centered_CAM_Face_Data[VECTOR_SIZE];
// float Normalized_Detected_Face_IMG[Normalized_Detected_Face_IMG_SIZE];

// // Methods and Functions Definitions
// void apply_PCA(int idx, int vectorStart, int vectorEnd);
// void pca_Task(void *params);
// void execute_Eigen_Faces_Projection(void *params);
// float compute_Euclidean_Distance(const float* a, const float* b, int length);
// void centering_CAM_Face_Data(const float *Normalized_Detected_Face_IMG, float *centered);
// void crop_CAM_Face_IMG(uint8_t *src, uint8_t *dest, int src_width, int crop_x, int crop_y, int crop_width, int crop_height);
// void convert_RGB565_to_Gray_Scale(uint8_t *src, uint8_t *dest, int width, int height);
// void resize_CAM_Face_IMG(uint8_t *src, int src_width, int src_height, uint8_t *dest, int dest_width, int dest_height);
// void process_CAM_Face_IMG(camera_fb_t *fb, const dl::detect::result_t& result);
// void detect_Face(camera_fb_t *fb);
// void execute_PCA_Task_Grp_1();
// void execute_PCA_Task_Grp_2();
// void execute_PCA_Task_Grp_3();
// void execute_PCA_Task_Grp_4();
// void execute_PCA_Task_Grp_5();
// void apply_Centering_CAM_Face_Data(void *params);
// void pcaTask_Core_0(void *params);
// void pcaTask_Core_1(void *params);

// // Methods Implementations
// void centering_CAM_Face_Data(const float *Normalized_Detected_Face_IMG, float* Centered_CAM_Face_Data) {
//     static int params_core_0[2];
//     params_core_0[0] = 0;
//     params_core_0[1] = HALF_VECTOR_SIZE;

//     static int params_core_1[2];
//     params_core_1[0] = HALF_VECTOR_SIZE;
//     params_core_1[1] = VECTOR_SIZE;

//     xTaskCreatePinnedToCore(apply_Centering_CAM_Face_Data, "Centering_CAM_Face_Data_Core_0", STACK_SIZE, params_core_0, BASE_PRIORITY, NULL, 0);
//     xTaskCreatePinnedToCore(apply_Centering_CAM_Face_Data, "Centering_CAM_Face_Data_Core_1", STACK_SIZE, params_core_1, BASE_PRIORITY, NULL, 1);
// }

// void apply_Centering_CAM_Face_Data(void *params) {
//     int VECTOR_START = ((int*)params)[0];
//     int VECTOR_END = ((int*)params)[1];
//     for (int i = VECTOR_START; i < VECTOR_END; i += 64) {
//         for (int j = 0; j < 64; j++) {
//             Centered_CAM_Face_Data[i + j] = Normalized_Detected_Face_IMG[i + j] - pgm_read_float_near(&pca_mean_vector[i + j]);
//         }
//     }
//     xSemaphoreGive(centering_CAM_Face_DATA_Semaphore);
//     vTaskDelete(NULL);
// }

// void applyPCA(int i, int vectorStart, int vectorEnd) {
//     float sum = 0;
//     for (int j = vectorStart; j < vectorEnd; j += 64) {
//         for (int k = 0; k < 64; k++) {
//             sum += Centered_CAM_Face_Data[j + k] * pgm_read_float_near(&pca_transformation_matrix[i][j + k]);
//         }
//     }
//     PCA_Output_Vector[i] += sum;
// }

// void pcaTask_Core_0(void *params) {
//     int idx = (int)params;
//     applyPCA(idx, 0, HALF_VECTOR_SIZE);
//     delay(5);
//     xSemaphoreGive(pca_Task_Core_0_Semaphore);
//     vTaskDelete(NULL);
// }

// void pcaTask_Core_1(void *params) {
//     int idx = (int)params;
//     applyPCA(idx, HALF_VECTOR_SIZE, VECTOR_SIZE);
//     delay(5);
//     xSemaphoreGive(pca_Task_Core_1_Semaphore);
//     vTaskDelete(NULL);
// }

// void createPCATasks(int idx) {
//     xTaskCreatePinnedToCore(pcaTask_Core_0, "PCA_Task_Core0", STACK_SIZE, (void *)idx, BASE_PRIORITY, NULL, 0);
//     xTaskCreatePinnedToCore(pcaTask_Core_1, "PCA_Task_Core1", STACK_SIZE, (void *)idx, BASE_PRIORITY, NULL, 1);
// }

// void execute_PCA_Task_Grp_1() {
//     if (xSemaphoreTake(execute_PCA_Grp_1_Semaphore, portMAX_DELAY)) {
//         createPCATasks(0);
//         for (int i = 1; i < 5; i++) {
//             if(xSemaphoreTake(pca_Task_Core_0_Semaphore, portMAX_DELAY)  && xSemaphoreTake(pca_Task_Core_1_Semaphore , portMAX_DELAY )) {
//                 createPCATasks(i);
//             }
//         }
//         xSemaphoreGive(execute_PCA_Grp_2_Semaphore);
//         execute_PCA_Task_Grp_2();
//     }
// }

// void execute_PCA_Task_Grp_2() {
//     if (xSemaphoreTake(execute_PCA_Grp_2_Semaphore, portMAX_DELAY)) {
//         for (int i = 5; i < 10; i++) {
//             if(xSemaphoreTake(pca_Task_Core_0_Semaphore, portMAX_DELAY) && xSemaphoreTake(pca_Task_Core_1_Semaphore , portMAX_DELAY )) {
//                 createPCATasks(i);
//             }
//         }
//         xSemaphoreGive(execute_PCA_Grp_3_Semaphore);
//         xSemaphoreGive(execute_PCA_Grp_2_Semaphore);
//         execute_PCA_Task_Grp_3();
//     }
// }

// void execute_PCA_Task_Grp_3() {
//     if (xSemaphoreTake(execute_PCA_Grp_3_Semaphore, portMAX_DELAY)) {
//         for (int i = 10; i < 15; i++) {
//             if(xSemaphoreTake(pca_Task_Core_0_Semaphore, portMAX_DELAY) && xSemaphoreTake(pca_Task_Core_1_Semaphore , portMAX_DELAY )) {
//                 createPCATasks(i);
//             }
//         }
//         xSemaphoreGive(execute_PCA_Grp_4_Semaphore);
//         xSemaphoreGive(execute_PCA_Grp_3_Semaphore);
//         execute_PCA_Task_Grp_4();
//     }
// }

// void execute_PCA_Task_Grp_4() {
//     if (xSemaphoreTake(execute_PCA_Grp_4_Semaphore, portMAX_DELAY)) {
//         for (int i = 15; i < 20; i++) {
//             if(xSemaphoreTake(pca_Task_Core_0_Semaphore, portMAX_DELAY) && xSemaphoreTake(pca_Task_Core_1_Semaphore , portMAX_DELAY )) {
//                 createPCATasks(i);
//             }
//         }
//         xSemaphoreGive(execute_PCA_Grp_4_Semaphore);
//         if(xSemaphoreTake(pca_Task_Core_0_Semaphore, portMAX_DELAY)  && xSemaphoreTake(pca_Task_Core_1_Semaphore , portMAX_DELAY )) {
//             xTaskCreate(execute_Eigen_Faces_Projection, "Eigenfaces", STACK_SIZE, NULL, BASE_PRIORITY, NULL);
//             xSemaphoreGive(execute_Eigen_Faces_Task_Semaphore);
//             xSemaphoreGive(pca_Task_Core_0_Semaphore);
//             xSemaphoreGive(pca_Task_Core_1_Semaphore);
//         }
//     }
// }

// void execute_Eigen_Faces_Projection(void *params) {
//     if (xSemaphoreTake(execute_Eigen_Faces_Task_Semaphore, portMAX_DELAY)&&  xSemaphoreTake(pca_Task_Core_0_Semaphore,portMAX_DELAY) &&xSemaphoreTake(pca_Task_Core_1_Semaphore,portMAX_DELAY)) {
//         float min_distance = 14.0;
//         int class_label = -1;
//         // Iterate through each class centroid and calculateEuclidean distance to the output vector
//         for (int i = 0; i < 4; i++) {
//           float distance = compute_Euclidean_Distance(PCA_Output_Vector,class_centroids[i], NUM_COMPONENTS);
          
//           if (distance <= min_distance) {
//               min_distance = distance;
//               class_label = i;
//           }
//         }
//         end_Face_Recognition_time = esp_timer_get_time(); // Capture end time
//         Serial.print("Execution Time (ms): ");
//         Serial.println((end_Face_Recognition_time - start_Face_Recognition_time) / 1000); // Print execution time in milliseconds
//          if (class_label != -1) {
//             blink_TWICE_LED(); // Blink twice for Face Recognition
//         } else {
//             blink_ONCE_LED(); // Blink once for Face Recognition
//         }
//         Serial.print("Predicted class :");
//         Serial.println(class_label);
//         Serial.println();
//         Serial.print("Mind_Distance :");
//         Serial.println(min_distance);
//         sendResultsToServer(class_label, min_distance);
//         reset_PCA_Output_Vector();
//         xSemaphoreGive(image_Capture_Semaphore);  // Allow frame capturing to resume after processing
//         vTaskDelete(NULL);
//     }
// }


// void blink_TWICE_LED(){
//   neopixelWrite(LED_BUILTIN,RGB_BRIGHTNESS,0,0); // Red
//   delay(400);
//   neopixelWrite(LED_BUILTIN,0,0,0); // Off / black
//   delay(400);
//   neopixelWrite(LED_BUILTIN,RGB_BRIGHTNESS,0,0); // Red
//   delay(400);
//   neopixelWrite(LED_BUILTIN,0,0,0); // Off / black
//   delay(400);

// }
// void blink_ONCE_LED(){
//   neopixelWrite(LED_BUILTIN,RGB_BRIGHTNESS,0,0); // Red
//   delay(400);
//   neopixelWrite(LED_BUILTIN,0,0,0); // Off / black
//   delay(400);

// }

// float compute_Euclidean_Distance(const float* a, const float* b, int length) {
//     float distance = 0;
//     for (int i = 0; i < length; i++) {
//         float diff = a[i] - b[i];
//         distance += (diff * diff);
//     }
//     return sqrt(distance);
// }

// // Function to crop the camera face image
// void crop_CAM_Face_IMG(uint8_t *src, uint8_t *dest, int src_width, int crop_x, int crop_y, int crop_width, int crop_height) {
//     for (int y = 0; y < crop_height; y++) {
//         memcpy(dest + y * crop_width * 2, src + ((crop_y + y) * src_width + crop_x) * 2, crop_width * 2);
//     }
// }

// // Function to convert RGB565 to grayscale
// void convert_RGB565_to_Gray_Scale(uint8_t *src, uint8_t *dest, int width, int height) {
//     for (int i = 0; i < width * height; i++) {
//         uint16_t pixel = (src[2 * i] << 8) | src[2 * i + 1];
//         uint8_t r = (pixel >> 11) & 0x1F;
//         uint8_t g = (pixel >> 5) & 0x3F;
//         uint8_t b = pixel & 0x1F;
//         // Convert to grayscale using luminosity method
//         uint8_t gray = (uint8_t)((r * 0.299 * 255.0 / 31) + (g * 0.587 * 255.0 / 63) + (b * 0.114 * 255.0 / 31));
//         dest[i] = gray;
//     }
// }

// // Function to resize the camera face image
// void resize_CAM_Face_IMG(uint8_t *src, int src_width, int src_height, uint8_t *dest, int dest_width, int dest_height) {
//     float x_scale = src_width / (float) dest_width;
//     float y_scale = src_height / (float) dest_height;

//     for (int y = 0; y < dest_height; y++) {
//         for (int x = 0; x < dest_width; x++) {
//             int src_x = (int)(x * x_scale);
//             int src_y = (int)(y * y_scale);
//             src_x = src_x < src_width ? src_x : src_width - 1;   // Clamping to ensure we don't go out of bounds
//             src_y = src_y < src_height ? src_y : src_height - 1; // Clamping to ensure we don't go out of bounds
//             dest[y * dest_width + x] = src[src_y * src_width + src_x];
//         }
//     }
// }

// // Function to process the camera face image
// void process_CAM_Face_IMG(camera_fb_t *fb, const dl::detect::result_t& result) {
//     // sendImageToServer(fb->buf, fb->len, "initial_image", -1, -1);
//     int x = result.box[0];
//     int y = result.box[1];
//     int w = result.box[2] - result.box[0];
//     int h = result.box[3] - result.box[1];

//     uint8_t *cropped_img = (uint8_t *)malloc(w * h * 2);
//     if (!cropped_img) {
//         Serial.println("Failed to allocate memory for cropped image");
//         return;
//     }
//     crop_CAM_Face_IMG(fb->buf, cropped_img, fb->width, x, y, w, h);
//     // sendImageToServer(cropped_img, w * h * 2, "cropped_image", -1, -1);

//     uint8_t *gray_img = (uint8_t *)malloc(w * h);
//     if (!gray_img) {
//         Serial.println("Failed to allocate memory for grayscale image");
//         free(cropped_img);
//         return;
//     }
//     convert_RGB565_to_Gray_Scale(cropped_img, gray_img, w, h);

//     uint8_t *resized_img = (uint8_t *)malloc(64 * 64);
//     if (!resized_img) {
//         Serial.println("Failed to allocate memory for resized image");
//         free(cropped_img);
//         free(gray_img);
//         return;
//     }
//     resize_CAM_Face_IMG(gray_img, w, h, resized_img, 64, 64);
//     // sendImageToServer(resized_img, 64 * 64, "resized_image", -1, -1);

//     // Normalize the image data
//     for (int i = 0; i < 64 * 64; i++) {
//         Normalized_Detected_Face_IMG[i] = resized_img[i] / 255.0f;
//     }
//     centering_CAM_Face_Data(Normalized_Detected_Face_IMG, Centered_CAM_Face_Data);
//     delay(5);
//     if (uxSemaphoreGetCount(centering_CAM_Face_DATA_Semaphore) == 2) {
//         xSemaphoreTake(centering_CAM_Face_DATA_Semaphore, portMAX_DELAY);
//         xSemaphoreTake(centering_CAM_Face_DATA_Semaphore , portMAX_DELAY);
//         xSemaphoreGive(execute_PCA_Grp_1_Semaphore);
//         execute_PCA_Task_Grp_1();
//     }

//     // Cleanup
//     free(cropped_img);
//     free(gray_img);
//     free(resized_img);
// }

// void reset_PCA_Output_Vector() {
//     for (int i = 0; i < NUM_COMPONENTS; i++) {
//         PCA_Output_Vector[i] = 0.0f;
//     }
// }

// void detect_Face(camera_fb_t *fb) {
//     if (!fb) return;
//     std::list<dl::detect::result_t> results;

//     if (TWO_STAGE) {
//         auto candidates = stage1.infer((uint16_t *)fb->buf, {fb->height, fb->width, 3});
//         results = stage2.infer((uint16_t *)fb->buf, {fb->height, fb->width, 3}, candidates);
//     } else {
//         results = stage1.infer((uint16_t *)fb->buf, {fb->height, fb->width, 3});
//     }

//     if (!results.empty()) {
//         Serial.println("Face Detected Successfully !!!!!");
//         start_Face_Recognition_time = esp_timer_get_time(); // Capture start time
//         for (const auto& result : results) {
//             process_CAM_Face_IMG(fb, result);
//         }
//     } else {
//         Serial.println("no face detected !");
//         xSemaphoreGive(image_Capture_Semaphore);
//     }
//     esp_camera_fb_return(fb);
// }
// void sendResultsToServer(int class_label, float min_distance) {
//     esp_http_client_config_t config = {
//         .url = SERVER_URL,
//         .method = HTTP_METHOD_POST
//     };
//     esp_http_client_handle_t client = esp_http_client_init(&config);
//     if (client == NULL) {
//         Serial.println("Failed to initialize HTTP client");
//         return;
//     }
//     Serial.println("HTTP client initialized");

//     char jsonPayload[100];
//     snprintf(jsonPayload, sizeof(jsonPayload), "{\"class_label\": %d, \"min_distance\": %.2f}", class_label, min_distance);
//     Serial.print("JSON Payload: ");
//     Serial.println(jsonPayload);

//     esp_http_client_set_header(client, "Content-Type", "application/json");
//     esp_http_client_set_post_field(client, jsonPayload, strlen(jsonPayload));

//     esp_err_t err = esp_http_client_perform(client);
//     if (err == ESP_OK) {
//         Serial.printf("HTTP POST Status = %d, content_length = %d\n",
//                       esp_http_client_get_status_code(client),
//                       esp_http_client_get_content_length(client));
//     } else {
//         Serial.printf("HTTP POST request failed: %s\n", esp_err_to_name(err));
//     }

//     esp_http_client_cleanup(client);
// }




// // void sendImageToServer(const uint8_t* img, size_t len, const char* description, int class_label, float min_distance) {
// //     esp_http_client_config_t config = {
// //         .url = SERVER_URL,
// //         .method = HTTP_METHOD_POST
// //     };
// //     esp_http_client_handle_t client = esp_http_client_init(&config);

// //     char boundary[] = "----ESP32Boundary";
// //     char header[512];
// //     snprintf(header, sizeof(header),
// //              "--%s\r\nContent-Disposition: form-data; name=\"metadata\"\r\n\r\n{\"description\": \"%s\", \"class_label\": %d, \"min_distance\": %.2f}\r\n--%s\r\nContent-Disposition: form-data; name=\"file\"; filename=\"%s.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n",
// //              boundary, description, class_label, min_distance, boundary, description);

// //     // Calculate the length of the base64 encoded image
// //     size_t base64Len = Base64.encodedLength(len);
// //     char *base64Image = (char *)malloc(base64Len + 1);  // Allocate memory for base64 image
// //     if (base64Image) {
// //         // Encode the image to base64
// //         Base64.encode(base64Image, (char *)img, len);

// //         esp_http_client_set_header(client, "Content-Type", "multipart/form-data; boundary=----ESP32Boundary");

// //         // Open HTTP connection
// //         esp_err_t err = esp_http_client_open(client, strlen(header) + base64Len + strlen(boundary) + 8);
// //         if (err == ESP_OK) {
// //             esp_http_client_write(client, header, strlen(header));
// //             esp_http_client_write(client, base64Image, base64Len);
// //             esp_http_client_write(client, "\r\n--", 4);
// //             esp_http_client_write(client, boundary, strlen(boundary));
// //             esp_http_client_write(client, "--\r\n", 4);
// //         }

// //         esp_http_client_perform(client);
// //         free(base64Image);
// //     }

// //     esp_http_client_cleanup(client);
// // }



// void setup() {
//     Serial.begin(115200);
//     while (!Serial);
//     // Camera configurations
//     // pinMode(LED_BUILTIN , OUTPUT);
//     camera_config_t config;
//     config.ledc_channel = LEDC_CHANNEL_0;
//     config.ledc_timer = LEDC_TIMER_0;
//     config.pin_d0 = Y2_GPIO_NUM;
//     config.pin_d1 = Y3_GPIO_NUM;
//     config.pin_d2 = Y4_GPIO_NUM;
//     config.pin_d3 = Y5_GPIO_NUM;
//     config.pin_d4 = Y6_GPIO_NUM;
//     config.pin_d5 = Y7_GPIO_NUM;
//     config.pin_d6 = Y8_GPIO_NUM;
//     config.pin_d7 = Y9_GPIO_NUM;
//     config.pin_xclk = XCLK_GPIO_NUM;
//     config.pin_pclk = PCLK_GPIO_NUM;
//     config.pin_vsync = VSYNC_GPIO_NUM;
//     config.pin_href = HREF_GPIO_NUM;
//     config.pin_sccb_sda = SIOD_GPIO_NUM;
//     config.pin_sccb_scl = SIOC_GPIO_NUM;
//     config.pin_pwdn = PWDN_GPIO_NUM;
//     config.pin_reset = RESET_GPIO_NUM;
//     config.xclk_freq_hz = 2000000;
//     config.frame_size = FRAMESIZE_240X240;
//     config.pixel_format = PIXFORMAT_RGB565; // for face detection/recognition
//     config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
//     config.fb_location = CAMERA_FB_IN_PSRAM;
//     config.jpeg_quality = 12;
//     config.fb_count = 1;

//     if(config.pixel_format == PIXFORMAT_JPEG) {
//         if(psramFound()) {
//             config.jpeg_quality = 10;
//             config.fb_count = 2;
//             config.grab_mode = CAMERA_GRAB_LATEST;
//         } else {
//             // Limit the frame size when PSRAM is not available
//             config.frame_size = FRAMESIZE_SVGA;
//             config.fb_location = CAMERA_FB_IN_DRAM;
//         }
//     } else {
//         #if CONFIG_IDF_TARGET_ESP32S3
//             config.fb_count = 2;
//         #endif
//     }

//     esp_err_t err = esp_camera_init(&config);
//     if (err != ESP_OK) {
//         Serial.printf("Camera init failed with error 0x%x", err);
//         return;
//     }

//     execute_PCA_Grp_1_Semaphore = xSemaphoreCreateBinary();
//     execute_PCA_Grp_2_Semaphore = xSemaphoreCreateBinary();
//     execute_PCA_Grp_3_Semaphore = xSemaphoreCreateBinary();
//     execute_PCA_Grp_4_Semaphore = xSemaphoreCreateBinary();
//     execute_PCA_Grp_5_Semaphore = xSemaphoreCreateBinary();
//     execute_Eigen_Faces_Task_Semaphore = xSemaphoreCreateBinary();
//     sync_PCA_Loop_Semaphore = xSemaphoreCreateMutex();
//     pca_Task_Core_0_Semaphore = xSemaphoreCreateBinary();
//     pca_Task_Core_1_Semaphore = xSemaphoreCreateBinary();
//     centering_CAM_Face_DATA_Semaphore = xSemaphoreCreateCounting(2, 0);
//     image_Capture_Semaphore = xSemaphoreCreateBinary();

//     if (!sync_PCA_Loop_Semaphore || !pca_Task_Core_1_Semaphore || !pca_Task_Core_0_Semaphore || !image_Capture_Semaphore || !Centered_CAM_Face_Data || !execute_PCA_Grp_1_Semaphore || !execute_PCA_Grp_2_Semaphore || !execute_PCA_Grp_3_Semaphore || !execute_PCA_Grp_4_Semaphore || !execute_PCA_Grp_5_Semaphore || !execute_Eigen_Faces_Task_Semaphore || !centering_CAM_Face_DATA_Semaphore) {
//         Serial.println("Failed to create task Semaphore or MUTEX!");
//         return;
//     }
//     xSemaphoreGive(image_Capture_Semaphore);

//     WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
//     while (WiFi.status() != WL_CONNECTED) {
//         delay(500);
//         Serial.print(".");
//     }
//     Serial.println("");
//     Serial.println("WiFi connected");
//     Serial.println("IP address: ");
//     Serial.println(WiFi.localIP());
// }



// void loop() {
//     if (xSemaphoreTake(image_Capture_Semaphore, portMAX_DELAY)) {
//         camera_fb_t *fb = esp_camera_fb_get();
//         if (!fb) {
//             Serial.println("Camera capture failed");
//         } else {
//             detect_Face(fb);
//         }
//     }
//     // delay(1000);
// }







// Camera Model
// #define CAMERA_MODEL_XIAO_ESP32S3 // Has PSRAM

// // Libraries included
// #include "freertos/FreeRTOS.h"
// #include "freertos/task.h"
// #include "freertos/semphr.h"
// #include "pca_mean_vector.h"
// #include <string.h>
// #include "pca_transformation_matrix.h"
// #include "class_centroids.h"
// #include "esp_timer.h"
// #include "esp_camera.h"
// #include "fb_gfx.h"
// #include "camera_pins.h"
// #include "img_converters.h"
// #include "human_face_detect_msr01.hpp"
// #include "human_face_detect_mnp01.hpp"
// // #include "bilieEillish_p.h"

// // Predefined Attributes
// #define NUM_TASKS 20
// #define BASE_PRIORITY 1
// #define NUM_COMPONENTS 20
// #define VECTOR_SIZE 4096
// #define STACK_SIZE 10000
// #define HALF_VECTOR_SIZE 2048
// #define FIRST_HALF_COMPONENTS 12
// #define SECOND_HALF_COMPONENTS 13
// #define FACE_COLOR_YELLOW 0xFFFF00
// #define TWO_STAGE 1
// #define Normalized_Detected_Face_IMG_SIZE 4096
// #define RGB_BRIGHTNESS 255

// uint64_t start_Face_Recognition_time = 0;
// uint64_t end_Face_Recognition_time = 0;

// HumanFaceDetectMSR01 stage1(0.1F, 0.5F, 10, 0.2F);
// HumanFaceDetectMNP01 stage2(0.5F, 0.3F, 5);

// int Centering_Data_Counter = 0;

// // Semaphores
// SemaphoreHandle_t sync_PCA_Loop_Semaphore;
// SemaphoreHandle_t pca_Task_Core_0_Semaphore;
// SemaphoreHandle_t pca_Task_Core_1_Semaphore;
// SemaphoreHandle_t centering_CAM_Face_DATA_Semaphore;
// SemaphoreHandle_t execute_PCA_Grp_1_Semaphore;
// SemaphoreHandle_t execute_PCA_Grp_2_Semaphore;
// SemaphoreHandle_t execute_PCA_Grp_3_Semaphore;
// SemaphoreHandle_t execute_PCA_Grp_4_Semaphore;
// SemaphoreHandle_t execute_PCA_Grp_5_Semaphore;
// SemaphoreHandle_t execute_Eigen_Faces_Task_Semaphore;
// SemaphoreHandle_t image_Capture_Semaphore;

// // DataStructures
// float PCA_Output_Vector[NUM_COMPONENTS];
// float Centered_CAM_Face_Data[VECTOR_SIZE];
// float Normalized_Detected_Face_IMG[Normalized_Detected_Face_IMG_SIZE];

// // Methods and Functions Definitions
// void apply_PCA(int idx, int vectorStart, int vectorEnd);
// void pca_Task(void *params);
// void execute_Eigen_Faces_Projection(void *params);
// float compute_Euclidean_Distance(const float* a, const float* b, int length);
// void centering_CAM_Face_Data(const float *Normalized_Detected_Face_IMG, float *centered);
// void crop_CAM_Face_IMG(uint8_t *src, uint8_t *dest, int src_width, int crop_x, int crop_y, int crop_width, int crop_height);
// void convert_RGB565_to_Gray_Scale(uint8_t *src, uint8_t *dest, int width, int height);
// void resize_CAM_Face_IMG(uint8_t *src, int src_width, int src_height, uint8_t *dest, int dest_width, int dest_height);
// void process_CAM_Face_IMG(camera_fb_t *fb, const dl::detect::result_t& result);
// void detect_Face(camera_fb_t *fb);
// void execute_PCA_Task_Grp_1();
// void execute_PCA_Task_Grp_2();
// void execute_PCA_Task_Grp_3();
// void execute_PCA_Task_Grp_4();
// void execute_PCA_Task_Grp_5();
// void apply_Centering_CAM_Face_Data(void *params);
// void pcaTask_Core_0(void *params);
// void pcaTask_Core_1(void *params);

// // Methods Implementations
// void centering_CAM_Face_Data(const float *Normalized_Detected_Face_IMG, float* Centered_CAM_Face_Data) {
//     static int params_core_0[2];
//     params_core_0[0] = 0;
//     params_core_0[1] = HALF_VECTOR_SIZE;

//     static int params_core_1[2];
//     params_core_1[0] = HALF_VECTOR_SIZE;
//     params_core_1[1] = VECTOR_SIZE;

//     xTaskCreatePinnedToCore(apply_Centering_CAM_Face_Data, "Centering_CAM_Face_Data_Core_0", STACK_SIZE, params_core_0, BASE_PRIORITY, NULL, 0);
//     xTaskCreatePinnedToCore(apply_Centering_CAM_Face_Data, "Centering_CAM_Face_Data_Core_1", STACK_SIZE, params_core_1, BASE_PRIORITY, NULL, 1);
// }

// void apply_Centering_CAM_Face_Data(void *params) {
//     int VECTOR_START = ((int*)params)[0];
//     int VECTOR_END = ((int*)params)[1];
//     for (int i = VECTOR_START; i < VECTOR_END; i += 64) {
//         for (int j = 0; j < 64; j++) {
//             Centered_CAM_Face_Data[i + j] = Normalized_Detected_Face_IMG[i + j] - pgm_read_float_near(&pca_mean_vector[i + j]);
//         }
//     }
//     xSemaphoreGive(centering_CAM_Face_DATA_Semaphore);
//     vTaskDelete(NULL);
// }

// void applyPCA(int i, int vectorStart, int vectorEnd) {
//     float sum = 0;
//     for (int j = vectorStart; j < vectorEnd; j += 64) {
//         for (int k = 0; k < 64; k++) {
//             sum += Centered_CAM_Face_Data[j + k] * pgm_read_float_near(&pca_transformation_matrix[i][j + k]);
//         }
//     }
//     PCA_Output_Vector[i] += sum;
// }

// void pcaTask_Core_0(void *params) {
//     int idx = (int)params;
//     applyPCA(idx, 0, HALF_VECTOR_SIZE);
//     delay(5);
//     xSemaphoreGive(pca_Task_Core_0_Semaphore);
//     vTaskDelete(NULL);
// }

// void pcaTask_Core_1(void *params) {
//     int idx = (int)params;
//     applyPCA(idx, HALF_VECTOR_SIZE, VECTOR_SIZE);
//     delay(5);
//     xSemaphoreGive(pca_Task_Core_1_Semaphore);
//     vTaskDelete(NULL);
// }

// void createPCATasks(int idx) {
//     xTaskCreatePinnedToCore(pcaTask_Core_0, "PCA_Task_Core0", STACK_SIZE, (void *)idx, BASE_PRIORITY, NULL, 0);
//     xTaskCreatePinnedToCore(pcaTask_Core_1, "PCA_Task_Core1", STACK_SIZE, (void *)idx, BASE_PRIORITY, NULL, 1);
// }

// void execute_PCA_Task_Grp_1() {
//     if (xSemaphoreTake(execute_PCA_Grp_1_Semaphore, portMAX_DELAY)) {
//         createPCATasks(0);
//         for (int i = 1; i < 5; i++) {
//             if(xSemaphoreTake(pca_Task_Core_0_Semaphore, portMAX_DELAY)  && xSemaphoreTake(pca_Task_Core_1_Semaphore , portMAX_DELAY )) {
//                 createPCATasks(i);
//             }
//         }
//         xSemaphoreGive(execute_PCA_Grp_2_Semaphore);
//         execute_PCA_Task_Grp_2();
//     }
// }

// void execute_PCA_Task_Grp_2() {
//     if (xSemaphoreTake(execute_PCA_Grp_2_Semaphore, portMAX_DELAY)) {
//         for (int i = 5; i < 10; i++) {
//             if(xSemaphoreTake(pca_Task_Core_0_Semaphore, portMAX_DELAY) && xSemaphoreTake(pca_Task_Core_1_Semaphore , portMAX_DELAY )) {
//                 createPCATasks(i);
//             }
//         }
//         xSemaphoreGive(execute_PCA_Grp_3_Semaphore);
//         xSemaphoreGive(execute_PCA_Grp_2_Semaphore);
//         execute_PCA_Task_Grp_3();
//     }
// }

// void execute_PCA_Task_Grp_3() {
//     if (xSemaphoreTake(execute_PCA_Grp_3_Semaphore, portMAX_DELAY)) {
//         for (int i = 10; i < 15; i++) {
//             if(xSemaphoreTake(pca_Task_Core_0_Semaphore, portMAX_DELAY) && xSemaphoreTake(pca_Task_Core_1_Semaphore , portMAX_DELAY )) {
//                 createPCATasks(i);
//             }
//         }
//         xSemaphoreGive(execute_PCA_Grp_4_Semaphore);
//         xSemaphoreGive(execute_PCA_Grp_3_Semaphore);
//         execute_PCA_Task_Grp_4();
//     }
// }

// void execute_PCA_Task_Grp_4() {
//     if (xSemaphoreTake(execute_PCA_Grp_4_Semaphore, portMAX_DELAY)) {
//         for (int i = 15; i < 20; i++) {
//             if(xSemaphoreTake(pca_Task_Core_0_Semaphore, portMAX_DELAY) && xSemaphoreTake(pca_Task_Core_1_Semaphore , portMAX_DELAY )) {
//                 createPCATasks(i);
//             }
//         }
//         xSemaphoreGive(execute_PCA_Grp_4_Semaphore);
//         if(xSemaphoreTake(pca_Task_Core_0_Semaphore, portMAX_DELAY)  && xSemaphoreTake(pca_Task_Core_1_Semaphore , portMAX_DELAY )) {
//             xTaskCreate(execute_Eigen_Faces_Projection, "Eigenfaces", STACK_SIZE, NULL, BASE_PRIORITY, NULL);
//             xSemaphoreGive(execute_Eigen_Faces_Task_Semaphore);
//             xSemaphoreGive(pca_Task_Core_0_Semaphore);
//             xSemaphoreGive(pca_Task_Core_1_Semaphore);
//         }
//     }
// }

// void execute_Eigen_Faces_Projection(void *params) {
//     if (xSemaphoreTake(execute_Eigen_Faces_Task_Semaphore, portMAX_DELAY)&&  xSemaphoreTake(pca_Task_Core_0_Semaphore,portMAX_DELAY) &&xSemaphoreTake(pca_Task_Core_1_Semaphore,portMAX_DELAY)) {
//         float min_distance = 14.0;
//         int class_label = -1;
//         // Iterate through each class centroid and calculateEuclidean distance to the output vector
//         for (int i = 0; i < 4; i++) {
//           float distance = compute_Euclidean_Distance(PCA_Output_Vector,class_centroids[i], NUM_COMPONENTS);
          
//           if (distance <= min_distance) {
//               min_distance = distance;
//               class_label = i;
//           }
//         }
//         end_Face_Recognition_time = esp_timer_get_time(); // Capture end time
//         Serial.print("Execution Time (ms): ");
//         Serial.println((end_Face_Recognition_time - start_Face_Recognition_time) / 1000); // Print execution time in milliseconds
//          if (class_label != -1) {
//             blink_TWICE_LED(); // Blink twice for Face Recognition
//         } else {
//             blink_ONCE_LED(); // Blink once for Face Recognition
//         }
//         Serial.print("Predicted class :");
//         Serial.println(class_label);
//         Serial.println();
//         Serial.print("Mind_Distance :");
//         Serial.println(min_distance);
//         reset_PCA_Output_Vector();
//         xSemaphoreGive(image_Capture_Semaphore);  // Allow frame capturing to resume after processing
//         vTaskDelete(NULL);
//     }
// }

// void blink_TWICE_LED(){
//   neopixelWrite(LED_BUILTIN,RGB_BRIGHTNESS,0,0); // Red
//   delay(400);
//   neopixelWrite(LED_BUILTIN,0,0,0); // Off / black
//   delay(400);
//   neopixelWrite(LED_BUILTIN,RGB_BRIGHTNESS,0,0); // Red
//   delay(400);
//   neopixelWrite(LED_BUILTIN,0,0,0); // Off / black
//   delay(400);

// }
// void blink_ONCE_LED(){
//   neopixelWrite(LED_BUILTIN,RGB_BRIGHTNESS,0,0); // Red
//   delay(400);
//   neopixelWrite(LED_BUILTIN,0,0,0); // Off / black
//   delay(400);

// }

// float compute_Euclidean_Distance(const float* a, const float* b, int length) {
//     float distance = 0;
//     for (int i = 0; i < length; i++) {
//         float diff = a[i] - b[i];
//         distance += (diff * diff);
//     }
//     return sqrt(distance);
// }

// // Function to crop the camera face image
// void crop_CAM_Face_IMG(uint8_t *src, uint8_t *dest, int src_width, int crop_x, int crop_y, int crop_width, int crop_height) {
//     for (int y = 0; y < crop_height; y++) {
//         memcpy(dest + y * crop_width * 2, src + ((crop_y + y) * src_width + crop_x) * 2, crop_width * 2);
//     }
// }

// // Function to convert RGB565 to grayscale
// void convert_RGB565_to_Gray_Scale(uint8_t *src, uint8_t *dest, int width, int height) {
//     for (int i = 0; i < width * height; i++) {
//         uint16_t pixel = (src[2 * i] << 8) | src[2 * i + 1];
//         uint8_t r = (pixel >> 11) & 0x1F;
//         uint8_t g = (pixel >> 5) & 0x3F;
//         uint8_t b = pixel & 0x1F;
//         // Convert to grayscale using luminosity method
//         uint8_t gray = (uint8_t)((r * 0.299 * 255.0 / 31) + (g * 0.587 * 255.0 / 63) + (b * 0.114 * 255.0 / 31));
//         dest[i] = gray;
//     }
// }

// // Function to resize the camera face image
// void resize_CAM_Face_IMG(uint8_t *src, int src_width, int src_height, uint8_t *dest, int dest_width, int dest_height) {
//     float x_scale = src_width / (float) dest_width;
//     float y_scale = src_height / (float) dest_height;

//     for (int y = 0; y < dest_height; y++) {
//         for (int x = 0; x < dest_width; x++) {
//             int src_x = (int)(x * x_scale);
//             int src_y = (int)(y * y_scale);
//             src_x = src_x < src_width ? src_x : src_width - 1;   // Clamping to ensure we don't go out of bounds
//             src_y = src_y < src_height ? src_y : src_height - 1; // Clamping to ensure we don't go out of bounds
//             dest[y * dest_width + x] = src[src_y * src_width + src_x];
//         }
//     }
// }

// // Function to process the camera face image
// void process_CAM_Face_IMG(camera_fb_t *fb, const dl::detect::result_t& result) {
//     int x = result.box[0];
//     int y = result.box[1];
//     int w = result.box[2] - result.box[0];
//     int h = result.box[3] - result.box[1];

//     uint8_t *cropped_img = (uint8_t *)malloc(w * h * 2);
//     if (!cropped_img) {
//         Serial.println("Failed to allocate memory for cropped image");
//         return;
//     }
//     crop_CAM_Face_IMG(fb->buf, cropped_img, fb->width, x, y, w, h);

//     uint8_t *gray_img = (uint8_t *)malloc(w * h);
//     if (!gray_img) {
//         Serial.println("Failed to allocate memory for grayscale image");
//         free(cropped_img);
//         return;
//     }
//     convert_RGB565_to_Gray_Scale(cropped_img, gray_img, w, h);

//     uint8_t *resized_img = (uint8_t *)malloc(64 * 64);
//     if (!resized_img) {
//         Serial.println("Failed to allocate memory for resized image");
//         free(cropped_img);
//         free(gray_img);
//         return;
//     }
//     resize_CAM_Face_IMG(gray_img, w, h, resized_img, 64, 64);

//     // Normalize the image data
//     for (int i = 0; i < 64 * 64; i++) {
//         Normalized_Detected_Face_IMG[i] = resized_img[i] / 255.0f;
//     }
//     centering_CAM_Face_Data(Normalized_Detected_Face_IMG, Centered_CAM_Face_Data);
//     delay(5);
//     if (uxSemaphoreGetCount(centering_CAM_Face_DATA_Semaphore) == 2) {
//         xSemaphoreTake(centering_CAM_Face_DATA_Semaphore, portMAX_DELAY);
//         xSemaphoreTake(centering_CAM_Face_DATA_Semaphore , portMAX_DELAY);
//         xSemaphoreGive(execute_PCA_Grp_1_Semaphore);
//         execute_PCA_Task_Grp_1();
//     }

//     // Cleanup
//     free(cropped_img);
//     free(gray_img);
//     free(resized_img);
// }

// void reset_PCA_Output_Vector() {
//     for (int i = 0; i < NUM_COMPONENTS; i++) {
//         PCA_Output_Vector[i] = 0.0f;
//     }
// }

// void detect_Face(camera_fb_t *fb) {
//     if (!fb) return;
//     std::list<dl::detect::result_t> results;

//     if (TWO_STAGE) {
//         auto candidates = stage1.infer((uint16_t *)fb->buf, {fb->height, fb->width, 3});
//         results = stage2.infer((uint16_t *)fb->buf, {fb->height, fb->width, 3}, candidates);
//     } else {
//         results = stage1.infer((uint16_t *)fb->buf, {fb->height, fb->width, 3});
//     }

//     if (!results.empty()) {
//         Serial.println("Face Detected Successfully !!!!!");
//         start_Face_Recognition_time = esp_timer_get_time(); // Capture start time
//         for (const auto& result : results) {
//             process_CAM_Face_IMG(fb, result);
//         }
//     } else {
//         Serial.println("no face detected !");
//         xSemaphoreGive(image_Capture_Semaphore);
//     }
//     esp_camera_fb_return(fb);
// }




// void setup() {
//     Serial.begin(115200);
//     while (!Serial);
//     // Camera configurations
//     // pinMode(LED_BUILTIN , OUTPUT);
//     camera_config_t config;
//     config.ledc_channel = LEDC_CHANNEL_0;
//     config.ledc_timer = LEDC_TIMER_0;
//     config.pin_d0 = Y2_GPIO_NUM;
//     config.pin_d1 = Y3_GPIO_NUM;
//     config.pin_d2 = Y4_GPIO_NUM;
//     config.pin_d3 = Y5_GPIO_NUM;
//     config.pin_d4 = Y6_GPIO_NUM;
//     config.pin_d5 = Y7_GPIO_NUM;
//     config.pin_d6 = Y8_GPIO_NUM;
//     config.pin_d7 = Y9_GPIO_NUM;
//     config.pin_xclk = XCLK_GPIO_NUM;
//     config.pin_pclk = PCLK_GPIO_NUM;
//     config.pin_vsync = VSYNC_GPIO_NUM;
//     config.pin_href = HREF_GPIO_NUM;
//     config.pin_sccb_sda = SIOD_GPIO_NUM;
//     config.pin_sccb_scl = SIOC_GPIO_NUM;
//     config.pin_pwdn = PWDN_GPIO_NUM;
//     config.pin_reset = RESET_GPIO_NUM;
//     config.xclk_freq_hz = 2000000;
//     config.frame_size = FRAMESIZE_240X240;
//     config.pixel_format = PIXFORMAT_RGB565; // for face detection/recognition
//     config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
//     config.fb_location = CAMERA_FB_IN_PSRAM;
//     config.jpeg_quality = 12;
//     config.fb_count = 1;

//     if(config.pixel_format == PIXFORMAT_JPEG) {
//         if(psramFound()) {
//             config.jpeg_quality = 10;
//             config.fb_count = 2;
//             config.grab_mode = CAMERA_GRAB_LATEST;
//         } else {
//             // Limit the frame size when PSRAM is not available
//             config.frame_size = FRAMESIZE_SVGA;
//             config.fb_location = CAMERA_FB_IN_DRAM;
//         }
//     } else {
//         #if CONFIG_IDF_TARGET_ESP32S3
//             config.fb_count = 2;
//         #endif
//     }

//     esp_err_t err = esp_camera_init(&config);
//     if (err != ESP_OK) {
//         Serial.printf("Camera init failed with error 0x%x", err);
//         return;
//     }

//     execute_PCA_Grp_1_Semaphore = xSemaphoreCreateBinary();
//     execute_PCA_Grp_2_Semaphore = xSemaphoreCreateBinary();
//     execute_PCA_Grp_3_Semaphore = xSemaphoreCreateBinary();
//     execute_PCA_Grp_4_Semaphore = xSemaphoreCreateBinary();
//     execute_PCA_Grp_5_Semaphore = xSemaphoreCreateBinary();
//     execute_Eigen_Faces_Task_Semaphore = xSemaphoreCreateBinary();
//     sync_PCA_Loop_Semaphore = xSemaphoreCreateMutex();
//     pca_Task_Core_0_Semaphore = xSemaphoreCreateBinary();
//     pca_Task_Core_1_Semaphore = xSemaphoreCreateBinary();
//     centering_CAM_Face_DATA_Semaphore = xSemaphoreCreateCounting(2, 0);
//     image_Capture_Semaphore = xSemaphoreCreateBinary();

//     if (!sync_PCA_Loop_Semaphore || !pca_Task_Core_1_Semaphore || !pca_Task_Core_0_Semaphore || !image_Capture_Semaphore || !Centered_CAM_Face_Data || !execute_PCA_Grp_1_Semaphore || !execute_PCA_Grp_2_Semaphore || !execute_PCA_Grp_3_Semaphore || !execute_PCA_Grp_4_Semaphore || !execute_PCA_Grp_5_Semaphore || !execute_Eigen_Faces_Task_Semaphore || !centering_CAM_Face_DATA_Semaphore) {
//         Serial.println("Failed to create task Semaphore or MUTEX!");
//         return;
//     }
//     xSemaphoreGive(image_Capture_Semaphore);
// }

// void loop() {
//     if (xSemaphoreTake(image_Capture_Semaphore, portMAX_DELAY)) {
//         camera_fb_t *fb = esp_camera_fb_get();
//         if (!fb) {
//             Serial.println("Camera capture failed");
//         } else {
//             detect_Face(fb);
//         }
//     }
//     // delay(1000);
// }


// The provided code is designed for a facial recognition system using an ESP32-S3 microcontroller with a camera module. Here’s a detailed analysis of its functionality and implementation:

// ### Key Components and Functional Blocks:

// 1. **Camera Initialization and Configuration**:
//    - The `setup()` function configures the camera hardware, setting various pins and parameters, such as frame size and pixel format, ensuring compatibility with face detection and recognition tasks.

// 2. **Face Detection**:
//    - The code uses two stages for face detection:
//      - **Stage 1** (`HumanFaceDetectMSR01`): Identifies potential face regions.
//      - **Stage 2** (`HumanFaceDetectMNP01`): Refines these regions to confirm the presence of faces.
//    - The `detect_Face()` function handles the detection process, leveraging these two stages to identify faces within a captured frame.

// 3. **Image Processing**:
//    - Several functions handle various image processing tasks:
//      - **Cropping** (`crop_CAM_Face_IMG()`): Extracts the detected face region from the full image.
//      - **Conversion to Grayscale** (`convert_RGB565_to_Gray_Scale()`): Converts the cropped image to grayscale, simplifying further processing.
//      - **Resizing** (`resize_CAM_Face_IMG()`): Resizes the grayscale image to a standard 64x64 size for normalization.

// 4. **Data Normalization and Centering**:
//    - The detected face image is normalized to a range of 0 to 1.
//    - **Centering** (`centering_CAM_Face_Data()` and `apply_Centering_CAM_Face_Data()`): Adjusts the normalized image data by subtracting a predefined mean vector (`pca_mean_vector`).

// 5. **Principal Component Analysis (PCA)**:
//    - PCA is applied to the centered data to reduce its dimensionality:
//      - **Task Creation** (`createPCATasks()`): Spawns tasks for PCA computation, distributed across two cores for efficiency.
//      - **PCA Execution** (`applyPCA()`): Projects the centered data onto a PCA transformation matrix (`pca_transformation_matrix`).

// 6. **Eigen Faces Projection and Classification**:
//    - **Eigen Faces Projection** (`execute_Eigen_Faces_Projection()`): Projects the PCA output vector onto a set of class centroids (`class_centroids`) and computes the Euclidean distance to classify the face.
//    - The closest centroid determines the class label (predicted face identity).

// 7. **Semaphore Management**:
//    - Semaphores are extensively used to synchronize tasks and ensure the orderly execution of the PCA process and face recognition workflow:
//      - **Binary Semaphores**: Coordinate task execution across different cores.
//      - **Counting Semaphore**: Manages the centering process, allowing two tasks to operate concurrently.

// 8. **LED Indications**:
//    - **Blinking Functions** (`blink_TWICE_LED()` and `blink_ONCE_LED()`): Provide visual feedback for face recognition status:
//      - Blink twice if a face is recognized.
//      - Blink once if no face is recognized.

// ### Workflow Overview:

// 1. **Setup Phase**:
//    - Initialize camera hardware and configuration.
//    - Create semaphores and initialize synchronization primitives.

// 2. **Main Loop**:
//    - Capture a frame from the camera.
//    - Perform face detection.
//    - If a face is detected, process the image:
//      - Crop, convert to grayscale, resize, and normalize.
//      - Center the data using predefined mean vector.
//    - Execute PCA to reduce dimensionality.
//    - Project the result onto eigen faces and classify the face.
//    - Provide visual feedback using LEDs.
//    - Reset and prepare for the next frame capture.

// ### Summary:

// The code is a well-structured implementation of a facial recognition system on an ESP32-S3 microcontroller. It utilizes a two-stage face detection approach, PCA for dimensionality reduction, and semaphores for task synchronization across dual cores. The system provides real-time face recognition with visual feedback via LED indicators.


// Yes, this project is well-suited for a bachelor's thesis on AI-based facial recognition for security and surveillance, particularly focusing on AI and security on low-power microprocessors. Here’s why it makes a compelling project:

// ### Key Strengths of the Project:

// 1. **Relevance to AI and Security**:
//    - **AI Component**: The project utilizes machine learning techniques such as PCA (Principal Component Analysis) and face detection algorithms, which are fundamental aspects of AI.
//    - **Security and Surveillance**: Facial recognition is a critical technology in modern security systems, making this project highly relevant to surveillance applications.

// 2. **Use of Low-Power Microprocessor (ESP32-S3)**:
//    - The ESP32-S3 is known for its low power consumption, making it ideal for battery-operated or energy-efficient devices, which is a significant advantage in security and surveillance applications where power efficiency is crucial.

// 3. **Practical Application**:
//    - The implementation demonstrates a real-world application of AI on a microcontroller, showcasing the feasibility of deploying advanced AI algorithms on resource-constrained devices.

// 4. **Comprehensive Technical Scope**:
//    - **Camera Integration**: The project involves setting up and configuring camera hardware, which is fundamental for image capture in surveillance systems.
//    - **Image Processing**: Includes critical steps like cropping, grayscale conversion, and resizing, demonstrating an understanding of image preprocessing techniques.
//    - **Dimensionality Reduction (PCA)**: Highlights knowledge of statistical methods to handle high-dimensional data efficiently.
//    - **Task Synchronization**: Utilizes FreeRTOS for multi-core task management, showing proficiency in concurrent programming and real-time operating systems.

// 5. **Innovative Aspects**:
//    - **Two-Stage Face Detection**: The use of a two-stage face detection process improves accuracy and robustness, which is crucial for reliable security systems.
//    - **Efficiency**: The project is designed to be efficient in terms of processing and power consumption, aligning with the constraints and requirements of embedded systems.

// 6. **Practical Outcomes**:
//    - **Visual Feedback**: LED indications for face recognition status provide a simple yet effective user interface.
//    - **Execution Time Measurement**: Capturing and printing execution time shows a focus on performance evaluation, which is important for optimizing real-time systems.

// ### Potential Enhancements for Thesis Depth:

// 1. **Performance Metrics**:
//    - **Accuracy and Precision**: Evaluate the accuracy, precision, recall, and F1 score of the facial recognition system.
//    - **Power Consumption Analysis**: Measure and analyze the power consumption of the system under different operating conditions.

// 2. **Algorithmic Improvements**:
//    - **Advanced Algorithms**: Explore the integration of more sophisticated face recognition algorithms or deep learning models, if feasible within the constraints of the microcontroller.

// 3. **Security Enhancements**:
//    - **Data Encryption**: Implement encryption for image data transmission to enhance security.
//    - **Tamper Detection**: Add features to detect and respond to tampering or unauthorized access.

// 4. **Scalability and Flexibility**:
//    - **Multi-Face Detection**: Extend the system to handle multiple faces in a single frame.
//    - **Dynamic Configuration**: Implement features to adjust detection parameters dynamically based on environmental conditions.

// 5. **Real-World Testing and Validation**:
//    - **Deployment**: Test the system in real-world scenarios to validate its effectiveness and reliability.
//    - **User Feedback**: Gather feedback from potential users to identify areas for improvement.

// ### Conclusion:

// This project is robust and aligns well with the objectives of a bachelor's thesis in AI-based facial recognition for security and surveillance on low-power microprocessors. It demonstrates a solid integration of AI techniques with practical hardware implementation, making it a valuable contribution to the field of AI and embedded systems for security applications. Enhancing the project with additional performance metrics, security features, and real-world testing will further strengthen its impact and relevance.

// The provided code implements a Principal Component Analysis (PCA)-based approach, which is a foundational method in the creation and handling of eigenfaces for facial recognition. Here's an evaluation of the code's handling of eigenfaces and some suggestions for improvements:
// Analysis of Eigenfaces Handling:

//     PCA Computation:
//         The code divides the computation of PCA across two cores, improving efficiency.
//         Functionality:
//             applyPCA() function computes the projection of the centered face data onto the PCA transformation matrix.
//             Tasks are created to handle parts of the PCA computation (pcaTask_Core_0 and pcaTask_Core_1).

//     Data Centering:
//         Centering:
//             The function apply_Centering_CAM_Face_Data() subtracts the mean vector from the normalized face image data.
//             This is essential to zero-center the data, which is a standard preprocessing step before applying PCA.

//     Projection onto PCA Components:
//         Eigen Faces Projection:
//             The execute_Eigen_Faces_Projection() function projects the PCA output vector onto class centroids to find the closest match using Euclidean distance.




// Yes, your project definitely qualifies as an edge computing IoT AI facial recognition system for security and surveillance. Here's how each component of your project aligns with this classification:

// ### Edge Computing

// **Definition**: Edge computing involves processing data closer to where it is generated (at the "edge" of the network) rather than in a centralized data center or cloud.

// **Your Project**:
// - **On-device Processing**: The ESP32-S3 microcontroller processes the camera feed, detects faces, and performs facial recognition using PCA directly on the device.
// - **Real-time Decision Making**: By processing data locally, your system can make real-time decisions, such as identifying a face and providing immediate feedback through LEDs.

// ### IoT (Internet of Things)

// **Definition**: IoT refers to the network of physical devices that are connected to the internet, collecting and sharing data.

// **Your Project**:
// - **Connected Device**: The ESP32-S3 microcontroller is connected to a Wi-Fi network, enabling communication with other devices and systems.
// - **Data Sharing**: Your system sends facial recognition results to a server and a mobile application, integrating with a broader IoT ecosystem.

// ### AI (Artificial Intelligence)

// **Definition**: AI involves the simulation of human intelligence processes by machines, especially computer systems, which include learning, reasoning, and self-correction.

// **Your Project**:
// - **Face Detection and Recognition**: The use of machine learning techniques such as PCA for facial recognition constitutes AI. The system identifies and classifies faces based on learned patterns.
// - **PCA and Class Centroids**: Principal Component Analysis (PCA) is a machine learning technique used for dimensionality reduction, and the class centroids are part of the AI model used for face recognition.

// ### Facial Recognition for Security and Surveillance

// **Definition**: Using facial recognition technology to enhance security by identifying individuals and monitoring areas for surveillance purposes.

// **Your Project**:
// - **Security Application**: The primary function of your system is to recognize faces, which is a core requirement for many security and surveillance applications.
// - **Real-time Monitoring**: The system can be used to monitor premises in real time, providing immediate recognition and feedback, which is crucial for security purposes.

// ### Integration with a Mobile Application

// **Definition**: Developing a mobile application that interacts with your IoT system for real-time data visualization and management.

// **Your Project**:
// - **Real-time Results Display**: The mobile app will store and show the results of the facial recognition in real-time, enhancing the system's usability and providing a user-friendly interface for monitoring.
// - **Data Storage and Access**: The app can store recognition logs and provide access to historical data, which is valuable for security audits and investigations.

// ### Summary

// Your project exemplifies an edge computing IoT AI facial recognition system for security and surveillance by combining on-device processing, IoT connectivity, AI-based facial recognition, and integration with a mobile application for real-time monitoring and data management. This holistic approach makes it a comprehensive solution that leverages cutting-edge technology to enhance security and surveillance capabilities.





