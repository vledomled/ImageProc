#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <libraw/libraw.h>

#define WAVES_FILE "Sensitivity.txt"

void export_to_csv(const char *out_filename, libraw_processed_image_t *img, float exposure) {
    if (!img) return;

    FILE *wf = fopen(WAVES_FILE, "r");
    FILE *out = fopen(out_filename, "w");
    if (!wf || !out) {
        if (wf) fclose(wf);
        if (out) fclose(out);
        return;
    }

    unsigned short *data = (unsigned short *)img->data;
    int w = img->width; 
    int h = img->height;

    long long noise_sum = 0;
    int sample_count = 1000; 
    for (int i = 0; i < sample_count; i++) {
        unsigned short val = data[i * 3];
        noise_sum += (unsigned short)((val << 8) | (val >> 8));
    }
    int baseline = (int)(noise_sum / sample_count);

    printf("FINAL EXPORT: Baseline %d, Exposure %.6f\n", baseline, exposure);

    int lines_written = 0;
    for (int y = 0; y < h; y++) {
        double wave_length = 0, sensitivity = 1.0;
        if (fscanf(wf, "%lf %lf", &wave_length, &sensitivity) == EOF) break;

        if (wave_length < 430.0 || wave_length > 600.0) continue; 
        if (sensitivity <= 0.000001) sensitivity = 1.0;

        fprintf(out, "%.4f", wave_length);

        for (int x = 0; x < w; x++) {
            int idx = (y * w + x) * 3;
            unsigned short val = data[idx];
            
            int native_val = (int)((unsigned short)((val << 8) | (val >> 8)));
            
            double signal = (double)(native_val - baseline);
            if (signal < 0) signal = 0;

            double final_val = signal / (sensitivity * (double)exposure);
            
            fprintf(out, ",%.6f", final_val);
        }
        fprintf(out, "\n");
        lines_written++;
    }

    fclose(wf);
    fclose(out);
    printf("Successfully finished. Total rows in CSV: %d\n", lines_written);
}

void debug_export_raw_match(const char *out_filename, libraw_processed_image_t *img) {
    if (!img) return;

    FILE *wf = fopen(WAVES_FILE, "r");
    if (!wf) {
        fprintf(stderr, "Error: Could not find '%s'\n", WAVES_FILE);
        return;
    }

    FILE *out = fopen(out_filename, "w");
    if (!out) {
        perror("Failed to create debug CSV");
        fclose(wf);
        return;
    }

    unsigned short *data = (unsigned short *)img->data;
    int w = img->width; 
    int h = img->height;

    printf("DEBUG MODE: Simple row-to-row matching (no calibration, no offset)...\n");

    for (int y = 0; y < h; y++) {
        double wave_length = 0;
        double sensitivity = 1.0;

        if (fscanf(wf, "%lf %lf", &wave_length, &sensitivity) == EOF) {
            printf("DEBUG: File ended at row %d, but image has %d rows\n", y, h);
            break;
        }

        fprintf(out, "%.4f", wave_length);

        for (int x = 0; x < w; x++) {
            int idx = (y * w + x) * 3;
            unsigned short val = data[idx];
            unsigned short native_val = (unsigned short)((val << 8) | (val >> 8));
            
            fprintf(out, ",%hu", native_val);
        }
        fprintf(out, "\n");
    }

    fclose(wf);
    fclose(out);
    printf("Debug CSV saved to %s\n", out_filename);
}

void crop_center_1000(libraw_processed_image_t **image_ptr) {
    libraw_processed_image_t *img = *image_ptr;
    if (!img || img->width <= 1000) return;

    int target_width = 1000;
    int height = img->height;
    int channels = 3;
    int bytes_per_sample = (img->bits == 16) ? 2 : 1;
    int pixel_size = channels * bytes_per_sample;
    
    int offset_x = (img->width - target_width) / 2;

    size_t new_data_size = (size_t)target_width * height * pixel_size;
    libraw_processed_image_t *new_img = (libraw_processed_image_t *)malloc(sizeof(libraw_processed_image_t) + new_data_size);
    
    if (!new_img) return;

    memcpy(new_img, img, sizeof(libraw_processed_image_t));
    new_img->width = target_width;
    new_img->data_size = (unsigned int)new_data_size;

    for (int y = 0; y < height; y++) {
        unsigned char *src_row = img->data + (y * img->width * pixel_size) + (offset_x * pixel_size);
        unsigned char *dst_row = new_img->data + (y * target_width * pixel_size);
        memcpy(dst_row, src_row, target_width * pixel_size);
    }

    printf("Cropped (16-bit): %d x %d -> %d x %d\n", img->width, height, target_width, height);

    libraw_dcraw_clear_mem(img);
    *image_ptr = new_img;
}

void convert_to_grayscale(libraw_processed_image_t *image) {
    if (!image || !image->data) return;
    printf("Converting to grayscale (16-bit) with Byte Swap...\n");

    int num_pixels = image->width * image->height;
    unsigned short *data = (unsigned short *)image->data;

    for (int i = 0; i < num_pixels; i++) {
        int idx = i * 3;
        unsigned short r = data[idx];
        unsigned short g = data[idx + 1];
        unsigned short b = data[idx + 2];

        float gray_val = (0.299f * r + 0.587f * g + 0.114f * b);
        
        unsigned short gray = (unsigned short)gray_val;

        unsigned short swapped = (unsigned short)((gray << 8) | (gray >> 8));

        data[idx] = data[idx + 1] = data[idx + 2] = swapped;
    }
}

void save_ppm(const char *filename, libraw_processed_image_t *image) {
    if (!image) return;

    FILE *fp = fopen(filename, "wb");
    if (!fp) { perror("Failed to open file"); return; }

    fprintf(fp, "P6\n%d %d\n65535\n", image->width, image->height);
    
    fwrite(image->data, 1, image->data_size, fp);
    fclose(fp);
    
    printf("File saved: %s (Size: %u bytes)\n", filename, image->data_size);
}

int main(int argc, char *argv[]) {
    if (argc < 3) return 1;

    libraw_data_t *lr_data = libraw_init(0);
    if (libraw_open_file(lr_data, argv[1]) != LIBRAW_SUCCESS) return 1;
    libraw_unpack(lr_data);

    float exposure = lr_data->other.shutter;
    printf("Detected exposure: %.6f seconds\n", exposure);

    lr_data->params.output_bps = 16;       
    lr_data->params.no_auto_bright = 1;    
    lr_data->params.use_camera_wb = 0;     
    lr_data->params.use_camera_matrix = 0; 
    lr_data->params.output_color = 0;      
    
    lr_data->params.gamm[0] = 1.0; 
    lr_data->params.gamm[1] = 1.0;

    for(int i=0; i<4; i++) lr_data->params.user_mul[i] = 1.0;
    
    if (libraw_dcraw_process(lr_data) != LIBRAW_SUCCESS) return 1;

    int ret;
    libraw_processed_image_t *processed_image = libraw_dcraw_make_mem_image(lr_data, &ret);

    if (processed_image) {
        crop_center_1000(&processed_image);
        convert_to_grayscale(processed_image);
        save_ppm(argv[2], processed_image);
        export_to_csv(argv[3], processed_image, exposure);
        debug_export_raw_match(argv[4], processed_image);
        free(processed_image); 
    }

    libraw_close(lr_data);
    return 0;
}