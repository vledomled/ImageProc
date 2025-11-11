#include <stdio.h>
#include <stdlib.h>
#include <libraw/libraw.h>

// NEW: Function to convert an image to grayscale
void convert_to_grayscale(libraw_processed_image_t *image) {
    if (!image || !image->data) return;

    printf("Converting to grayscale...\n");
    for (int i = 0; i < image->data_size; i += 3) {
        // Get the R, G, B components of the current pixel
        unsigned char r = image->data[i];
        unsigned char g = image->data[i + 1];
        unsigned char b = image->data[i + 2];

        // Calculate the grayscale value using the luminosity formula
        // The result is cast to unsigned char after calculation.
        unsigned char gray = (unsigned char)(0.299 * r + 0.587 * g + 0.114 * b);

        // Set all three components (R, G, B) to the new gray value
        image->data[i] = gray;
        image->data[i + 1] = gray;
        image->data[i + 2] = gray;
    }
}

// Function to save the image in PPM (P6) format.
void save_ppm(const char *filename, libraw_processed_image_t *image) {
    if (!image || image->type != LIBRAW_IMAGE_BITMAP || image->colors != 3 || image->bits != 8) {
        fprintf(stderr, "Error: The processed image has an incorrect format!\n");
        return;
    }

    printf("Saving to file: %s\n", filename);
    printf("Image dimensions: %d x %d\n", image->width, image->height);

    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        perror("Failed to open file for writing");
        return;
    }

    fprintf(fp, "P6\n%d %d\n255\n", image->width, image->height);
    fwrite(image->data, 1, image->data_size, fp);
    fclose(fp);
    
    printf("File saved successfully!\n");
}


int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input_raw_file> <output_ppm_file>\n", argv[0]);
        return 1;
    }

    const char* raw_file_path = argv[1];
    const char* output_file_path = argv[2];
    int ret;

    libraw_data_t *lr_data = libraw_init(0);
    if (!lr_data) {
        fprintf(stderr, "Failed to initialize libraw\n");
        return 1;
    }

    printf("Opening file: %s\n", raw_file_path);
    if ((ret = libraw_open_file(lr_data, raw_file_path)) != LIBRAW_SUCCESS) {
        fprintf(stderr, "Failed to open file: %s (%s)\n", raw_file_path, libraw_strerror(ret));
        libraw_close(lr_data);
        return 1;
    }

    printf("Unpacking data...\n");
    if ((ret = libraw_unpack(lr_data)) != LIBRAW_SUCCESS) {
        fprintf(stderr, "Failed to unpack data: %s\n", libraw_strerror(ret));
        libraw_close(lr_data);
        return 1;
    }

    printf("Applying in-camera settings...\n");
    lr_data->params.use_camera_wb = 1;
    lr_data->params.no_auto_bright = 1;
    lr_data->params.highlight = 2; // A good default value.
    lr_data->params.use_camera_matrix = 1; 

    printf("Processing image...\n");
    if ((ret = libraw_dcraw_process(lr_data)) != LIBRAW_SUCCESS) {
        fprintf(stderr, "Failed to process image: %s\n", libraw_strerror(ret));
        libraw_close(lr_data);
        return 1;
    }

    libraw_processed_image_t *processed_image = libraw_dcraw_make_mem_image(lr_data, &ret);
    if (!processed_image) {
        fprintf(stderr, "Failed to get access to the processed image: %s\n", libraw_strerror(ret));
        libraw_close(lr_data);
        return 1;
    }

    convert_to_grayscale(processed_image);

    // Save the now-grayscale image to a PPM file
    save_ppm(output_file_path, processed_image);

    // Clean up resources
    libraw_dcraw_clear_mem(processed_image);
    libraw_close(lr_data);

    return 0;
}