#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

int main() {
    std::ifstream infile("6224.csv");
    std::ofstream outfile("6224.csv");

    if (!infile.is_open() || !outfile.is_open()) {
        std::cerr << "Error opening files!" << std::endl;
        return 1;
    }

    std::string line;
    int lower_bound = 521.7;
    int upper_bound = 521.8;

    // Читаем заголовок
    std::getline(infile, line);
    outfile << line << "\n";

    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string pixel_str, intensity_str;
        if (std::getline(iss, pixel_str, ',') && std::getline(iss, intensity_str)) {
            int pixel = std::stoi(pixel_str);
            if (pixel >= lower_bound && pixel <= upper_bound) {
                outfile << line << "\n";
            }
        }
    }

    infile.close();
    outfile.close();
    std::cout << "Filtering complete!" << std::endl;
    return 0;
}
