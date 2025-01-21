#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include "Network.h"

#define ROW_SIZE 28
#define COL_SIZE 28
#define IMAGE_SIZE 784
using namespace std;

// Function to read 4 bytes from the file as a big-endian integer
int readInt(std::ifstream& file) {
    unsigned char buffer[4];
    file.read(reinterpret_cast<char*>(buffer), 4);
    return (buffer[0] << 24) | (buffer[1] << 16) | (buffer[2] << 8) | buffer[3];
}

// Function to extract images from an idx3-ubyte file
vector<vector<double>> extractImages(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    // Read header information
    int magicNumber = readInt(file);
    if (magicNumber != 2051) { // Magic number for IDX3 files
        throw std::runtime_error("Invalid IDX3 file: Magic number mismatch");
    }
    int numImages = readInt(file);
    int numRows = readInt(file);
    int numCols = readInt(file);

    std::cout << "Number of images: " << numImages << std::endl;
    std::cout << "Image dimensions: " << numRows << " x " << numCols << std::endl;

    // Read image data
    vector<vector<double>> images(numImages,vector<double>(IMAGE_SIZE));

    cout<<"Image vector created\n";
    for (int i = 0; i < numImages; ++i) {
        for (int j = 0;j<IMAGE_SIZE;j++) {
            char ch;
            file.read(&ch,1);
            images[i][j] = (int)(unsigned char)(ch);
        }
    }
    cout<<endl;
    return images;
}

// Function to extract labels from an idx1-ubyte file
std::vector<double> extractLabels(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    // Read header information
    int magicNumber = readInt(file);
    if (magicNumber != 2049) { // Magic number for IDX1 files
        throw std::runtime_error("Invalid IDX1 file: Magic number mismatch");
    }
    int numLabels = readInt(file);

    std::cout << "Number of labels: " << numLabels << std::endl;

    // Read label data
    vector<double> labels(numLabels);
    for(int i=0;i<numLabels;i++)
    {
            char ch;
            file.read(&ch,1);
            labels[i] = (int)(unsigned char)(ch);
    }

    return labels;
}


// Function to display an image (for debugging)
void displayImage(vector<double> image ,double label) {
    int k=0;
    for (int i=0;i<ROW_SIZE;i++) {
        for (int j=0;j<COL_SIZE;j++) {
            std::cout << (image[k] > 0 ? '#' : '.') << " ";
            k+=1;
        }
        std::cout << std::endl;
    }
    cout<<"Image is "<<label;
    cout<<endl;
}

int main() {
    try {
        // Specify the path to your idx3-ubyte file
        std::string fileTrainImagesname = "MNIST_ORG/train-images.idx3-ubyte";
        std::string fileTrainLabelsname = "MNIST_ORG/train-labels.idx1-ubyte";
        std::string fileTestImagesname = "MNIST_ORG/t10k-images.idx3-ubyte";
        std::string fileTestLabelsname = "MNIST_ORG/t10k-labels.idx1-ubyte";
        
        // Extract images from the file
        auto imagesTrain = extractImages(fileTrainImagesname);

        auto labelsTrain = extractLabels(fileTrainLabelsname);

        auto imagesTest = extractImages(fileTestImagesname);

        auto labelsTest = extractLabels(fileTestLabelsname);


        Network network;

        network.setLearningRate(10);

        network.AddLayer(784);
        network.AddLayer(1000);
        network.AddLayer(1000);
        network.AddLayer(9);

        int epochs = 1;
        int batchSizeTrain = 1;
        int batchSizeTest = 1;
        for(int i=0;i<epochs;i++)
        {

            for(int j=0;j<batchSizeTrain;j++)
            {
                network.trainNetwork(imagesTrain[j],labelsTrain[j]);
            }

            double totalError = 0;

            for(int j=0;j<batchSizeTest;j++)
            {
                totalError+=network.testNetwork(imagesTest[j],labelsTest[j]);
            }

            cout<<"Total Loss is "<<totalError<<"\n";
            if(totalError<1)
            {
                break;
            }

    }


    for(int i=0;i<batchSizeTest;i++)
    {
            auto output=network.getOutput(imagesTest[i]);
            for(int j=0;j<output.size();j++)
            {
                cout<<output[j]<<" ";
            }
            cout<<"Expected is "<<labelsTest[i];
            cout<<endl;
    }

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
    }

    return 0;
}
