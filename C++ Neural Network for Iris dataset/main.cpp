#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include "MyClass.h"
#include "Network.h"
#include "Eigen/Dense"
#include <chrono> // For high-resolution timing
using namespace std;
using namespace Eigen;

// Struct to hold a single data point
struct IrisData {
    std::vector<double> features;  // Features: sepal length, sepal width, petal length, petal width
    std::string label;           // Label: Species (e.g., Iris-setosa)
    int labelClassification;
};

int classifier(const std::string label) {
    if (label == "Setosa") {
        return 0;
    }
    if (label == "Versicolor") {
        return 1;
    }
    if (label == "Virginica") {
        return 2;
    }
    return 3;  // Return 3 if the label does not match any known species
}


// Function to read the Iris dataset
std::vector<IrisData> loadIrisDataset(const std::string& filename) {
    std::vector<IrisData> dataset;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return dataset;
    }

    std::string line;

    int k=0;
    while (std::getline(file, line)) {
                    if(k>=1)
            {
        if (line.empty()) continue;

        std::stringstream ss(line);
        IrisData data;
        std::string value;
        int column = 0;


        // Read features and label
        while (std::getline(ss, value, ',')) {
            if (column < 4) {

                // Convert feature values to double
                data.features.push_back(std::stof(value));
            } else {
                // Read the label (last column)
                    // Replace the first occurrence of 'o' with 'x'
                value = value.substr(1,value.size()-2);
                data.label = value;
                data.labelClassification = classifier(value);
            }
            column++;
        }
        dataset.push_back(data);
            }
            k+=1;
    }

    file.close();
    return dataset;
}


// Main function
int main() {


    std::string filename = "output_sequantial.csv";

    // Open file in write mode
    std::ofstream file(filename);

    file.clear();

    filename = "iris.csv";
    std::vector<IrisData> irisDataset = loadIrisDataset(filename);

    Network network;
    network.setLearningRate(0.5);

    network.AddLayer(4);
    network.AddLayer(1000);
    network.AddLayer(1000);
    network.AddLayer(3);

    for(int j=0;j<10000;j++)
    {
        double totalError = 0;
        for(int i=0;i<irisDataset.size();i++)
        {
            
            // Start timing
            auto start = std::chrono::high_resolution_clock::now();
            network.trainNetwork(irisDataset[i].features,irisDataset[i].labelClassification);
            // Stop timing
            auto stop = std::chrono::high_resolution_clock::now();
        
            // Calculate the duration
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
            
            // Print the duration
            std::cout << "Execution Time: " << duration.count() << " ms" << " For iteration "<<i<<endl;
        }
        for(int i=0;i<irisDataset.size();i++)
        {
            totalError+=network.testNetwork(irisDataset[i].features,irisDataset[i].labelClassification);
        }
        cout<<"number of epochs are "<<j<<"\n";
        cout<<"Total Loss is "<<totalError<<"\n";
        file <<j;
        file<<","<<totalError;
        file<<"\n";
        if(totalError<1)
        {
            cout<<"Total number of epochs are "<<j<<"\n";
            break;
        }

    }

    /*
    for(int i=0;i<irisDataset.size();i++)
    {
            auto output=network.getOutput(irisDataset[i].features);
            for(int j=0;j<output.size();j++)
            {
                cout<<output[j]<<" ";
            }
            cout<<"Expected is "<<irisDataset[i].labelClassification;
            cout<<endl;
    }
    */

    return 0;
}
