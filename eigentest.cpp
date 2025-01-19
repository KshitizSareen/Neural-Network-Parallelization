#include <iostream>
#include <Eigen/Dense>


using namespace std;
using namespace Eigen;

int main(){
        // Define a 4x3 matrix
    MatrixXd mat(5,4);
    mat << 0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0;

    // Define a 3x1 matrix (vector)
    MatrixXd vec(4,1);
    vec << 5.1,
           3.5,
           1.4,
           0.2;

    MatrixXd bias(5,1);
    bias << 1,0,1,0,1;

    // Perform matrix multiplication
    MatrixXd result = (mat * vec)+bias;

    // Output the result
    cout << "Matrix:\n" << mat << "\n\n";
    cout << "Vector:\n" << vec << "\n\n";
    cout << "Result:\n" << result << std::endl;
}