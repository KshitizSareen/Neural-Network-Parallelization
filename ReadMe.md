# **Parallelized Neural Network in C++ Using OpenMP**  

## **Project Overview**  
This project implements a fully functional neural network from scratch in C++, focusing on high-performance computing using OpenMP for Parallelization. The neural network is optimized for multi-threaded execution, significantly improving training time compared to a sequential implementation.  

The project showcases the power of forward propagation, backpropagation, and weight updates with parallelized computations using OpenMP. We have made a strategic choice in using the Eigen library for efficient matrix operations, which has seamlessly handled weight updates and activations.  

## **Key Features**  
- **Implemented a Neural Network from Scratch** 
- **Parallelized Forward Propagation & Backpropagation** – Optimized training using **OpenMP**.  
- **Utilized Eigen for Matrix Operations** – Boosted performance with Eigen’s **expression templates**.  
- **Significant Speed Improvement** – Reduced training time from **20 seconds per iteration** to **9 seconds per iteration**.  

---

## **Installation & Compilation**  
### **Compiling the Project with g++**  
To compile the project using **g++ (GCC 14) with OpenMP**, navigate to the project directory and run:  

g++-14 -fopenmp main.cpp Weight.cpp Neuron.cpp Network.cpp -o program

- The `-fopenmp` flag enables **OpenMP** support.  
- This will generate an **executable** named `program,` 
- We can run the program using:  
 ./program

---

## **Dependencies**  
This project relies on two key dependencies:  

### **1. OpenMP (for Parallelization)**  
OpenMP is used to parallelize **computationally intensive loops**, distributing workloads across multiple CPU cores for faster execution.  

**Installation Guide:**  
- **Linux/macOS:** OpenMP is included with **GCC**. Ensure you use a compatible version (`g++-14` or later).  
- **Windows:** Use **MinGW-w64** to install a GCC version with OpenMP support.  

For more details on installing OpenMP:  
🔗 [OpenMP Official Documentation](https://www.openmp.org/)  

---

### **2. Eigen (for High-Performance Matrix Operations)**  
Eigen is a powerful **C++ library** optimized for **linear algebra operations** such as matrix multiplication.


For more details on installing Eigen:  
🔗 [Eigen Official Documentation](https://eigen.tuxfamily.org/)  

---

## **Project Structure**  
```
/project-folder
│── main.cpp         # Entry point of the neural network
│── Network.cpp      # Implements the neural network logic
│── Network.h        # Header file for the Network class
│── Neuron.cpp       # Implements neuron behavior
│── Neuron.h         # Header file for the Neuron class
│── Weight.cpp       # Implements weight-related computations
│── Weight.h         # Header file for the Weight class
│── README.md        # This file
│── /eigen           # Eigen library (if manually added)
```

---

## **Usage**  
1. **Compile the project** using the provided `g++` command.  
2. **Run the executable** using:
   ./program
3. The program will **train the neural network** and output training performance metrics.  

---

## **Results & Performance Gains**  
- **Sequential Training Time:** 20 seconds per iteration  
- **Parallelized Training Time:** 9 seconds per iteration  
- **Speedup Achieved:** **2.22× improvement** using OpenMP  

---

## **Further Optimizations** :  
✔ **Vectorization Techniques** – Using **SIMD instructions** for even faster execution.  
✔ **GPU Acceleration** – Offloading computations to a **CUDA-compatible GPU** for deep learning tasks.  
✔ **Optimized Memory Access** – Improving **cache locality** to reduce memory bottlenecks.  

---

## **Conclusion**  
This project demonstrates how low-level optimizations and parallel computing can drastically improve the performance of machine learning models. By using OpenMP for multi-threaded execution and Eigen for efficient matrix computations, we successfully built a high-performance neural network entirely in C++.  

If you want to explore the full implementation, you can find the **code and detailed explanations on GitHub** (link below). Happy coding, and let's keep pushing the boundaries of AI performance! 🚀  

🔗 **[GitHub Link]** (Insert your GitHub repo link here)  

