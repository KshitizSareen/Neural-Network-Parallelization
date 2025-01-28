// MyClass.cpp
#include "MyClass.h"  // Include the header file

// Constructor implementation
MyClass::MyClass(int value) : data(value) {}

// Member function to set data
void MyClass::setData(int value) {
    data = value;
}

// Member function to get data
int MyClass::getData() const {
    return data;
}
