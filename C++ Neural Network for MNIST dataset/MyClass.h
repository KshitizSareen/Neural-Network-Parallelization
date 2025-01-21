// MyClass.h
#ifndef MYCLASS_H  // Include guard to prevent multiple inclusions
#define MYCLASS_H

class MyClass {
private:
    int data; // Private member variable

public:
    // Constructor
    MyClass(int value);

    // Member function to set data
    void setData(int value);

    // Member function to get data
    int getData() const;
};

#endif