#include <cuda.h>
#include <math.h>

// Inline is a hint to the compiler to replace the function at the point where its called to redude overhead


// Addition

inline __host__ __device__ float3 operator+(float a, float3 b){
    return make_float3(a + b.x, a + b.y, a + b.z);
}

inline __host__ __device__ float3 operator+(float3 a, float b){
    return make_float3(a.x + b, a.y + b, a.z + b);
}

inline __host__ __device__ float3 operator+(float3 a, float3 b){
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}



// Subtraction
inline __host__ __device__ float3 operator-(float3 a, float3 b){
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}



// Multiplication
inline __host__ __device__ float3 operator*(float a, float3 b){
    return make_float3(a * b.x, a * b.y, a * b.z);
}

inline __host__ __device__ float3 operator*(float3 a, float b){
    return make_float3(a.x * b, a.y * b, a.z * b);
}

inline __host__ __device__ float3 operator*(float3 a, float3 b){
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}



// Division
inline __host__ __device__ float3 operator/(float a, float3 b){
    return make_float3(a / b.x, a / b.y, a / b.z);
}

inline __host__ __device__ float3 operator/(float3 a, float b){
    return make_float3(a.x / b, a.y / b, a.z / b);
}

inline __host__ __device__ float3 operator/(float3 a, float3 b){
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}



// Sum
inline __host__ __device__ float sum(float3 a){
    return (a.x + a.y + a.z);
}
