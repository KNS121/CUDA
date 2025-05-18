#pragma once

#include <GL/glew.h>
#include <cuda_runtime.h>

#define PARTICLE_TYPE_A 0
#define PARTICLE_TYPE_B 1
#define MAX_PARTICLES 1024
#define GRID_SIZE 32
#define BLOCK_SIZE 256


struct Particle {
    float x, y;    
    float vx, vy;   
    int type;       
};

struct CudaParams {
    // Graphics
    cudaGraphicsResource* cudaResource = nullptr;
    cudaSurfaceObject_t surface = 0;
    GLuint texture = 0;

    // Physics
    float* positions_x = nullptr;
    float* positions_y = nullptr;
    float* velocities_x = nullptr;
    float* velocities_y = nullptr;
    int* types = nullptr;
    int* cellIndices = nullptr;
    int* cellStarts = nullptr;
    int* cellEnds = nullptr;
};