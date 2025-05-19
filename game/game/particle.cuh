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
    int* active;
    int* cellIndices = nullptr;
    int* cellStarts = nullptr;
    int* cellEnds = nullptr;
};


__device__ int countCaptured;

__constant__ float d_world_size = 2.0f;
const float min_distance = 0.025f;
int numParticles = 1024;
int winWidth = 1024, winHeight = 1024;
CudaParams cudaParams;
float deltaTime = 0.016f;

__constant__ float particle_radius;
__constant__ float basket_left;
__constant__ float basket_right;
__constant__ float basket_bottom;
__constant__ float basket_top;
__constant__ float basket_thickness;

__constant__ float gravity;

__constant__ float max_speed;


__constant__ float segment_x;
__constant__ float segment_y;
__constant__ float segment_length;
__constant__ float segment_thickness;

__device__ float rect_left;
__device__ float rect_right;
__device__ float rect_bottom;
__device__ float rect_top;
__device__ bool rect_exists;


int capturedParticles = 0;


int startTime;
const int GAME_SECONDS = 150;
bool gameOver = false;
bool gameWon = false;

bool timerSet = false;
int windowID;