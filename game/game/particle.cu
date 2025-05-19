#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include "particle.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#include <string>
#include <sstream>

using namespace std;

#include <thrust/remove.h>
#include <thrust/execution_policy.h>




///// GAME MANAGMENT /////

float h_segment_x = 0.0f;
float h_segment_y = 0.0f;
const float h_segment_length = 0.4f;
const float h_segment_thickness = 0.02f;
float move_speed = 0.12f;


float h_rect_left = -0.2f;
float h_rect_right = 0.2f;
float h_rect_bottom = 0.5f;
float h_rect_top = 0.7f;
bool h_rect_exists = false;

void closeWindow(int value) {
    glutDestroyWindow(windowID);
    exit(0);
}

void keyboard(unsigned char key, int x, int y) {
    switch (key) {
    case 'w':
        h_segment_y += move_speed;
        break;
    case 's':
        h_segment_y -= move_speed;
        break;
    case 'a':
        h_segment_x -= move_speed;
        break;
    case 'd':
        h_segment_x += move_speed;
        break;


    case 'f':
        h_rect_exists = !h_rect_exists;
        cudaMemcpyToSymbol(rect_exists, &h_rect_exists, sizeof(bool));
        break;
    }


    // ne nado za okno ubegat'
    h_segment_x = max(-1.0f + h_segment_length / 2, min(1.0f - h_segment_length / 2, h_segment_x));
    h_segment_y = max(-1.0f + h_segment_thickness / 2, min(1.0f - h_segment_thickness / 2, h_segment_y));


    cudaMemcpyToSymbol(segment_x, &h_segment_x, sizeof(float));
    cudaMemcpyToSymbol(segment_y, &h_segment_y, sizeof(float));

    glutPostRedisplay();
}


///// GAME MANAGMENT /////





//////// CUDA CUDA CUDA CUDA CUDA CUDA CUDA CUDA ////////////////////////


// for cuda
__device__ int calcGridHash(float x, float y) {
    int gridX = static_cast<int>((x + 1.0f) * (GRID_SIZE / (2.0f * d_world_size)));
    int gridY = static_cast<int>((y + 1.0f) * (GRID_SIZE / (2.0f * d_world_size)));
    gridX = max(0, min(gridX, GRID_SIZE - 1));
    gridY = max(0, min(gridY, GRID_SIZE - 1));
    return gridY * GRID_SIZE + gridX;
}

__global__ void sortParticles(float* positions_x, float* positions_y, int* cellIndices, int numParticles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    int hash = calcGridHash(positions_x[idx], positions_y[idx]);
    cellIndices[idx] = (hash << 16) | idx;
}

__global__ void setupGrid(int* cellIndices, int* cellStarts, int* cellEnds, int numParticles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    int currentHash = cellIndices[idx] >> 16;

    // start yacheyki odin hash  - odna gruppa
    if (idx == 0 || currentHash != (cellIndices[idx - 1] >> 16)) {
        cellStarts[currentHash] = idx;
        //atomicMin(&cellStarts[currentHash], idx); // shob ne ustraivat' draku between threads
    }

    // end_yacheyki
    if (idx == numParticles - 1 || currentHash != (cellIndices[idx + 1] >> 16)) {
        cellEnds[currentHash] = idx + 1;
        //atomicMax(&cellEnds[currentHash], idx + 1);
    }
}

__global__ void updateParticles(float* positions_x, float* positions_y,
    float* velocities_x, float* velocities_y,
    int numParticles, float deltaTime, int* active)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles || !active[idx]) return;

    velocities_y[idx] += gravity * deltaTime;

    float prev_x = positions_x[idx];
    float prev_y = positions_y[idx];

    float current_speed = sqrtf(velocities_x[idx] * velocities_x[idx] +
        velocities_y[idx] * velocities_y[idx]);
    if (current_speed > max_speed) {
        float ratio = max_speed / current_speed;
        velocities_x[idx] *= ratio;
        velocities_y[idx] *= ratio;
    }


    positions_x[idx] += velocities_x[idx] * deltaTime;
    positions_y[idx] += velocities_y[idx] * deltaTime;

    //prik skok s kraev
    if (fabsf(positions_x[idx]) + particle_radius >= 1.0f) {
        velocities_x[idx] *= -1.0f;
        positions_x[idx] = copysignf(1.0f - particle_radius, positions_x[idx]);
    }
    if (fabsf(positions_y[idx]) + particle_radius >= 1.0f) {
        velocities_y[idx] *= -1.0f;
        positions_y[idx] = copysignf(1.0f - particle_radius, positions_y[idx]);
    }


    const float thickness = basket_thickness;
    const float left = basket_left;
    const float right = basket_right;
    const float bottom = basket_bottom;
    const float top = basket_top;


    // vnutri
    if (positions_y[idx] >= bottom && positions_y[idx] <= top) {
        // left
        if (positions_x[idx] - particle_radius < left + thickness &&
            prev_x - particle_radius >= left + thickness) {
            velocities_x[idx] *= -1.0f;
            positions_x[idx] = left + thickness + particle_radius;
        }
        // right
        if (positions_x[idx] + particle_radius > right - thickness &&
            prev_x + particle_radius <= right - thickness) {
            velocities_x[idx] *= -1.0f;
            positions_x[idx] = right - thickness - particle_radius;
        }
    }

    // vne
    // left
    const float outer_left = left - thickness;
    if (positions_x[idx] + particle_radius > outer_left &&
        positions_x[idx] - particle_radius < outer_left + thickness &&
        positions_y[idx] >= bottom &&
        positions_y[idx] <= top) {
        positions_x[idx] = outer_left - particle_radius;
        velocities_x[idx] *= -1.0f;
    }

    // rignt
    const float outer_right = right + thickness;
    if (positions_x[idx] - particle_radius < outer_right &&
        positions_x[idx] + particle_radius > outer_right - thickness &&
        positions_y[idx] >= bottom &&
        positions_y[idx] <= top) {
        positions_x[idx] = outer_right + particle_radius;
        velocities_x[idx] *= -1.0f;
    }

    // niz
    if (positions_x[idx] >= left && positions_x[idx] <= right) {
        //vnutri niz
        if (positions_y[idx] - particle_radius < bottom + thickness &&
            prev_y - particle_radius >= bottom + thickness) {
            positions_y[idx] = bottom + thickness + particle_radius;
            velocities_y[idx] *= -1.0f;
        }
        //vne niz
        const float outer_bottom = bottom - thickness;
        if (positions_y[idx] + particle_radius > outer_bottom && positions_y[idx] - particle_radius < outer_bottom + thickness) {
            positions_y[idx] = outer_bottom - particle_radius;
            velocities_y[idx] *= -1.0f;
        }
    }

    // korzina ubivaet
    if (positions_x[idx] >= basket_left && positions_x[idx] <= basket_right && positions_y[idx] >= basket_bottom && positions_y[idx] <= basket_top)
    {
        active[idx] = 0;
        atomicAdd(&countCaptured, 1);
    }


    float seg_left = segment_x - segment_length / 2;
    float seg_right = segment_x + segment_length / 2;
    float seg_bottom = segment_y - segment_thickness / 2;
    float seg_top = segment_y + segment_thickness / 2;

    // my otrezok
    if (positions_x[idx] >= seg_left - particle_radius &&
        positions_x[idx] <= seg_right + particle_radius &&
        positions_y[idx] >= seg_bottom - particle_radius &&
        positions_y[idx] <= seg_top + particle_radius)
    {

        if (prev_y > seg_top && positions_y[idx] <= seg_top) {
            velocities_y[idx] *= -1.0f;
            positions_y[idx] = seg_top + particle_radius;
        }
        else if (prev_y < seg_bottom && positions_y[idx] >= seg_bottom) {
            velocities_y[idx] *= -1.0f;
            positions_y[idx] = seg_bottom - particle_radius;
        }

        // horizontal
        if (prev_x < seg_left && positions_x[idx] >= seg_left) {
            velocities_x[idx] *= -1.0f;
            positions_x[idx] = seg_left - particle_radius;
        }
        else if (prev_x > seg_right && positions_x[idx] <= seg_right) {
            velocities_x[idx] *= -1.0f;
            positions_x[idx] = seg_right + particle_radius;
        }
    }


    if (rect_exists) {
        float r_left = rect_left - particle_radius;
        float r_right = rect_right + particle_radius;
        float r_bottom = rect_bottom - particle_radius;
        float r_top = rect_top + particle_radius;

        if (positions_x[idx] >= r_left &&
            positions_x[idx] <= r_right &&
            positions_y[idx] >= r_bottom &&
            positions_y[idx] <= r_top) {


            if (prev_y > rect_top && positions_y[idx] <= rect_top) {
                velocities_y[idx] *= -1.0f;
                positions_y[idx] = rect_top + particle_radius;
            }
            else if (prev_y < rect_bottom && positions_y[idx] >= rect_bottom) {
                velocities_y[idx] *= -1.0f;
                positions_y[idx] = rect_bottom - particle_radius;
            }

            if (prev_x < rect_left && positions_x[idx] >= rect_left) {
                velocities_x[idx] *= -1.0f;
                positions_x[idx] = rect_left - particle_radius;
            }
            else if (prev_x > rect_right && positions_x[idx] <= rect_right) {
                velocities_x[idx] *= -1.0f;
                positions_x[idx] = rect_right + particle_radius;
            }
        }
    }
}

__global__ void processBlockPrikSkok(float* positions_x, float* positions_y, float* velocities_x, float* velocities_y, int* types, int numParticles, float min_dist) {
    // usaem shared
    __shared__ struct {
        float x[BLOCK_SIZE];
        float y[BLOCK_SIZE];
        int type[BLOCK_SIZE];
        float vx[BLOCK_SIZE];
        float vy[BLOCK_SIZE];
    } shared;

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;

    // chto-to tupanul, lishnego nam ne nado
    if (tid < numParticles - blockIdx.x * blockDim.x) { 
        shared.x[tid] = positions_x[idx];
        shared.y[tid] = positions_y[idx];
        shared.type[tid] = types[idx];
        shared.vx[tid] = velocities_x[idx];
        shared.vy[tid] = velocities_y[idx];
    }
    __syncthreads();

    if (idx >= numParticles) return;

    const int type1 = shared.type[tid];
    float delta_vx = 0.0f, delta_vy = 0.0f;
    float delta_x = 0.0f, delta_y = 0.0f;

    //ostanetsa obrabotat'
    int ostatok_num = numParticles - blockIdx.x * blockDim.x;


    for (int i = 0; i < min(BLOCK_SIZE, ostatok_num); ++i) {
        const int type2 = shared.type[i];
        //sebya ne obrabatyvaem
        if (type1 == type2 || i == tid) continue;

        const float dx = shared.x[tid] - shared.x[i];
        const float dy = shared.y[tid] - shared.y[i];
        const float distSq = dx * dx + dy * dy;

        if (distSq < min_dist * min_dist && distSq > 1e-8f) {
            const float dist = sqrtf(distSq);
            const float nx = dx / dist, ny = dy / dist;
            const float relVel = (shared.vx[tid] - shared.vx[i]) * nx
                + (shared.vy[tid] - shared.vy[i]) * ny;

            if (relVel >= 0) continue;

            const float impulse = -1.0f * relVel;
            delta_vx += impulse * nx;
            delta_vy += impulse * ny;

            const float overlap = 0.5f * (min_dist - dist);
            delta_x += overlap * nx;
            delta_y += overlap * ny;
        }
    }

    __syncthreads();

    //ot greha podalshe
    //atomicAdd(&velocities_x[idx], delta_vx);
    //atomicAdd(&velocities_y[idx], delta_vy);
    //atomicAdd(&positions_x[idx], delta_x);
    //atomicAdd(&positions_y[idx], delta_y);

    // every thread obrabatyvaet svoe, mozghno bystree

    if (idx < numParticles) {
        velocities_x[idx] = shared.vx[tid];
        velocities_y[idx] = shared.vy[tid];
        positions_x[idx] = shared.x[tid];
        positions_y[idx] = shared.y[tid];
    }

}

// mezhgdu gridami
__global__ void processGridPrikSkok(
    float* positions_x,
    float* positions_y,
    float* velocities_x,
    float* velocities_y,
    int* types,
    int* cellStarts,
    int* cellEnds,
    int* cellIndices,
    int numParticles,
    float min_dist
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    const float p1_x = positions_x[idx], p1_y = positions_y[idx];
    const float v1_x = velocities_x[idx], v1_y = velocities_y[idx];
    const int type1 = types[idx];

    const int gridHash = calcGridHash(p1_x, p1_y);
    const int gridX = gridHash % GRID_SIZE, gridY = gridHash / GRID_SIZE;

    float delta_vx = 0.0f, delta_vy = 0.0f;
    float delta_x = 0.0f, delta_y = 0.0f;

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            const int neighborX = gridX + dx, neighborY = gridY + dy;
            if (neighborX < 0 || neighborX >= GRID_SIZE) continue;
            if (neighborY < 0 || neighborY >= GRID_SIZE) continue;

            const int neighborHash = neighborY * GRID_SIZE + neighborX;
            const int start = cellStarts[neighborHash], end = cellEnds[neighborHash];
            if (start == -1 || end == -1) continue;

            for (int k = start; k < end; k++) {
                const int idx2 = cellIndices[k] & 0xFFFF;

                // nado ne peresechsya s block-prik-skok
                if (idx2 >= blockIdx.x * blockDim.x && idx2 < (blockIdx.x + 1) * blockDim.x) continue;
                
                if (idx2 <= idx) continue;

                const int type2 = types[idx2];
                if (type1 == type2) continue;

                const float p2_x = positions_x[idx2], p2_y = positions_y[idx2];
                const float dx = p1_x - p2_x, dy = p1_y - p2_y;
                const float distSq = dx * dx + dy * dy;

                if (distSq < min_dist * min_dist && distSq > 1e-8f) {
                    const float dist = sqrtf(distSq);
                    const float nx = dx / dist, ny = dy / dist;
                    const float relVel = (v1_x - velocities_x[idx2]) * nx
                        + (v1_y - velocities_y[idx2]) * ny;

                    if (relVel >= 0) continue;

                    const float impulse = -1.0f * relVel; // nado vernut'... mne ne nrav bez edinicy
                    delta_vx += impulse * nx;
                    delta_vy += impulse * ny;

                    velocities_x[idx2] -= impulse * nx;
                    velocities_y[idx2] -= impulse * ny;

                    const float overlap = 0.5f * (min_dist - dist);
                    delta_x += overlap * nx;
                    delta_y += overlap * ny;
                    positions_x[idx2] -= overlap * nx;
                    positions_y[idx2] -= overlap * ny;
                }
            }
        }
    }

    //atomicAdd(&velocities_x[idx], delta_vx);
    //atomicAdd(&velocities_y[idx], delta_vy);
    //atomicAdd(&positions_x[idx], delta_x);
    //atomicAdd(&positions_y[idx], delta_y);

    // mozghno pobystree
    velocities_x[idx] += delta_vx;
    velocities_y[idx] += delta_vy;
    positions_x[idx] += delta_x;
    positions_y[idx] += delta_y;
}


__global__ void ClearTexture(cudaSurfaceObject_t surface, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;


    float worldX = (x / (float)width) * 2.0f - 1.0f;
    float worldY = (y / (float)height) * 2.0f - 1.0f;

    uchar4 color = make_uchar4(0, 0, 0, 255);


    const float thickness = basket_thickness;
    const float left = basket_left;
    const float right = basket_right;
    const float bottom = basket_bottom;
    const float top = basket_top;


    // vnutr
    bool isBasket =
        // vnutri levaya
        (fabs(worldX - (left + thickness / 2)) < thickness / 2 &&
            worldY >= bottom && worldY <= top) ||

        // vnutr pravaya
        (fabs(worldX - (right - thickness / 2)) < thickness / 2 &&
            worldY >= bottom && worldY <= top) ||

        // niz vnutri
        (fabs(worldY - (bottom + thickness / 2)) < thickness / 2 &&
            worldX >= left && worldX <= right);

    // vnesh
    isBasket = isBasket ||
        // vneshn levaya
        (fabs(worldX - (left - thickness / 2)) < thickness / 2 &&
            worldY >= bottom && worldY <= top) ||

        // vneshnyaya pravaya
        (fabs(worldX - (right + thickness / 2)) < thickness / 2 &&
            worldY >= bottom && worldY <= top) ||

        // vneshnyaya niz
        (fabs(worldY - (bottom - thickness / 2)) < thickness / 2 &&
            worldX >= left && worldX <= right);

    if (isBasket) {
        color = make_uchar4(0, 255, 0, 255);
    }

    //surf2Dwrite(color, surface, x * sizeof(uchar4), y);

    float seg_left = segment_x - segment_length / 2;
    float seg_right = segment_x + segment_length / 2;
    float seg_bottom = segment_y - segment_thickness / 2;
    float seg_top = segment_y + segment_thickness / 2;

    if (worldX >= seg_left && worldX <= seg_right &&
        worldY >= seg_bottom && worldY <= seg_top) {
        color = make_uchar4(255, 255, 255, 255);
    }

    if (rect_exists) {
        if (worldX >= rect_left && worldX <= rect_right &&
            worldY >= rect_bottom && worldY <= rect_top) {
            color = make_uchar4(255, 255, 0, 255); // pust budet yello
        }
    }

    surf2Dwrite(color, surface, x * sizeof(uchar4), y);
}

__global__ void drawParticles(cudaSurfaceObject_t surface,
    float* positions_x, float* positions_y,
    int* types, int numParticles,
    int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    float x = positions_x[idx];
    float y = positions_y[idx];
    int type = types[idx];

    int px = static_cast<int>((x + 1.0f) * 0.5f * width);
    int py = static_cast<int>((y + 1.0f) * 0.5f * height);

    uchar4 color = (type == PARTICLE_TYPE_A)
        ? make_uchar4(255, 50, 50, 255)
        : make_uchar4(50, 50, 255, 255);


    for (int dy = -5; dy <= 5; ++dy) {
        for (int dx = -5; dx <= 5; ++dx) {
            int x = px + dx;
            int y = py + dy;
            if (x >= 0 && x < width && y >= 0 && y < height) {
                surf2Dwrite(color, surface, x * sizeof(uchar4), y);
            }
        }
    }
}

// dlya lovli oshibok
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}


//////// CUDA CUDA CUDA CUDA CUDA CUDA CUDA CUDA ////////////////////////




////// INIT ///////

void initCuda() {
    checkCudaError(cudaMalloc(&cudaParams.positions_x, MAX_PARTICLES * sizeof(float)), "Alloc positions X");
    checkCudaError(cudaMalloc(&cudaParams.positions_y, MAX_PARTICLES * sizeof(float)), "Alloc positions Y");
    checkCudaError(cudaMalloc(&cudaParams.velocities_x, MAX_PARTICLES * sizeof(float)), "Alloc velocities X");
    checkCudaError(cudaMalloc(&cudaParams.velocities_y, MAX_PARTICLES * sizeof(float)), "Alloc velocities Y");
    checkCudaError(cudaMalloc(&cudaParams.types, MAX_PARTICLES * sizeof(int)), "Alloc types");
    checkCudaError(cudaMalloc(&cudaParams.cellIndices, MAX_PARTICLES * sizeof(int)), "Alloc cell indices");
    checkCudaError(cudaMalloc(&cudaParams.cellStarts, GRID_SIZE * GRID_SIZE * sizeof(int)), "Alloc cell starts");
    checkCudaError(cudaMalloc(&cudaParams.cellEnds, GRID_SIZE * GRID_SIZE * sizeof(int)), "Alloc cell ends");
    checkCudaError(cudaMalloc(&cudaParams.active, MAX_PARTICLES * sizeof(int)), "Alloc active");

    cudaMemcpyToSymbol(segment_x, &h_segment_x, sizeof(float));
    cudaMemcpyToSymbol(segment_y, &h_segment_y, sizeof(float));
    cudaMemcpyToSymbol(segment_length, &h_segment_length, sizeof(float));
    cudaMemcpyToSymbol(segment_thickness, &h_segment_thickness, sizeof(float));
}



void initParticles() {
    std::vector<float> pos_x(numParticles);
    std::vector<float> pos_y(numParticles);
    std::vector<float> vel_x(numParticles);
    std::vector<float> vel_y(numParticles);
    std::vector<int> types(numParticles);

    const float spawnWidth = 0.2f; // smestili
    const float spawnHeight = 0.1f;
    const float spawnY = -0.9f;
    const float leftSpawnStart = -1.0f;
    const float leftSpawnEnd = leftSpawnStart + spawnWidth;
    const float rightSpawnStart = 1.0f - spawnWidth;
    const float rightSpawnEnd = 1.0f;

    for (int i = 0; i < numParticles; ++i) {
        types[i] = (i % 2 == 0) ? PARTICLE_TYPE_A : PARTICLE_TYPE_B;

        if (types[i] == PARTICLE_TYPE_A) {
            pos_x[i] = leftSpawnStart + static_cast<float>(rand()) / RAND_MAX * spawnWidth;
            pos_y[i] = spawnY + static_cast<float>(rand()) / RAND_MAX * spawnHeight;
            vel_x[i] = (rand() % 100) / 200.0f + 0.2f;  // 0.2 - 0.7
            vel_y[i] = (rand() % 100) / 200.0f + 0.5f;  // 0.5 - 1.0
        }
        else {
            pos_x[i] = rightSpawnStart + static_cast<float>(rand()) / RAND_MAX * spawnWidth;
            pos_y[i] = spawnY + static_cast<float>(rand()) / RAND_MAX * spawnHeight;
            vel_x[i] = -(rand() % 100) / 200.0f - 0.2f; // -0.2 - -0.7
            vel_y[i] = (rand() % 100) / 200.0f + 0.5f; // 0.5 - 1.0
        }
    }

    cudaMemcpy(cudaParams.positions_x, pos_x.data(), numParticles * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaParams.positions_y, pos_y.data(), numParticles * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaParams.velocities_x, vel_x.data(), numParticles * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaParams.velocities_y, vel_y.data(), numParticles * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaParams.types, types.data(), numParticles * sizeof(int), cudaMemcpyHostToDevice);

    std::vector<int> activeHost(numParticles, 1);

    cudaMemcpy(cudaParams.active, activeHost.data(), numParticles * sizeof(int), cudaMemcpyHostToDevice);

}


void initBasketParams() {

    const float h_basket_left = 0.3f;
    const float h_basket_right = 0.7f;
    const float h_basket_bottom = 0.4f;
    const float h_basket_top = 0.5f;
    const float h_basket_thickness = 0.005f;

    // copy to videokarta
    cudaMemcpyToSymbol(basket_left, &h_basket_left, sizeof(float));
    cudaMemcpyToSymbol(basket_right, &h_basket_right, sizeof(float));
    cudaMemcpyToSymbol(basket_bottom, &h_basket_bottom, sizeof(float));
    cudaMemcpyToSymbol(basket_top, &h_basket_top, sizeof(float));
    cudaMemcpyToSymbol(basket_thickness, &h_basket_thickness, sizeof(float));

    const float h_particle_radius = 0.02f;
    cudaMemcpyToSymbol(particle_radius, &h_particle_radius, sizeof(float));

    const float h_gravity = -0.4f;
    cudaMemcpyToSymbol(gravity, &h_gravity, sizeof(float));

    const float h_max_speed = 1.5f;
    cudaMemcpyToSymbol(max_speed, &h_max_speed, sizeof(float));


    checkCudaError(cudaGetLastError(), "Basket params copy to device");
}

////// INIT ///////






///// RENDERING  ////////////////

void renderText(const std::string& text, float x, float y) {
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, winWidth, 0, winHeight);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glColor3f(1.0, 1.0, 1.0);
    glRasterPos2f(x, y);

    for (const char& c : text) {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, c);
    }

    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}


float actual_segment_x = h_segment_x;
float actual_segment_y = h_segment_y;

void display() {

    int currentTime = glutGet(GLUT_ELAPSED_TIME);
    float elapsedSeconds = (currentTime - startTime) / 1000.0f;
    float remainingTime = GAME_SECONDS - elapsedSeconds;

    if (!gameOver && !gameWon) {
        if (remainingTime <= 0) {
            gameOver = true;
        }
        else if (capturedParticles >= 50) {
            gameWon = true;
        }
    }

    if (!gameOver && !gameWon) {

        actual_segment_x += (h_segment_x - actual_segment_x) * deltaTime * 10.0f;
        actual_segment_y += (h_segment_y - actual_segment_y) * deltaTime * 10.0f;
        cudaMemcpyToSymbol(segment_x, &actual_segment_x, sizeof(float));
        cudaMemcpyToSymbol(segment_y, &actual_segment_y, sizeof(float));

        dim3 block(BLOCK_SIZE);
        dim3 grid((numParticles + BLOCK_SIZE - 1) / BLOCK_SIZE);
        
        // upldate
        updateParticles << <grid, block >> > (cudaParams.positions_x, cudaParams.positions_y, cudaParams.velocities_x, cudaParams.velocities_y, numParticles, deltaTime, cudaParams.active);
        cudaDeviceSynchronize();
        checkCudaError(cudaGetLastError(), "Update particles kernel");

        // sort po grid
        sortParticles << <grid, block >> > (cudaParams.positions_x, cudaParams.positions_y, cudaParams.cellIndices, numParticles);
        cudaDeviceSynchronize();
        checkCudaError(cudaGetLastError(), "Sort particles kernel");

        // trust
        thrust::sort(
            thrust::device_ptr<int>(cudaParams.cellIndices),
            thrust::device_ptr<int>(cudaParams.cellIndices + numParticles)
        );

        // init grid
        cudaMemset(cudaParams.cellStarts, 0xFF, GRID_SIZE * GRID_SIZE * sizeof(int));
        cudaMemset(cudaParams.cellEnds, 0xFF, GRID_SIZE * GRID_SIZE * sizeof(int));

        // stroim grid
        dim3 setupBlock(256);
        dim3 gridSetup((numParticles + setupBlock.x - 1) / setupBlock.x);
        setupGrid << <gridSetup, setupBlock >> > (
            cudaParams.cellIndices,
            cudaParams.cellStarts,
            cudaParams.cellEnds,
            numParticles
            );
        cudaDeviceSynchronize();
        checkCudaError(cudaGetLastError(), "Setup grid kernel");

        // Prik-Skok blok
        processBlockPrikSkok << <grid, block >> > (cudaParams.positions_x, cudaParams.positions_y, cudaParams.velocities_x, cudaParams.velocities_y, cudaParams.types, numParticles, min_distance);
        cudaDeviceSynchronize();
        
        // prik skok grid
        processGridPrikSkok << <grid, block >> > ( cudaParams.positions_x, cudaParams.positions_y,cudaParams.velocities_x,
            cudaParams.velocities_y,
            cudaParams.types,
            cudaParams.cellStarts,
            cudaParams.cellEnds,
            cudaParams.cellIndices,
            numParticles,
            min_distance
            );
        cudaDeviceSynchronize();
        checkCudaError(cudaGetLastError(), "Collision processing kernel");

        // delete
        thrust::device_ptr<float> d_pos_x(cudaParams.positions_x);
        thrust::device_ptr<float> d_pos_y(cudaParams.positions_y);
        thrust::device_ptr<float> d_vel_x(cudaParams.velocities_x);
        thrust::device_ptr<float> d_vel_y(cudaParams.velocities_y);
        thrust::device_ptr<int> d_types(cudaParams.types);
        thrust::device_ptr<int> d_active(cudaParams.active);

        auto begin = thrust::make_zip_iterator(
            thrust::make_tuple(d_pos_x, d_pos_y, d_vel_x, d_vel_y, d_types, d_active) //ediniy array from many arrays and parameters
        );
        auto end = begin + numParticles;

        // nado vse active 0 -> v konec
        auto new_end = thrust::remove_if(
            thrust::cuda::par,
            begin,
            end,
            [] __device__(const thrust::tuple<float, float, float, float, int, int>&t) {
            return thrust::get<5>(t) == 0; // 5-iy element - active
        }
        );
        numParticles = new_end - begin;

        // recreate grid
        dim3 blockSort(BLOCK_SIZE);
        dim3 gridSort((numParticles + BLOCK_SIZE - 1) / BLOCK_SIZE);

        sortParticles << <gridSort, blockSort >> > (
            cudaParams.positions_x,
            cudaParams.positions_y,
            cudaParams.cellIndices,
            numParticles
            );
        cudaDeviceSynchronize();

        thrust::sort(
            thrust::device_ptr<int>(cudaParams.cellIndices),
            thrust::device_ptr<int>(cudaParams.cellIndices + numParticles)
        );


        cudaGraphicsMapResources(1, &cudaParams.cudaResource, 0);
        cudaArray* array;
        cudaGraphicsSubResourceGetMappedArray(&array, cudaParams.cudaResource, 0, 0);

        cudaResourceDesc resDesc = {};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = array;
        cudaSurfaceObject_t surface;
        cudaCreateSurfaceObject(&surface, &resDesc);


        dim3 clearBlock(32, 32);
        dim3 clearGrid(
            (winWidth + clearBlock.x - 1) / clearBlock.x,
            (winHeight + clearBlock.y - 1) / clearBlock.y
        );
        ClearTexture << <clearGrid, clearBlock >> > (surface, winWidth, winHeight);
        cudaDeviceSynchronize();


        drawParticles << <grid, block >> > (
            surface,
            cudaParams.positions_x,
            cudaParams.positions_y,
            cudaParams.types,
            numParticles,
            winWidth,
            winHeight
            );
        cudaDeviceSynchronize();


        cudaDestroySurfaceObject(surface);
        cudaGraphicsUnmapResources(1, &cudaParams.cudaResource, 0);


        glClear(GL_COLOR_BUFFER_BIT);
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, cudaParams.texture);

        glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
        glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, -1.0f);
        glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, 1.0f);
        glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, 1.0f);
        glEnd();

        glDisable(GL_TEXTURE_2D);

        cudaMemcpyFromSymbol(&capturedParticles, countCaptured, sizeof(int));
    }


    std::stringstream ss;
    ss << "Poymano: " << capturedParticles << " /50";
    renderText(ss.str(), 20, winHeight - 40);

    ss.str("");
    ss << "Time: " << (remainingTime > 0 ? static_cast<int>(remainingTime) : 0) << " sec";
    renderText(ss.str(), 20, winHeight - 70);

    if (gameOver) {
        renderText("Game Over! Ne poymano 50 za 15 sec.", winWidth / 2 - 150, winHeight / 2);
    }
    else if (gameWon) {
        renderText("Pobeda! 50 poymano!", winWidth / 2 - 100, winHeight / 2);

    }

    if ((gameOver || gameWon) && !timerSet) {
        timerSet = true;
        glutTimerFunc(3500, closeWindow, 0);
    }

    glutSwapBuffers();

    //int captured;
    //cudaMemcpyFromSymbol(&captured, countCaptured, sizeof(int));
    //std::cout << "Particles in basket: " << captured << std::endl;
}

///// RENDERING  ////////////////

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitWindowSize(winWidth, winHeight);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    windowID = glutCreateWindow("GPU Particle System");

    glewInit();



    glGenTextures(1, &cudaParams.texture);
    glBindTexture(GL_TEXTURE_2D, cudaParams.texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, winWidth, winHeight, 0,
        GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);



    cudaError_t err = cudaGraphicsGLRegisterImage(&cudaParams.cudaResource,
        cudaParams.texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
    checkCudaError(err, "Register OpenGL texture");

    initCuda();
    initBasketParams();


    // pryamougolnik
    cudaMemcpyToSymbol(rect_left, &h_rect_left, sizeof(float));
    cudaMemcpyToSymbol(rect_right, &h_rect_right, sizeof(float));
    cudaMemcpyToSymbol(rect_bottom, &h_rect_bottom, sizeof(float));
    cudaMemcpyToSymbol(rect_top, &h_rect_top, sizeof(float));
    cudaMemcpyToSymbol(rect_exists, &h_rect_exists, sizeof(bool));

    int zero = 0;
    cudaMemcpyToSymbol(countCaptured, &zero, sizeof(int));
    startTime = glutGet(GLUT_ELAPSED_TIME);
    initParticles();
    glutKeyboardFunc(keyboard);
    glutDisplayFunc(display);
    glutIdleFunc([]() {
        if (!gameOver && !gameWon) {
            glutPostRedisplay();
        }
        });
    glutMainLoop();



    cudaFree(cudaParams.positions_x);
    cudaFree(cudaParams.positions_y);
    cudaFree(cudaParams.velocities_x);
    cudaFree(cudaParams.velocities_y);
    cudaFree(cudaParams.types);

    cudaFree(cudaParams.cellIndices);
    cudaFree(cudaParams.cellStarts);
    cudaFree(cudaParams.cellEnds);
    cudaGraphicsUnregisterResource(cudaParams.cudaResource);
    glDeleteTextures(1, &cudaParams.texture);

    return 0;
}