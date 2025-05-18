#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include "particle.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
using namespace std;


__constant__ float d_world_size = 2.0f;
const float min_distance = 0.05f;
int numParticles = 256;
int winWidth = 1024, winHeight = 1024;
CudaParams cudaParams;
float deltaTime = 0.016f;

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

    // start yacheyki
    if (idx == 0 || currentHash != (cellIndices[idx - 1] >> 16)) {
        cellStarts[currentHash] = idx; 
    }

    // end_yacheyki
    if (idx == numParticles - 1 || currentHash != (cellIndices[idx + 1] >> 16)) {
        cellEnds[currentHash] = idx + 1; 
    }
}

__global__ void updateParticles(float* positions_x, float* positions_y,
    float* velocities_x, float* velocities_y,
    int numParticles, float deltaTime) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    // update pos
    positions_x[idx] += velocities_x[idx] * deltaTime;
    positions_y[idx] += velocities_y[idx] * deltaTime;

    // prik-skok ot kraev
    if (fabsf(positions_x[idx]) >= 1.0f) velocities_x[idx] *= -1.0f;
    if (fabsf(positions_y[idx]) >= 1.0f) velocities_y[idx] *= -1.0f;
}

__global__ void processBlockPrikSkok(
    float* positions_x,
    float* positions_y,
    float* velocities_x,
    float* velocities_y,
    int* types,
    int numParticles,
    float min_dist
) {
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

    if (idx < numParticles) {
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

    
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        const int type2 = shared.type[i];
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
    //ot greha podalshe
    atomicAdd(&velocities_x[idx], delta_vx);
    atomicAdd(&velocities_y[idx], delta_vy);
    atomicAdd(&positions_x[idx], delta_x);
    atomicAdd(&positions_y[idx], delta_y);
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

                    const float impulse = -1.0f * relVel;
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

    atomicAdd(&velocities_x[idx], delta_vx);
    atomicAdd(&velocities_y[idx], delta_vy);
    atomicAdd(&positions_x[idx], delta_x);
    atomicAdd(&positions_y[idx], delta_y);
}


__global__ void ClearTexture(cudaSurfaceObject_t surface, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        surf2Dwrite(make_uchar4(0, 0, 0, 255), surface, x * sizeof(uchar4), y);
    }
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


    for (int dy = -10; dy <= 10; ++dy) {
        for (int dx = -10; dx <= 10; ++dx) {
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

void initCuda() {
    checkCudaError(cudaMalloc(&cudaParams.positions_x, MAX_PARTICLES * sizeof(float)), "Alloc positions X");
    checkCudaError(cudaMalloc(&cudaParams.positions_y, MAX_PARTICLES * sizeof(float)), "Alloc positions Y");
    checkCudaError(cudaMalloc(&cudaParams.velocities_x, MAX_PARTICLES * sizeof(float)), "Alloc velocities X");
    checkCudaError(cudaMalloc(&cudaParams.velocities_y, MAX_PARTICLES * sizeof(float)), "Alloc velocities Y");
    checkCudaError(cudaMalloc(&cudaParams.types, MAX_PARTICLES * sizeof(int)), "Alloc types");
    checkCudaError(cudaMalloc(&cudaParams.cellIndices, MAX_PARTICLES * sizeof(int)), "Alloc cell indices");
    checkCudaError(cudaMalloc(&cudaParams.cellStarts, GRID_SIZE * GRID_SIZE * sizeof(int)), "Alloc cell starts");
    checkCudaError(cudaMalloc(&cudaParams.cellEnds, GRID_SIZE * GRID_SIZE * sizeof(int)), "Alloc cell ends");

}


void initParticles() {
    std::vector<float> pos_x(numParticles);
    std::vector<float> pos_y(numParticles);
    std::vector<float> vel_x(numParticles);
    std::vector<float> vel_y(numParticles);
    std::vector<int> types(numParticles);

    for (int i = 0; i < numParticles; ++i) {
        pos_x[i] = (rand() % 2000) / 1000.0f - 1.0f;
        pos_y[i] = (rand() % 2000) / 1000.0f - 1.0f;
        vel_x[i] = (rand() % 100 - 100) / 500.0f;
        vel_y[i] = (rand() % 100 - 100) / 500.0f;
        types[i] = rand() % 2;
    }

    cudaMemcpy(cudaParams.positions_x, pos_x.data(), numParticles * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaParams.positions_y, pos_y.data(), numParticles * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaParams.velocities_x, vel_x.data(), numParticles * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaParams.velocities_y, vel_y.data(), numParticles * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaParams.types, types.data(), numParticles * sizeof(int), cudaMemcpyHostToDevice);
}

void display() {
    
    dim3 block(BLOCK_SIZE);
    dim3 grid((numParticles + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // upldate
    updateParticles << <grid, block >> > (
        cudaParams.positions_x,
        cudaParams.positions_y,
        cudaParams.velocities_x,
        cudaParams.velocities_y,
        numParticles,
        deltaTime
        );
    cudaDeviceSynchronize();
    checkCudaError(cudaGetLastError(), "Update particles kernel");

    // sort po grid
    sortParticles << <grid, block >> > (
        cudaParams.positions_x,
        cudaParams.positions_y,
        cudaParams.cellIndices,
        numParticles
        );
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
    processBlockPrikSkok << <grid, block >> > (
        cudaParams.positions_x,
        cudaParams.positions_y,
        cudaParams.velocities_x,
        cudaParams.velocities_y,
        cudaParams.types,
        numParticles,
        min_distance
        );
    cudaDeviceSynchronize();
    // prik skok grid
    processGridPrikSkok << <grid, block >> > (
        cudaParams.positions_x,
        cudaParams.positions_y,
        cudaParams.velocities_x,
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
    checkCudaError(cudaGetLastError(), "Clear texture kernel");

    
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
    checkCudaError(cudaGetLastError(), "Draw particles kernel");

    
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
    glutSwapBuffers();

    

}

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitWindowSize(winWidth, winHeight);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutCreateWindow("GPU Particle System");

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
    initParticles();

    glutDisplayFunc(display);
    glutIdleFunc([]() { glutPostRedisplay(); });
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
