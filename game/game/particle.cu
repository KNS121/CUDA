#include <GL/glew.h>
#include "particle.cuh"
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>
#include <surface_functions.h>

#include <cstdio>

struct CudaResources {
    cudaGraphicsResource* cudaResource = nullptr;
    cudaSurfaceObject_t surface = 0;
    cudaArray* array = nullptr;

    GLuint texture = 0;
    GLuint fbo = 0;
} cudaRes;

Particle particles[MAX_PARTICLES];
int numParticles = 0;
int winWidth = 720, winHeight = 720;

Particle* d_particles = nullptr;

float deltaTime = 0.016f;

__global__ void DrawParticles(cudaSurfaceObject_t surface, Particle* particles,
    int numParticles, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    Particle p = particles[idx];
    int px = static_cast<int>((p.x + 1.0f) * 0.5f * width);
    int py = static_cast<int>((p.y + 1.0f) * 0.5f * height);

    uchar4 color = (p.type == PARTICLE_TYPE_A) ? make_uchar4(255, 0, 0, 255) : make_uchar4(0, 0, 255, 255);

    for (int y = -PARTICLE_SIZE / 2; y < PARTICLE_SIZE / 2; ++y) {
        for (int x = -PARTICLE_SIZE / 2; x < PARTICLE_SIZE / 2; ++x) {
            int xpos = px + x;
            int ypos = py + y;
            if (xpos >= 0 && xpos < width && ypos >= 0 && ypos < height) {
                surf2Dwrite(color, surface, xpos * sizeof(uchar4), ypos);
            }
        }
    }
}

__global__ void ClearTexture(cudaSurfaceObject_t surface, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        surf2Dwrite(make_uchar4(0, 0, 0, 255), surface, x * sizeof(uchar4), y); // ярко-красный
    }
}


__global__ void UpdateParticles(Particle* particles, int numParticles, float deltaTime) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    particles[idx].x += particles[idx].vx * deltaTime;
    particles[idx].y += particles[idx].vy * deltaTime;


    if (particles[idx].x < -1.0f || particles[idx].x > 1.0f) particles[idx].vx *= -1;
    if (particles[idx].y < -1.0f || particles[idx].y > 1.0f) particles[idx].vy *= -1;
}



void initGL(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitWindowSize(winWidth, winHeight);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutCreateWindow("CUDA Particle System");


    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        fprintf(stderr, "GLEW Error: %s\n", glewGetErrorString(err));
        exit(1);
    }

    //TEX
    glGenTextures(1, &cudaRes.texture);
    glBindTexture(GL_TEXTURE_2D, cudaRes.texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, winWidth, winHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // Cuda registraciya
    cudaError_t cudaErr = cudaGraphicsGLRegisterImage(&cudaRes.cudaResource, cudaRes.texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
    if (cudaErr != cudaSuccess) {
        fprintf(stderr, "CUDA Register Error: %s\n", cudaGetErrorString(cudaErr));
        exit(1);
    }

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
}

void initParticles() {
    numParticles = 8;
    for (int i = 0; i < numParticles; ++i) {
        particles[i].x = (rand() % 1000) / 500.0f - 1.0f;
        particles[i].y = (rand() % 1000) / 500.0f - 1.0f;
        particles[i].type = rand() % 2;
        particles[i].vx = particles[i].vy = (rand() % 100 - 100) / 100.0f;
    }

    cudaMalloc((void**)&d_particles, MAX_PARTICLES * sizeof(Particle));
    cudaMemcpy(d_particles, particles, MAX_PARTICLES * sizeof(Particle), cudaMemcpyHostToDevice);
}

void display() {
    // map
    cudaGraphicsMapResources(1, &cudaRes.cudaResource, 0);
    cudaGraphicsSubResourceGetMappedArray(&cudaRes.array, cudaRes.cudaResource, 0, 0);

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cudaRes.array;
    cudaCreateSurfaceObject(&cudaRes.surface, &resDesc);

    dim3 blockSize(256);
    dim3 gridSize((numParticles + blockSize.x - 1) / blockSize.x);

    UpdateParticles <<<gridSize, blockSize>>>(d_particles, numParticles, deltaTime);
    cudaDeviceSynchronize();
    // chistim
    dim3 clearBlocks(32, 32);
    dim3 clearGrid((winWidth + 31) / 32, (winHeight + 31) / 32);
    ClearTexture <<<clearGrid, clearBlocks>>>(cudaRes.surface, winWidth, winHeight);
    cudaDeviceSynchronize();

    // risovlaka
    DrawParticles <<<gridSize, blockSize>>>(cudaRes.surface, d_particles, numParticles, winWidth, winHeight);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();



    // osvobodim resursy
    cudaDestroySurfaceObject(cudaRes.surface);
    cudaGraphicsUnmapResources(1, &cudaRes.cudaResource, 0);

    // draw tex
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, cudaRes.texture);

    glBegin(GL_QUADS);
    glTexCoord2f(0, 1); glVertex2f(-1, -1);
    glTexCoord2f(1, 1); glVertex2f(1, -1);
    glTexCoord2f(1, 0); glVertex2f(1, 1);
    glTexCoord2f(0, 0); glVertex2f(-1, 1);
    glEnd();

    glutSwapBuffers();
}

void timer(int) {
    glutPostRedisplay();
    glutTimerFunc(16, timer, 0);
}

int main(int argc, char** argv) {
    initGL(argc, argv);
    initParticles();

    glutDisplayFunc(display);
    glutTimerFunc(0, timer, 0);
    glutMainLoop();

    // osvobodim resursy
    cudaGraphicsUnregisterResource(cudaRes.cudaResource);
    glDeleteTextures(1, &cudaRes.texture);

    return 0;
}