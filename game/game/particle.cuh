#pragma once
#include <cstdint>

#define PARTICLE_TYPE_A 0
#define PARTICLE_TYPE_B 1

struct Particle {
    float x, y;
    float vx, vy;
    int type;
};

constexpr int MAX_PARTICLES = 1024;
constexpr float PARTICLE_RADIUS = 0.04f;
constexpr float PARTICLE_SIZE = 16.0f;