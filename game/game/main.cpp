#include <GL/freeglut.h>
#include <vector>
#include <cstdlib>

struct Particle {
    float x, y;
    float vx, vy;
    float r, g, b;
};

const int NUM_PARTICLES = 100;
std::vector<Particle> particles;

// init chasticy
void initParticles() {
    particles.resize(NUM_PARTICLES);
    for (auto& p : particles) {
 
        p.x = (rand() % 2000 - 1000) / 1000.0f;
        p.y = (rand() % 2000 - 1000) / 1000.0f;
        p.vx = (rand() % 100 - 50) / 500.0f;
        p.vy = (rand() % 100 - 50) / 500.0f;
        p.r = rand() % 100 / 100.0f;
        p.g = rand() % 100 / 100.0f;
        p.b = rand() % 100 / 100.0f;
    }
}

// +coord x +coord y, proverka na predely ekrana
void updateParticles() {
    for (auto& p : particles) {
        p.x += p.vx;
        p.y += p.vy;
        if (p.x < -1.0f || p.x > 1.0f) p.vx *= -1;
        if (p.y < -1.0f || p.y > 1.0f) p.vy *= -1;
    }
}

// risovalka
void display() {
    glClear(GL_COLOR_BUFFER_BIT);

    glPointSize(10.0f);
    glBegin(GL_POINTS);
    for (const auto& p : particles) {
        glColor3f(p.r, p.g, p.b);
        glVertex2f(p.x, p.y);
    }
    glEnd();

    glutSwapBuffers();
}

// FPS
void timer(int) {
    updateParticles();
    glutPostRedisplay();
    glutTimerFunc(16, timer, 0);
}


int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitWindowSize(512, 512);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutCreateWindow("Particle System");

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    initParticles();

    glutDisplayFunc(display);
    glutTimerFunc(0, timer, 0);

    glutMainLoop();
    return 0;
}