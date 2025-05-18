#include <GL/freeglut.h>
#include <vector>
#include <cstdlib>


struct Particle {
    float x, y;
    float vx, vy;
    int type;
};

const float PARTICLE_RADIUS = 0.04f;
const int NUM_PARTICLES = 64;
std::vector<Particle> particles;

const float min_distance = 2 * PARTICLE_RADIUS;


// init chasticy
void initParticles() {
    particles.resize(NUM_PARTICLES);
    for (auto& p : particles) {

        p.x = (rand() % 2000 - 1000) / 1000.0f;
        p.y = (rand() % 2000 - 1000) / 1000.0f;
        p.vx = (rand() % 100 - 100) / 10000.0f;
        p.vy = (rand() % 100 - 100) / 10000.0f;
        p.type = rand() % 2;

    }
}


void ottalkivanie_dvuh(Particle& particle_1, Particle& particle_2) {


    float dx = particle_1.x - particle_2.x;
    float dy = particle_1.y - particle_2.y;

    float distance_between_centers = sqrtf(dx * dx + dy * dy);

    if (distance_between_centers < min_distance && distance_between_centers > 0) {

        float n_x = dx / distance_between_centers;
        float n_y = dy / distance_between_centers;


        float overlap = 0.2f * (min_distance - distance_between_centers);

        particle_1.x -= overlap * n_x;
        particle_1.y -= overlap * n_y;
        particle_2.x += overlap * n_x;
        particle_2.y += overlap * n_y;

        float delta_V_norm_x = (particle_1.vx - particle_2.vx);
        float delta_V_norm_y = (particle_1.vy - particle_2.vy);

        float delta_V_norm = n_x * delta_V_norm_x + n_y * delta_V_norm_y;

        if (delta_V_norm > 0) return;

        float imp = -1.0f * delta_V_norm;

        particle_1.vx += imp * n_x;
        particle_1.vy += imp * n_y;
        particle_2.vx -= imp * n_x;
        particle_2.vy -= imp * n_y;


    }
}


void many_ootalkivaniya() {


    for (size_t i = 0; i < particles.size(); ++i) {
        for (size_t j = i + 1; j < particles.size(); ++j) {

            Particle& particle_1 = particles[i];
            Particle& particle_2 = particles[j];

            if (particle_1.type == particle_2.type) continue;

            ottalkivanie_dvuh(particle_1, particle_2);

        }
    }
}




void drawCircle(float x, float y, float r) {
    glBegin(GL_TRIANGLE_FAN);
    glVertex2f(x, y);
    for (int i = 0; i <= 32; ++i) {
        float angle = 2.0f * 3.1415 * i / 32;
        glVertex2f(x + cos(angle) * r, y + sin(angle) * r);
    }
    glEnd();
}




// +coord x +coord y, proverka na predely ekrana
void updateParticles() {

    // ot kraev
    for (auto& p : particles) {
        p.x += p.vx;
        p.y += p.vy;
        if (p.x - PARTICLE_RADIUS < -1.0f || p.x + PARTICLE_RADIUS > 1.0f) p.vx *= -1;
        if (p.y - PARTICLE_RADIUS < -1.0f || p.y + PARTICLE_RADIUS > 1.0f) p.vy *= -1;
    }

    // mezghdu soboy
    many_ootalkivaniya();

}

// risovalka
void display() {
    glClear(GL_COLOR_BUFFER_BIT);

    //glPointSize(40.0f);
    //glBegin(GL_POINTS);

    for (const auto& p : particles) {

        //type to RGB
        if (p.type == 0) {
            glColor3f(1.0f, 0.0f, 0.0f);
        }
        else glColor3f(0.0f, 0.0f, 1.0f);

        drawCircle(p.x, p.y, PARTICLE_RADIUS);
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
    glutInitWindowSize(720, 720);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutCreateWindow("Particle System");

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    initParticles();

    glutDisplayFunc(display);
    glutTimerFunc(0, timer, 0);

    glutMainLoop();
    return 0;
}
