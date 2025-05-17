#include <GL/freeglut.h>
#include <vector>
#include <cstdlib>

struct Particle {
    float x, y;
    float vx, vy;
    int type;
};

const int NUM_PARTICLES = 8;
std::vector<Particle> particles;



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

    // uprugo: delta_p = 2 x delta_V_norm; delta_V_norm = n_vector x delta_V_vector;
    // delta_V_vector = ( V1_vector - V_2_vector ) = ( V_1_x - V_2_x + V_1_y - V_2_y );
    // n_vector = n_x*i + n_y*j;
    // n_x(or y) = dx(or y) / distance_between_centers;
    // delta_V_norm = n_x*(V_1_x - V_2_x) + n_y*(V_1_y - V_2_y);
    // togda delta_p = 2 x [ n_x*(V_1_x - V_2_x) + n_y*(V_1_y - V_2_y) ];
    // V_1_x_final = V_1_x + delta_p x n_x, V_1_y_final = V_1_y + delta_p x n_y;
    // V_2_x_final = V_2_x - delta_p x n_x, V_2_y_final = V_2_y - delta_p x n_y;

        // rasstoyaniye
    float dx = particle_1.x - particle_2.x;
    float dy = particle_1.y - particle_2.y;

    float distance_between_centers = sqrtf(dx * dx + dy * dy);

    float n_x = dx / distance_between_centers;
    float n_y = dy / distance_between_centers;

    //float delta_p = 2.0 * (n_x * (particle_1.vx - particle_2.vx) + n_y * (particle_1.vy - particle_2.vy));

    ///float delta_p;
    //float delta_V_norm;

    float delta_V_norm_x = (particle_1.vx - particle_2.vx);
    float delta_V_norm_y = (particle_1.vy - particle_2.vy);

    float delta_V_norm = n_x*delta_V_norm_x + n_y*delta_V_norm_y;
    //delta_p = delta_V_norm;



    //float _vremennaya_v = particle_1.vx;

    if (delta_V_norm < 0) {

        float imp = -2.0f * delta_V_norm;

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

            float min_distance = 0.05f;

            if (particle_1.type == particle_2.type) continue;

            float dx = particle_1.x - particle_2.x;
            float dy = particle_1.y - particle_2.y;

            float distance = sqrtf(dx * dx + dy * dy);

            if (distance < min_distance) {
                ottalkivanie_dvuh(particle_1, particle_2);
            }
        }
    }
}


// +coord x +coord y, proverka na predely ekrana
void updateParticles() {

    // ot kraev
    for (auto& p : particles) {
        p.x += p.vx;
        p.y += p.vy;
        if (p.x < -1.0f || p.x > 1.0f) p.vx *= -1;
        if (p.y < -1.0f || p.y > 1.0f) p.vy *= -1;
    }

    // mezghdu soboy
    many_ootalkivaniya();

}

// risovalka
void display() {
    glClear(GL_COLOR_BUFFER_BIT);

    glPointSize(40.0f);
    glBegin(GL_POINTS);

    for (const auto& p : particles) {

        //type to RGB
        if (p.type == 0) {
            glColor3f(1.0f, 0.0f, 0.0f);
        }
        else glColor3f(0.0f, 0.0f, 1.0f);

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