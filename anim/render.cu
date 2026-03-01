// #define GL_GLEXT_PROTOTYPES

#include <cuda.h>
#include <GL/gl.h>
#include <GL/glx.h>
#include <GL/glext.h>
#include <GL/glut.h>
#include <GL/freeglut_ucall.h>
#include <vector>
#include "board_manager.h"
#include "utils.h"

typedef void (*GLDEBUGPROC)(
    GLenum source,GLenum type,GLuint id,GLenum severity,GLsizei length,const GLchar *message,
    const void *userParam);
typedef void (*PFNGLDEBUGMESSAGECALLBACKPROC)(GLDEBUGPROC callback, const void * userParam);

void messageCallback(
    GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length,
    const GLchar* message, const void* userParam
) {
    fprintf(stderr, "GL CALLBACK: %s type = 0x%x, severity = 0x%x, message = %s\n",
        (type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : ""), type, severity, message);
}

void enableDebug() {
    GLubyte name[] = "glDebugMessageCallback";
    PFNGLDEBUGMESSAGECALLBACKPROC glDebugMessageCallback =
        (PFNGLDEBUGMESSAGECALLBACKPROC)glXGetProcAddress(name);
    if (glDebugMessageCallback) {
        glEnable(GL_DEBUG_OUTPUT);
        glDebugMessageCallback(messageCallback, 0);
    } else {
        fprintf(stderr, "warning: glDebugMessageCallback not found\n");
    }
}

class Renderer {
    BoardManager *boardManager;
    Size size;
    GLuint texId;
    Board<Pixel> h_tex;

    // For FPS
    int stepsCount;
    int frameCount;
    int timeBase;

public:
    Renderer(Size size, BoardManager *boardManager);
    ~Renderer();
    void initializeBoard();
    void stepBoard();
    void updateTexture();
    void render();
    void updateFPS();
};

Renderer::Renderer(Size size, BoardManager *boardManager) :
    boardManager(boardManager), size(size)
{
    CUDA_CHECK(cudaMallocHost(&h_tex.ptr, size.totalBytes<Pixel>()));
    h_tex.pitch = size.rowBytes<Pixel>();

    glGenTextures(1, &texId);
    glBindTexture(GL_TEXTURE_2D, texId);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGBA8, size.width, size.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);

    stepsCount = 0;
    frameCount = 0;
    timeBase = glutGet(GLUT_ELAPSED_TIME);
}

Renderer::~Renderer() {
    glDeleteTextures(1, &texId);
    CUDA_CHECK(cudaFreeHost(h_tex.ptr));
}

void Renderer::initializeBoard() {
    boardManager->initialize();
}

void Renderer::stepBoard() {
    boardManager->step();
    stepsCount++;
}

void Renderer::updateTexture() {
    boardManager->writeTexture(h_tex);
    glBindTexture(GL_TEXTURE_2D, texId);
    glTexSubImage2D(
        GL_TEXTURE_2D, 0, 0, 0, size.width, size.height, GL_RGBA, GL_UNSIGNED_BYTE, h_tex.ptr);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void Renderer::render() {
    glClear(GL_COLOR_BUFFER_BIT);

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, texId);
    glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, -1.0f);
        glTexCoord2f(1.0f, 1.0f); glVertex2f( 1.0f, -1.0f);
        glTexCoord2f(1.0f, 0.0f); glVertex2f( 1.0f,  1.0f);
        glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f,  1.0f);
    glEnd();
    glDisable(GL_TEXTURE_2D);
    glutSwapBuffers();
    frameCount++;
}

void Renderer::updateFPS() {
    int time = glutGet(GLUT_ELAPSED_TIME);
    if (time - timeBase < 1000) {
        return;
    }
    float sps = stepsCount * 1000.0f / (time - timeBase);
    float fps = frameCount * 1000.0f / (time - timeBase);
    stepsCount = 0;
    frameCount = 0;
    timeBase = time;

    char title[80];
    snprintf(title, sizeof title, "anim: %4.2f steps/s, %4.2f frame/s", sps, fps);
    glutSetWindowTitle(title);
}

void idle(void *cbData) {
    Renderer *renderer = static_cast<Renderer *>(cbData);
    renderer->stepBoard();
}

void timer(int value, void *cbData) {
    Renderer *renderer = static_cast<Renderer *>(cbData);
    renderer->updateTexture();
    glutPostRedisplay();
    glutTimerFuncUcall(1000 / 60, timer, 0, cbData);
}

void display(void *cbData) {
    Renderer *renderer = static_cast<Renderer *>(cbData);
    renderer->render();
    renderer->updateFPS();
}

#if !defined(BM_DBG_MAIN)
int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(1000, 1000);
    glutCreateWindow("anim");
    enableDebug();

    Size size = {1000, 1000};
    BoardManager bm {size};
    Renderer renderer {size, &bm};
    renderer.initializeBoard();
    renderer.updateTexture();

    glutIdleFuncUcall(idle, &renderer);
    timer(0, &renderer);
    glutDisplayFuncUcall(display, &renderer);
    glutMainLoop();
    return 0;
}
#endif
