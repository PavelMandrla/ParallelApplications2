#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>

#include <fstream>
#include <regex>
#include <string>
#include <algorithm>
#include <random>

#include <cudaDefs.h>
#include <FreeImage.h>
#include <imageManager.h>

using namespace std;

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

std::mt19937 generator(123);

struct GLData {
    unsigned int imageWidth;
    unsigned int imageHeight;
    unsigned int imageBPP;
    unsigned int imagePitch;

    unsigned int pboID;
    unsigned int textureID;
    unsigned int viewportWidth = 1024;
    unsigned int viewportHeight = 1024;
};

struct CudaData {
    cudaTextureDesc			texDesc;				// Texture descriptor used to describe texture parameters

    cudaArray_t				texArrayData;			// Source texture data
    cudaResourceDesc		resDesc;				// A resource descriptor for obtaining the texture data
    cudaChannelFormatDesc	texChannelDesc;			// Texture channel descriptor to define channel bytes
    cudaTextureObject_t		texObj;					// Cuda Texture Object to be produces

    cudaGraphicsResource_t  texResource;
    cudaGraphicsResource_t	pboResource;

    CudaData() {
        memset(this, 0, sizeof(CudaData));			// DO NOT DELETE THIS !!!
    }
};

struct Settings {
    int leaders;
    int followers;
    string heightMap;
    int heightmapGridX;
    int heightmapGridY;
    float leaderRadius;
    float speedFactor;
    string outputFile;

    string getAtribVal(string &str, const string& atribName) {
        smatch m;
        regex rt(atribName + "\":([\\S\\s]+?(?=,|}))");
        regex_search(str, m, rt);
        string val = m.str().substr(atribName.size()+2);
        if (val.at(0) == '\"') {
            val = val.substr(1, val.size()-2);
        }
        return val;
    }

    Settings() = default;

    Settings(const string& path) {
        fstream inStream(path);
        std::string str((std::istreambuf_iterator<char>(inStream)),std::istreambuf_iterator<char>());
        str.erase(remove_if(str.begin(), str.end(), ::isspace), str.end());

        this->leaders = std::stoi(getAtribVal(str, "leaders"));
        this->followers = std::stoi(getAtribVal(str, "followers"));
        this->heightmapGridX = std::stoi(getAtribVal(str, "heightmapGridX"));
        this->heightmapGridY = std::stoi(getAtribVal(str, "heightmapGridY"));
        this->leaderRadius = std::stof(getAtribVal(str, "leaderRadius"));
        this->speedFactor = std::stof(getAtribVal(str, "speedFactor"));
        this->heightMap = getAtribVal(str, "heightmap");
        this->outputFile = getAtribVal(str, "outputFile");
    }
};
Settings settings;

struct HeightMap {
    GLData glData;
    CudaData cudaData;

    void prepareGlObjects(const char* imageFileName) {  //alokovani zdroju na karte
        FIBITMAP* tmp = ImageManager::GenericLoader(imageFileName, 0);
        glData.imageWidth = FreeImage_GetWidth(tmp);
        glData.imageHeight = FreeImage_GetHeight(tmp);
        glData.imageBPP = FreeImage_GetBPP(tmp);
        glData.imagePitch = FreeImage_GetPitch(tmp);

        //OpenGL Texture
        glEnable(GL_TEXTURE_2D);
        glGenTextures(1, &glData.textureID);
        glBindTexture(GL_TEXTURE_2D, glData.textureID);

        //WARNING: Just some of inner format are supported by CUDA!!!
        glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, glData.imageWidth, glData.imageHeight, 0, GL_RED, GL_UNSIGNED_BYTE, FreeImage_GetBits(tmp));
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

        FreeImage_Unload(tmp);

        glGenBuffers(1, &glData.pboID);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, glData.pboID);														// Make this the current UNPACK buffer (OpenGL is state-based)
        glBufferData(GL_PIXEL_UNPACK_BUFFER, glData.imageWidth * glData.imageHeight * 4, NULL, GL_DYNAMIC_COPY);	// Allocate data for the buffer. 4-channel 8-bit image
    }

    void initCUDAObjects() {
        // Register Image to cuda tex resource
        checkCudaErrors(cudaGraphicsGLRegisterImage(
                &cudaData.texResource,
                glData.textureID,
                GL_TEXTURE_2D,
                cudaGraphicsRegisterFlags::cudaGraphicsRegisterFlagsReadOnly
        ));

        // Map reousrce and retrieve pointer to undelying array data
        checkCudaErrors(cudaGraphicsMapResources(1, &cudaData.texResource, 0)); //OPENGL, pls nepracuj ted s tou texturou
        checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&cudaData.texArrayData, cudaData.texResource, 0, 0));    //z resourcu chci tahat pixelova data

        // Set resource descriptor
        cudaData.resDesc.resType = cudaResourceType::cudaResourceTypeArray;
        cudaData.resDesc.res.array.array = cudaData.texArrayData;

        // Set Texture Descriptor: Tex Units will know how to read the texture
        cudaData.texDesc.readMode = cudaReadModeElementType;
        cudaData.texDesc.normalizedCoords = false;
        cudaData.texDesc.filterMode = cudaFilterModePoint;
        cudaData.texDesc.addressMode[0] = cudaAddressModeClamp;
        cudaData.texDesc.addressMode[1] = cudaAddressModeClamp;

        // Set Channel Descriptor: How to interpret individual bytes
        checkCudaErrors(cudaGetChannelDesc(&cudaData.texChannelDesc, cudaData.texArrayData));

        // Create CUDA Texture Object
        checkCudaErrors(cudaCreateTextureObject(&cudaData.texObj, &cudaData.resDesc, &cudaData.texDesc, nullptr));

        // Unmap resource: Release the resource for OpenGL
        checkCudaErrors(cudaGraphicsUnmapResources(1, &cudaData.texResource, 0));

        // Register PBO
        checkCudaErrors(cudaGraphicsGLRegisterBuffer(
                &cudaData.pboResource,
                glData.pboID,
                cudaGraphicsRegisterFlags::cudaGraphicsRegisterFlagsWriteDiscard
        ));
    }


    void init(const string& path) {
        prepareGlObjects(path.c_str());
        initCUDAObjects();
    }

    ~HeightMap() {
        //checkCudaErrors(cudaGraphicsUnregisterResource(this->cudaData.texResource));
        //checkCudaErrors(cudaGraphicsUnregisterResource(this->cudaData.pboResource));

        if (this->glData.textureID > 0)
            glDeleteTextures(1, &this->glData.textureID);
        if (this->glData.pboID > 0)
            glDeleteBuffers(1, &this->glData.pboID);
    }
};
HeightMap hMap;

struct Particle {
    float x, y;         // POSITION
    float v_x { 0.0f }; // VELOCITY IN DIRECTION X
    float v_y { 0.0f }; // VELOCITY IN DIRECTION Y
};

std::vector<Particle> generateParticles(int n) {
    std::vector<Particle> result;

    std::uniform_real_distribution<float> dis(0.0, 1.0);
    for (int i = 0; i < n; i++) {
        result.push_back(Particle{
            dis(generator) * settings.heightmapGridX,
            dis(generator) * settings.heightmapGridY,
            0.0f, 0.0f
        });
    }

    return result;
}

#pragma region --- OPEN_GL ---

void my_display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnable(GL_TEXTURE_2D);

    glBindTexture(GL_TEXTURE_2D, hMap.glData.textureID);
    glBegin(GL_QUADS);
    glTexCoord2d(0, 0);		glVertex2d(0, 0);
    glTexCoord2d(1, 0);		glVertex2d( hMap.glData.viewportWidth, 0);
    glTexCoord2d(1, 1);		glVertex2d( hMap.glData.viewportWidth,  hMap.glData.viewportHeight);
    glTexCoord2d(0, 1);		glVertex2d(0,  hMap.glData.viewportHeight);
    glEnd();

    glDisable(GL_TEXTURE_2D);

    glFlush();
    glutSwapBuffers();
}

void my_resize(GLsizei w, GLsizei h) {
    hMap.glData.viewportWidth = w;
    hMap.glData.viewportHeight = h;

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glViewport(0, 0, hMap.glData.viewportWidth, hMap.glData.viewportHeight);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, hMap.glData.viewportWidth, 0, hMap.glData.viewportHeight);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glutPostRedisplay();
}

void my_idle() {
    //cudaWorker();
    glutPostRedisplay();
}

void initGL(int argc, char** argv) {
    glutInit(&argc, argv);

    glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize(hMap.glData.viewportWidth, hMap.glData.viewportHeight);
    glutInitWindowPosition(0, 0);
    glutSetOption(GLUT_RENDERING_CONTEXT, false ? GLUT_USE_CURRENT_CONTEXT : GLUT_CREATE_NEW_CONTEXT);
    glutCreateWindow(0);

    char m_windowsTitle[512];
    snprintf(m_windowsTitle, 512, "SimpleView | context %s | renderer %s | vendor %s ",
             (const char*)glGetString(GL_VERSION),
             (const char*)glGetString(GL_RENDERER),
             (const char*)glGetString(GL_VENDOR));
    glutSetWindowTitle(m_windowsTitle);

    glutDisplayFunc(my_display);
    glutReshapeFunc(my_resize);
    glutIdleFunc(my_idle);
    glutSetCursor(GLUT_CURSOR_CROSSHAIR);

    // initialize necessary OpenGL extensions
    glewInit();

    glClearColor(0.0, 0.0, 0.0, 1.0);
    glShadeModel(GL_SMOOTH);
    glViewport(0, 0, hMap.glData.viewportWidth, hMap.glData.viewportHeight);

    glFlush();
}

#pragma endregion

int main(int argc, char* argv[]) {
    Particle* dLeaders;
    Particle* dFollowers;

    #pragma region initialize
    initializeCUDA(deviceProp);
    if (argc < 2) {
        printf("Please specify path to the configuration path");
        return 1;
    }
    Settings settings(argv[1]);
    initGL(1, argv);

    // INITIALIZE HEIHGHT MAP
    hMap.init(settings.heightMap);

    // CREATE LEADERS AND COPY TO DEVICE
    cudaMalloc((void**)&dLeaders, settings.leaders * sizeof(Particle));
    cudaMemcpy(dLeaders, generateParticles(settings.leaders).data(), settings.leaders * sizeof(Particle), cudaMemcpyHostToDevice);

    // CREATE FOLLOWERS AND COPY TO DEVICE
    cudaMalloc((void**)&dLeaders, settings.followers * sizeof(Particle));
    cudaMemcpy(dLeaders, generateParticles(settings.followers).data(), settings.followers * sizeof(Particle), cudaMemcpyHostToDevice);

    #pragma endregion

    glutMainLoop();

    #pragma region clean_up
    if (dLeaders) cudaFree(dLeaders);
    if (dFollowers) cudaFree(dFollowers);
    #pragma endregion

    return 0;
}