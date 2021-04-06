//#include <glew.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
//#include <freeglut.h>

#include <cudaDefs.h>

#include <cuda_gl_interop.h>
#include <helper_cuda.h>							// normalize method
#include <helper_math.h>							// normalize method

#include <imageManager.h>
#include <benchmark.h>

#define TPB_1D 8									// ThreadsPerBlock in one dimension
#define TPB_2D TPB_1D*TPB_1D						// ThreadsPerBlock = TPB_1D*TPB_1D (2D block)

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

using DT = uchar4;

//OpenGL
struct GLData
{
	unsigned int imageWidth;
	unsigned int imageHeight;
	unsigned int imageBPP;							//Bits Per Pixel = 8, 16, 24, or 32 bit
	unsigned int imagePitch;

	unsigned int pboID;
	unsigned int textureID;
	unsigned int viewportWidth = 1024;
	unsigned int viewportHeight = 1024;
};
GLData gl;

unsigned char someValue = 0;

//CUDA
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

CudaData cd;


#pragma region CUDA Routines

__global__ void applyFilter(const cudaTextureObject_t srcTex, const unsigned char someValue, const unsigned int pboWidth, const unsigned int pboHeight, unsigned char* pbo) {
	// TODO: Make some data processing
	unsigned int tx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int ty = blockIdx.y * blockDim.y + threadIdx.y;

	if (tx >= pboWidth || ty > pboHeight) return;

	unsigned int pboIdx = ((ty * pboWidth) + tx) * 4;
	const uchar4 texel = tex2D<uchar4>(srcTex, tx, ty);

	pbo[pboIdx++] = someValue;
	pbo[pboIdx++] = texel.y;
	pbo[pboIdx++] = texel.z;
	pbo[pboIdx]   = texel.w;
}

void cudaWorker() {
	// Map GL resources
    checkCudaErrors(cudaGraphicsMapResources(1, &cd.texResource, 0));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&cd.texArrayData, cd.texResource, 0, 0));

    checkCudaErrors(cudaGraphicsMapResources(1, &cd.pboResource, 0));
    unsigned char* pboData;
    size_t pboSize;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&pboData, &pboSize, cd.pboResource));

	// TODO: Run kernel
	dim3 block(TPB_1D, TPB_1D, 1);
	dim3 grid((gl.imageWidth + TPB_1D - 1) / TPB_1D, (gl.imageHeight + TPB_1D - 1) / TPB_1D, 1);
	if (someValue > 255) someValue = 0;
	applyFilter<<<grid, block>>>(cd.texObj, someValue, gl.imageWidth, gl.imageHeight, pboData);
	someValue++;

	// Unmap GL Resources
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cd.texResource, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cd.pboResource, 0));

	// This updates GL texture from PBO
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, gl.pboID);
	glBindTexture(GL_TEXTURE_2D, gl.textureID);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, gl.imageWidth, gl.imageHeight, GL_RGBA, GL_UNSIGNED_BYTE, NULL);   //Source parameter is NULL, Data is coming from a PBO, not host memory

	printf(".");
}

void initCUDAObjects() {
	// Register Image to cuda tex resource
	checkCudaErrors(cudaGraphicsGLRegisterImage(
	        &cd.texResource,
	        gl.textureID,
	        GL_TEXTURE_2D,
	        cudaGraphicsRegisterFlags::cudaGraphicsRegisterFlagsReadOnly
	        ));

	// Map reousrce and retrieve pointer to undelying array data
	checkCudaErrors(cudaGraphicsMapResources(1, &cd.texResource, 0)); //OPENGL, pls nepracuj ted s tou texturou
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&cd.texArrayData, cd.texResource, 0, 0));    //z resourcu chci tahat pixelova data

	// Set resource descriptor
	cd.resDesc.resType = cudaResourceType::cudaResourceTypeArray;
	cd.resDesc.res.array.array = cd.texArrayData;

	// Set Texture Descriptor: Tex Units will know how to read the texture
    cd.texDesc.readMode = cudaReadModeElementType;
    cd.texDesc.normalizedCoords = false;
    cd.texDesc.filterMode = cudaFilterModePoint;
    cd.texDesc.addressMode[0] = cudaAddressModeClamp;
    cd.texDesc.addressMode[1] = cudaAddressModeClamp;

	// Set Channel Descriptor: How to interpret individual bytes
    checkCudaErrors(cudaGetChannelDesc(&cd.texChannelDesc, cd.texArrayData));

	// Create CUDA Texture Object
    checkCudaErrors(cudaCreateTextureObject(&cd.texObj, &cd.resDesc, &cd.texDesc, nullptr));

	// Unmap resource: Release the resource for OpenGL
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cd.texResource, 0));

	// Register PBO
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(
	        &cd.pboResource,
	        gl.pboID,
	        cudaGraphicsRegisterFlags::cudaGraphicsRegisterFlagsWriteDiscard
	        ));
}

void releaseCUDA() {
	// Unregister resources
	checkCudaErrors(cudaGraphicsUnregisterResource(cd.texResource));
	checkCudaErrors(cudaGraphicsUnregisterResource(cd.pboResource));
}
#pragma endregion

#pragma region OpenGL Routines
void prepareGlObjects(const char* imageFileName) {  //alokovani zdroju na karte
	FIBITMAP* tmp = ImageManager::GenericLoader(imageFileName, 0);
	gl.imageWidth = FreeImage_GetWidth(tmp);
	gl.imageHeight = FreeImage_GetHeight(tmp);
	gl.imageBPP = FreeImage_GetBPP(tmp);
	gl.imagePitch = FreeImage_GetPitch(tmp);

	//OpenGL Texture
	glEnable(GL_TEXTURE_2D);
	glGenTextures(1, &gl.textureID);
	glBindTexture(GL_TEXTURE_2D, gl.textureID);

	//WARNING: Just some of inner format are supported by CUDA!!!
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, gl.imageWidth, gl.imageHeight, 0, GL_BGRA, GL_UNSIGNED_BYTE, FreeImage_GetBits(tmp));
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

	FreeImage_Unload(tmp);

	glGenBuffers(1, &gl.pboID);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, gl.pboID);														// Make this the current UNPACK buffer (OpenGL is state-based)
	glBufferData(GL_PIXEL_UNPACK_BUFFER, gl.imageWidth * gl.imageHeight * 4, NULL, GL_DYNAMIC_COPY);	// Allocate data for the buffer. 4-channel 8-bit image
}

void my_display()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, gl.textureID);

	glBegin(GL_QUADS);

	glTexCoord2d(0, 0);		glVertex2d(0, 0);
	glTexCoord2d(1, 0);		glVertex2d(gl.viewportWidth, 0);
	glTexCoord2d(1, 1);		glVertex2d(gl.viewportWidth, gl.viewportHeight);
	glTexCoord2d(0, 1);		glVertex2d(0, gl.viewportHeight);

	glEnd();

	glDisable(GL_TEXTURE_2D);

	glFlush();
	glutSwapBuffers();
}

void my_resize(GLsizei w, GLsizei h)
{
	gl.viewportWidth = w;
	gl.viewportHeight = h;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glViewport(0, 0, gl.viewportWidth, gl.viewportHeight);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0, gl.viewportWidth, 0, gl.viewportHeight);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glutPostRedisplay();
}

void my_idle()
{
	cudaWorker();
	glutPostRedisplay();
}

void initGL(int argc, char** argv)
{
	glutInit(&argc, argv);

	glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
	glutInitWindowSize(gl.viewportWidth, gl.viewportHeight);
	glutInitWindowPosition(0, 0);
	glutSetOption(GLUT_RENDERING_CONTEXT, false ? GLUT_USE_CURRENT_CONTEXT : GLUT_CREATE_NEW_CONTEXT);
	glutCreateWindow(0);

	char m_windowsTitle[512];
	//sprintf_s
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
	glViewport(0, 0, gl.viewportWidth, gl.viewportHeight);

	glFlush();
}

void releaseOpenGL()
{
	if (gl.textureID > 0)
		glDeleteTextures(1, &gl.textureID);
	if (gl.pboID > 0)
		glDeleteBuffers(1, &gl.pboID);
}
#pragma endregion OpenGL Routines

void releaseResources() {
	releaseCUDA();
	releaseOpenGL();
}

int main(int argc, char* argv[]) {
	initializeCUDA(deviceProp);
	FreeImage_Initialise();

	initGL(argc, argv);
	prepareGlObjects("/home/pavel/prg/cpp/ParallelApplications2/images/lena.png");

	initCUDAObjects();

	//start rendering mainloop
	glutMainLoop();
	FreeImage_DeInitialise();
	atexit(releaseResources);
}