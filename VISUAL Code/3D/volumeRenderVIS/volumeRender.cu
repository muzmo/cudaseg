/*
* Copyright 1993-2008 NVIDIA Corporation.  All rights reserved.
*
* NOTICE TO USER:
*
* This source code is subject to NVIDIA ownership rights under U.S. and
* international Copyright laws.
*
* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
* CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
* REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
* MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
* OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
* OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
* OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
* OR PERFORMANCE OF THIS SOURCE CODE.
*
* U.S. Government End Users.  This source code is a "commercial item" as
* that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
* "commercial computer software" and "commercial computer software
* documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
* and is provided to the U.S. Government only as a commercial end item.
* Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
* 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
* source code with only those rights set forth herein.
*/

/*
    Volume rendering sample

    This sample loads a 3D volume from disk and displays it using
    ray marching and 3D textures.
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <GL/glew.h>

#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <cuda_gl_interop.h>
#include <cutil_inline.h>
#include <cuda_runtime.h>
#include <cutil.h>

typedef unsigned int uint;
typedef unsigned char uchar;

#include <volumeRender_kernel.cu>
#include <kernel.cu>

#define ITERATIONS   500
#define THRESHOLD	 150
#define EPSILON		 50

#define BLOCKDIM_X	 32
#define BLOCKDIM_Y	 4
#define BLOCKDIM_Z	 1

#define RITS		 5

char *volumeFilename = "brain.raw";
char *maskFilename = "phi.raw";
char *outputFilename= "output.raw";
cudaExtent volumeSize = make_cudaExtent(181,217,181);

float *phi, *D, *contour;
char *path;
size_t size, pitchbytes;
unsigned char *input,*output;
int imageW, imageH, imageD, N, pitch;

float *d_phi, *d_phi1, *d_D;

int its=0;
unsigned int Timer = 0;
unsigned int IterationTimer = 0;

int i,j,k;



uint width = 600, height = 600;
dim3 blockSize(16, 16);
dim3 gridSize(width / blockSize.x, height / blockSize.y);

float3 viewRotation;
float3 viewTranslation = make_float3(0.0, 0.0, -4.0f);
float invViewMatrix[12];

float density = 0.1f;
float brightness = 0.9f;
float transferOffset = -0.5f;
float transferScale = 0.02f;
bool linearFiltering = true;

cudaArray *d_volumeArray = 0;
cudaArray *d_transferFuncArray;
GLuint pbo = 0;     // OpenGL pixel buffer object

void initPixelBuffer();
__global__ void updatephi( float *d_phi, float *d_phi1, float *d_D, int imageW, int imageH, int imageD, int pitch);
// render image using CUDA
void render()
{
    cutilSafeCall( cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix, sizeof(float4)*3) );

    // map PBO to get CUDA device pointer
    uint *d_output;
    cutilSafeCall(cudaGLMapBufferObject((void**)&d_output, pbo));

    cutilSafeCall(cudaMemset(d_output, 0, width*height*4));

    // call CUDA kernel, writing results to PBO
    d_render<<<gridSize, blockSize>>>(d_output, width, height, density, brightness, transferOffset, transferScale);
    cutilCheckMsg("kernel failed");

    cutilSafeCall(cudaGLUnmapBufferObject(pbo));
}
void initCuda(float *h_volume, cudaExtent volumeSize)
{
    // create 3D array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cutilSafeCall( cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize) );

    // copy data to 3D array
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr((void*)h_volume, volumeSize.width*sizeof(float), volumeSize.width, volumeSize.height);
    copyParams.dstArray = d_volumeArray;
    copyParams.extent   = volumeSize;
    copyParams.kind     = cudaMemcpyHostToDevice;
    cutilSafeCall( cudaMemcpy3D(&copyParams) );  

    // set texture parameters
    tex.normalized = true;                      // access with normalized texture coordinates
    tex.filterMode = cudaFilterModeLinear;      // linear interpolation
    tex.addressMode[0] = cudaAddressModeClamp;  // wrap texture coordinates
    tex.addressMode[1] = cudaAddressModeClamp;

    // bind array to 3D texture
    cutilSafeCall(cudaBindTextureToArray(tex, d_volumeArray, channelDesc));

    // create transfer function texture
    float4 transferFunc[] = {
        {  0.0, 0.0, 0.0, 0.0, },
        {  1.0, 0.0, 0.0, 1.0, },
        {  1.0, 0.5, 0.0, 1.0, },
        {  1.0, 1.0, 0.0, 1.0, },
        {  0.0, 1.0, 0.0, 1.0, },
        {  0.0, 1.0, 1.0, 1.0, },
        {  0.0, 0.0, 1.0, 1.0, },
        {  1.0, 0.0, 1.0, 1.0, },
        {  0.0, 0.0, 0.0, 0.0, },
    };

    cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc<float4>();
    cudaArray* d_transferFuncArray;
    cutilSafeCall(cudaMallocArray( &d_transferFuncArray, &channelDesc2, sizeof(transferFunc)/sizeof(float4), 1)); 
    cutilSafeCall(cudaMemcpyToArray( d_transferFuncArray, 0, 0, transferFunc, sizeof(transferFunc), cudaMemcpyHostToDevice));

    transferTex.filterMode = cudaFilterModeLinear;
    transferTex.normalized = true;    // access with normalized texture coordinates
    transferTex.addressMode[0] = cudaAddressModeClamp;   // wrap texture coordinates

    // Bind the array to the texture
    cutilSafeCall( cudaBindTextureToArray( transferTex, d_transferFuncArray, channelDesc2));
}
// display results using OpenGL (called by GLUT)
void cuda_update(){

	dim3 dimGrid( ((imageW-1)/BLOCKDIM_X) + 1, ((imageH-1)/BLOCKDIM_Y) +1);
	dim3 dimBlock(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
	
	updatephi<<< dimGrid, dimBlock>>>(d_phi, d_phi1, d_D,  imageW, imageH, imageD, pitch);
	d_phi1=d_phi;
	
	CUT_CHECK_ERROR("Kernel execution failed\n");

	cudaThreadSynchronize();

}
void display()
{


	if(its<ITERATIONS){
		
	cuda_update();

	
		if(its%RITS==0){

			cudaMemcpy2D(phi, sizeof(float)*imageW, d_phi1, pitchbytes, sizeof(float)*imageW, imageH*imageD, cudaMemcpyDeviceToHost);

			for(i=0;i<N;i++){
				if(phi[i]<0){contour[i] = 25;} else {contour[i]=0;}
			}
			cutilSafeCall(cudaFreeArray(d_volumeArray));
			cutilSafeCall(cudaFreeArray(d_transferFuncArray));
			initCuda(contour, volumeSize);
		}

		
	
	}
				// use OpenGL to build view matrix
			GLfloat modelView[16];
			glMatrixMode(GL_MODELVIEW);
			glPushMatrix();
			glLoadIdentity();
			glRotatef(-viewRotation.x, 1.0, 0.0, 0.0);
			glRotatef(-viewRotation.y, 0.0, 1.0, 0.0);
			glTranslatef(-viewTranslation.x, -viewTranslation.y, -viewTranslation.z);
			glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
			glPopMatrix();

			invViewMatrix[0] = modelView[0]; invViewMatrix[1] = modelView[4]; invViewMatrix[2] = modelView[8]; invViewMatrix[3] = modelView[12];
			invViewMatrix[4] = modelView[1]; invViewMatrix[5] = modelView[5]; invViewMatrix[6] = modelView[9]; invViewMatrix[7] = modelView[13];
			invViewMatrix[8] = modelView[2]; invViewMatrix[9] = modelView[6]; invViewMatrix[10] = modelView[10]; invViewMatrix[11] = modelView[14];

			render();

			// display results
			glClear(GL_COLOR_BUFFER_BIT);

			// draw image from PBO
			glDisable(GL_DEPTH_TEST);
			glRasterPos2i(0, 0);
			glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
			glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
			glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

			glutSwapBuffers();
			glutReportErrors();
			glutPostRedisplay();

			its++;

}

void idle()
{
}

void keyboard(unsigned char key, int x, int y)
{
    switch(key) {
        case 27:
            exit(0);
            break;
        case 'f':
            linearFiltering = !linearFiltering;
            tex.filterMode = linearFiltering ? cudaFilterModeLinear : cudaFilterModePoint;
            break;
        case '=':
            density += 0.01;
            break;
        case '-':
            density -= 0.01;
            break;
        case '+':
            density += 0.1;
            break;
        case '_':
            density -= 0.1;
            break;

        case ']':
            brightness += 0.1;
            break;
        case '[':
            brightness -= 0.1;
            break;

        case ';':
            transferOffset += 0.01;
            break;
        case '\'':
            transferOffset -= 0.01;
            break;

        case '.':
            transferScale += 0.01;
            break;
        case ',':
            transferScale -= 0.01;
            break;

        default:
            break;
    }
    printf("density = %.2f, brightness = %.2f, transferOffset = %.2f, transferScale = %.2f\n", density, brightness, transferOffset, transferScale);
    glutPostRedisplay();
}

int ox, oy;
int buttonState = 0;

void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
        buttonState |= 1<<button;
    else if (state == GLUT_UP)
        buttonState = 0;

    ox = x; oy = y;
    glutPostRedisplay();
}

void motion(int x, int y)
{
    float dx, dy;
    dx = x - ox;
    dy = y - oy;

    if (buttonState == 3) {
        // left+middle = zoom
        viewTranslation.z += dy / 100.0;
    } 
    else if (buttonState & 2) {
        // middle = translate
        viewTranslation.x += dx / 100.0;
        viewTranslation.y -= dy / 100.0;
    }
    else if (buttonState & 1) {
        // left = rotate
        viewRotation.x += dy / 5.0;
        viewRotation.y += dx / 5.0;
    }

    ox = x; oy = y;
    glutPostRedisplay();
}

void reshape(int x, int y)
{
    width = x; height = y;
    initPixelBuffer();

    glViewport(0, 0, x, y);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0); 
}



void cleanup()
{
    cutilSafeCall(cudaFreeArray(d_volumeArray));
    cutilSafeCall(cudaFreeArray(d_transferFuncArray));
	cutilSafeCall(cudaGLUnregisterBufferObject(pbo));    
	glDeleteBuffersARB(1, &pbo);
}

int iDivUp(int a, int b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

void initPixelBuffer()
{
    if (pbo) {
        // delete old buffer
        cutilSafeCall(cudaGLUnregisterBufferObject(pbo));
        glDeleteBuffersARB(1, &pbo);
    }

    // create pixel buffer object for display
    glGenBuffersARB(1, &pbo);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*sizeof(GLubyte)*4, 0, GL_STREAM_DRAW_ARB);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	cutilSafeCall(cudaGLRegisterBufferObject(pbo));

    // calculate new grid size
    gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));
}

// Load raw data from disk
uchar *loadRawFile(char *filename, size_t size)
{
	FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error opening file '%s'\n", filename);
        return 0;
    }

	uchar *data = (uchar *) malloc(size);
	size_t read = fread(data, 1, size, fp);
	fclose(fp);

    printf("Read '%s', %d bytes\n", filename, read);

    return data;
}

float *loadMask(char *filename, size_t size){

	FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error opening file '%s'\n", filename);
        exit(EXIT_FAILURE);
    }

	float *data = (float *) malloc(size*sizeof(float));
	size_t read = fread(data, 4, size, fp);
	fclose(fp);

    printf("Read '%s', %d elements\n", filename, read);

    return data;
}

void initseg(){

	size = volumeSize.width*volumeSize.height*volumeSize.depth;
	input = loadRawFile(volumeFilename, size);
	phi = loadMask(maskFilename, size);

	imageW=volumeSize.width;
	imageH=volumeSize.height;
	imageD=volumeSize.depth;
	N=imageW*imageH*imageD;

	if((D = (float *)malloc(imageW*imageH*imageD*sizeof(float)))==NULL)printf("ME_D\n");
	for(i=0;i<N;i++){
		D[i] = EPSILON - abs(input[i] - THRESHOLD);
	}

	// Set up CUDA Timer
	cutCreateTimer(&Timer);
	cutCreateTimer(&IterationTimer);

	cutStartTimer(Timer);

	if((contour= (float *)malloc(imageW*imageH*imageD*sizeof(float)))==NULL)printf("ME_D\n");

	// Allocate Memory on Device
	cudaMallocPitch((void**)&d_D,			  &pitchbytes, sizeof(float)*imageW, imageH*imageD);
	cudaMallocPitch((void**)&d_phi,           &pitchbytes, sizeof(float)*imageW, imageH*imageD);
	cudaMallocPitch((void**)&d_phi1,          &pitchbytes, sizeof(float)*imageW, imageH*imageD);

	pitch=pitchbytes/sizeof(float);

	// Copy Host Thresholding Data to Device Memory
	cudaMemcpy2D(d_D,    pitchbytes, D,	  sizeof(float)*imageW,	sizeof(float)*imageW, imageH*imageD, cudaMemcpyHostToDevice);
	cudaMemcpy2D(d_phi1, pitchbytes, phi, sizeof(float)*imageW, sizeof(float)*imageW, imageH*imageD, cudaMemcpyHostToDevice);


	}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
	initseg();



    
    

    printf("Press '=' and '-' to change density\n"
           "      ']' and '[' to change brightness\n"
           "      ';' and ''' to modify transfer function offset\n"
           "      '.' and ',' to modify transfer function scale\n");

    // initialize GLUT callback functions
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutCreateWindow("CUDA volume rendering");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutReshapeFunc(reshape);
    glutIdleFunc(idle);

    glewInit();
    if (!glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object")) {
        fprintf(stderr, "Required OpenGL extensions missing.");
        exit(-1);
    }
    initPixelBuffer();

    atexit(cleanup);

    glutMainLoop();

    cudaThreadExit();
    return 0;
}
