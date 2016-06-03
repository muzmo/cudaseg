#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cutil.h>
#include <math.h>
#include <GL/glut.h>

#define ITERATIONS   1000
#define THRESHOLD	 150
#define EPSILON		 50

#define BLOCKDIM_X	 32
#define BLOCKDIM_Y	 4
#define BLOCKDIM_Z	 1

char *volumeFilename = "brain_181_217_181.raw";
char *maskFilename = "phi.raw";
char *outputFilename= "output.raw";

//cudaExtent volumeSize = make_cudaExtent(87, 87, 111);
cudaExtent volumeSize = make_cudaExtent(181,217,181);
//cudaExtent volumeSize = make_cudaExtent(10,10,10);

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

__global__ void updatephi( float *d_phi, float *d_phi1, float *d_D, int imageW, int imageH, int imageD, int pitch);

unsigned char* loadRawUchar(char *filename, size_t size){

	FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error opening file '%s'\n", filename);
        exit(EXIT_FAILURE);
    }

	unsigned char *data = (unsigned char *) malloc(size);
	size_t read = fread(data, 1, size, fp);
	fclose(fp);

    printf("Read '%s', %d bytes\n", filename, read);

    return data;
}

short* loadRawShort(char *filename, size_t size){

	FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error opening file '%s'\n", filename);
        exit(EXIT_FAILURE);
    }

	short *data = (short *) malloc(size*sizeof(short));
	size_t read = fread(data, 2, size, fp);
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


void cuda_update(){

	dim3 dimGrid( ((imageW-1)/BLOCKDIM_X) + 1, ((imageH-1)/BLOCKDIM_Y) +1);
	dim3 dimBlock(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
	
	updatephi<<< dimGrid, dimBlock>>>(d_phi, d_phi1, d_D,  imageW, imageH, imageD, pitch);
	d_phi1=d_phi;
	
	CUT_CHECK_ERROR("Kernel execution failed\n");

	cudaThreadSynchronize();

}

int main(int argc, char** argv){

	size = volumeSize.width*volumeSize.height*volumeSize.depth;
	input = loadRawUchar(volumeFilename, size);
	phi = loadMask(maskFilename, size);

	imageW=volumeSize.width;
	imageH=volumeSize.height;
	imageD=volumeSize.depth;
	N=imageW*imageH*imageD;

	if((D = (float *)malloc(imageW*imageH*imageD*sizeof(float)))==NULL)printf("ME_D\n");
	for(i=0;i<N;i++){
		D[i] = EPSILON - abs((unsigned char)input[i] - THRESHOLD);
	}

	// Set up CUDA Timer
	cutCreateTimer(&Timer);
	cutCreateTimer(&IterationTimer);

	cutStartTimer(Timer);

	// Allocate Memory on Device
	cudaMallocPitch((void**)&d_D,			  &pitchbytes, sizeof(float)*imageW, imageH*imageD);
	cudaMallocPitch((void**)&d_phi,           &pitchbytes, sizeof(float)*imageW, imageH*imageD);
	cudaMallocPitch((void**)&d_phi1,          &pitchbytes, sizeof(float)*imageW, imageH*imageD);

	pitch=pitchbytes/sizeof(float);

	// Copy Host Thresholding Data to Device Memory
	cudaMemcpy2D(d_D,    pitchbytes, D,	  sizeof(float)*imageW,	sizeof(float)*imageW, imageH*imageD, cudaMemcpyHostToDevice);
	cudaMemcpy2D(d_phi1, pitchbytes, phi, sizeof(float)*imageW, sizeof(float)*imageW, imageH*imageD, cudaMemcpyHostToDevice);

	for(its=0;its<ITERATIONS;its++){
		cuda_update();
		if(its%50==0){
			printf("Iteration %3d Total Time: %3.2f ReInit Time: %3.2f\n", its, 0.001*cutGetTimerValue(Timer), 0.001*cutGetTimerValue(IterationTimer));}
	}


	if((output = (unsigned char *) malloc(imageW*imageH*imageD))==NULL)printf("ME_OUTPUT\n");
	cudaMemcpy2D(phi, sizeof(float)*imageW, d_phi1, pitchbytes, sizeof(float)*imageW, imageH*imageD, cudaMemcpyDeviceToHost);
	for(i=0;i<N;i++){
		if(phi[i]>0){output[i]=0;} else { output[i]=255; }
	}
	FILE *fp = fopen(outputFilename, "wb");
	size_t write = fwrite(output, 1, size, fp);
	fclose(fp);
    printf("Write '%s', %d bytes\n", outputFilename, write);

}

