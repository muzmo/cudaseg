#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cutil.h>

#define BLOCKDIM_X	 32
#define BLOCKDIM_Y	 4
#define BLOCKDIM_Z	 1

char *volumeFilename, *maskFilename;
int	ITERATIONS, THRESHOLD, EPSILON;
float alpha;

float *phi, *D;
size_t size, pitchbytes;
unsigned char *input,*output;

int NX, NY, NZ, N, pitch, sizeofelements;

float *d_phi, *d_phi1, *d_D;

int its=0;
unsigned int Timer = 0;
unsigned int IterationTimer = 0;

int i,j,k;

__global__ void updatephi( float *d_phi, float *d_phi1, float *d_D, int NX, int NY, int NZ, float alpha, int pitch);

unsigned char* loadRawUchar(char *filename, size_t size){

	FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error opening file '%s'\n", filename);
        exit(EXIT_FAILURE);
    }

	unsigned char *data;
	if((data = (unsigned char *) malloc(size))==NULL)printf("ME\n");
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

	float *data;
	if((data = (float *) malloc(size*sizeof(float)))==NULL)printf("ME\n");
	size_t read = fread(data, 4, size, fp);
	fclose(fp);

    printf("Read '%s', %d elements\n", filename, read);

    return data;
}

void cuda_update(){

	dim3 dimGrid( ((NX-1)/BLOCKDIM_X) + 1, ((NY-1)/BLOCKDIM_Y) +1);
	dim3 dimBlock(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
	
	updatephi<<< dimGrid, dimBlock>>>(d_phi, d_phi1, d_D,  NX, NY, NZ, alpha, pitch);
	d_phi1=d_phi;
	
	CUT_CHECK_ERROR("Kernel execution failed\n");

	CUDA_SAFE_CALL(cudaThreadSynchronize());

}

int main(int argc, char** argv){

	if(argc<9){
		printf("Too few command line arguments specified. Example: Seg -volume=brain_181_217_181.raw -mask=phi.raw -xsize=181 -ysize=217 -zsize=181 -iterations=1000 -threshold=150 -epsilon=50 -alpha=0.03\n");
		exit(0);
	}

	cutGetCmdLineArgumentstr( argc, (const char**) argv, "volume", &volumeFilename);
	cutGetCmdLineArgumentstr( argc, (const char**) argv, "mask", &maskFilename);
	cutGetCmdLineArgumenti( argc, (const char**) argv, "xsize", &NX);
	cutGetCmdLineArgumenti( argc, (const char**) argv, "ysize", &NY);
	cutGetCmdLineArgumenti( argc, (const char**) argv, "zsize", &NZ);
	cutGetCmdLineArgumenti( argc, (const char**) argv, "iterations", &ITERATIONS);
	cutGetCmdLineArgumenti( argc, (const char**) argv, "threshold", &THRESHOLD);
	cutGetCmdLineArgumenti( argc, (const char**) argv, "epsilon", &EPSILON);
	cutGetCmdLineArgumentf( argc, (const char**) argv, "alpha", &alpha);
	//cutGetCmdLineArgumenti( argc, (const char**) argv, "sizeofelements", &sizeofelements);

	N=NX*NY*NZ;
	input = loadRawUchar(volumeFilename, N);
	phi = loadMask(maskFilename, N);

	if((D = (float *)malloc(NX*NY*NZ*sizeof(float)))==NULL)printf("ME_D\n");
	for(i=0;i<N;i++){
		D[i] = EPSILON - abs((unsigned char)input[i] - THRESHOLD);
	}


	// Allocate Memory on Device
	CUDA_SAFE_CALL( cudaMallocPitch((void**)&d_D,			  &pitchbytes, sizeof(float)*NX, NY*NZ));
	CUDA_SAFE_CALL( cudaMallocPitch((void**)&d_phi,           &pitchbytes, sizeof(float)*NX, NY*NZ));
	CUDA_SAFE_CALL( cudaMallocPitch((void**)&d_phi1,          &pitchbytes, sizeof(float)*NX, NY*NZ));

	pitch=pitchbytes/sizeof(float);

	// Copy Host Thresholding Data to Device Memory
	CUDA_SAFE_CALL( cudaMemcpy2D(d_D,    pitchbytes, D,	  sizeof(float)*NX,	sizeof(float)*NX, NY*NZ, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL( cudaMemcpy2D(d_phi1, pitchbytes, phi, sizeof(float)*NX, sizeof(float)*NX, NY*NZ, cudaMemcpyHostToDevice));

	// Set up CUDA Timer
	cutCreateTimer(&Timer);
	cutCreateTimer(&IterationTimer);

	cutStartTimer(Timer);

	for(its=0;its<=ITERATIONS;its++){
		cuda_update();
		if(its%50==0){
			printf("Iteration %3d Total Time: %3.2f ReInit Time: %3.2f\n", its, 0.001*cutGetTimerValue(Timer), 0.001*cutGetTimerValue(IterationTimer));}
	}


	if((output = (unsigned char *) malloc(NX*NY*NZ))==NULL)printf("ME_OUTPUT\n");
	cudaMemcpy2D(phi, sizeof(float)*NX, d_phi1, pitchbytes, sizeof(float)*NX, NY*NZ, cudaMemcpyDeviceToHost);
	for(i=0;i<N;i++){
		if(phi[i]>0){output[i]=0;} else { output[i]=255; }
	}
	char *outputFilename= "output.raw";
	FILE *fp = fopen(outputFilename, "wb");
	size_t write = fwrite(output, 1, N, fp);
	fclose(fp);
    printf("Write '%s', %d bytes\n", outputFilename, write);

	char dummy[100];
	scanf("%c",dummy);

	CUDA_SAFE_CALL( cudaFree(d_phi) );
	CUDA_SAFE_CALL( cudaFree(d_phi1) );
	CUDA_SAFE_CALL( cudaFree(d_D) );
	free(D);
	free(phi);
	free(input);

}

