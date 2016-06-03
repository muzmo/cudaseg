#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cutil.h>
#include <math.h>
#include <GL/glut.h>

#define IMAGE			"liver.bmp"

#define ITERATIONS   5000
#define THRESHOLD	 180
#define EPSILON		 40

#define RITS		 50

#define BLOCKDIM_X	 16
#define BLOCKDIM_Y	 32

float *phi, *D;
uchar4 *h_Src, *h_Mask;
int imageW, imageH, N;

float *d_phi, *d_D;
float *d_phi1;

void LoadBMPFile(uchar4 **dst, int *width, int *height, const char *name);
void sedt2d(int *_d,unsigned char *_bimg,int _h,int _w);


int its=0;
unsigned int Timer = 0;
unsigned int ReInitTimer = 0;

int r;
int c;
int i;

__global__ void updatephi( float *d_phi, float *d_phi1, float *d_D, int imageW, int imageH);

void init_phi(){

	int *init;
	unsigned char *mask;
	const char *mask_path = "mask.bmp";
	if((init=(int *)malloc(imageW*imageH*sizeof(int)))==NULL)printf("ME_INIT\n");
	if((phi=(float *)malloc(imageW*imageH*sizeof(float)))==NULL)printf("ME_PHI\n");
	mask = (unsigned char *)malloc(imageW*imageH*sizeof(unsigned char));

	//printf("Init Mask\n");
	LoadBMPFile(&h_Mask, &imageW, &imageH, mask_path);
	

	for(r=0;r<imageH;r++){
		for(c=0;c<imageW;c++){
			mask[r*imageW+c] = (h_Mask[r*imageW+c].x)/255;
			//printf("%3d ", mask[r*imageW+c]);
		}
		//printf("\n");
	}

	sedt2d(init,mask,imageH,imageW);

	//printf("sdf of init mask\n");
	for(r=0;r<imageH;r++){
		for(c=0;c<imageW;c++){
			phi[r*imageW+c]=(float)init[r*imageW+c];
			if(phi[r*imageW+c]>0){
				phi[r*imageW+c]=0.5*sqrt(abs(phi[r*imageW+c]));
			} else {
				phi[r*imageW+c]=-0.5*sqrt(abs(phi[r*imageW+c]));
			}
			//printf("%6.3f ", phi[r*imageW+c]);
		}
		//printf("\n");
	}

	free(init);
	free(mask);
}

void reinit_phi(){

	int *intphi;
	unsigned char *reinit;
	if((intphi=(int *)malloc(imageW*imageH*sizeof(int)))==NULL)printf("ME_INIT\n");
	reinit=(unsigned char *)malloc(imageW*imageH*sizeof(unsigned char));//TODO check

	for(i=0;i<N;i++){
		if(phi[i]<0){
			phi[i]=1;
		} else {
			phi[i]=0;
		}
		reinit[i]=(int)phi[i];
	}


	sedt2d(intphi,reinit,imageH,imageW);

	/*printf("ReInit @ %4d its\n",its);*/
	for(r=0;r<imageH;r++){
		for(c=0;c<imageW;c++){
			phi[r*imageW+c]=(float)intphi[r*imageW+c];
			if(phi[r*imageW+c]>0){
				phi[r*imageW+c]=0.5*sqrt(abs(phi[r*imageW+c]));
			} else {
				phi[r*imageW+c]=-0.5*sqrt(abs(phi[r*imageW+c]));
			}
			//printf("%6.3f ", phi[r*imageW+c]);
		}
		//printf("\n");
	}

	free(reinit);
	free(intphi);
}

void cuda_update(){


	dim3 dimGrid( ((imageW-1)/BLOCKDIM_X) + 1, ((imageH-1)/BLOCKDIM_Y) +1 );
	dim3 dimBlock(BLOCKDIM_X, BLOCKDIM_Y);

	
	updatephi<<< dimGrid, dimBlock>>>(d_phi, d_phi1, d_D,  imageW, imageH);

	
	d_phi1=d_phi;
	


}

void disp(void){
	
	
	glClear(GL_COLOR_BUFFER_BIT);

	
	cuda_update();
	

	its++;

	if(its<ITERATIONS){
		glutPostRedisplay();
		
		if(its%50==0){
			
			printf("Iteration %3d Total Time: %3.2f ReInit Time: %3.2f\n", its, 0.001*cutGetTimerValue(Timer), 0.001*cutGetTimerValue(ReInitTimer));
			
			cutStartTimer(ReInitTimer); // ReInit Timer Start
			cudaMemcpy(phi, d_phi, sizeof(float)*imageW*imageH, cudaMemcpyDeviceToHost);

			reinit_phi(); // ReInit

			glDrawPixels(imageW, imageH, GL_GREEN, GL_FLOAT, phi);
			glutSwapBuffers();
			cutStopTimer(ReInitTimer); // ReInit Timer Stop
		}

	} else {
		
		printf("Iteration %3d Total Time: %3.2f ReInit Time: %3.2f\n", its, 0.001*cutGetTimerValue(Timer), 0.001*cutGetTimerValue(ReInitTimer));

		cudaMemcpy(phi, d_phi, sizeof(float)*imageW*imageH, cudaMemcpyDeviceToHost);
		glDrawPixels(imageW, imageH, GL_GREEN, GL_FLOAT, phi);
		glutSwapBuffers();



	}
	
}

int main(int argc, char** argv){

	// Load the Input Image using BMPLoader
	const char *image_path = IMAGE;
	LoadBMPFile(&h_Src, &imageW, &imageH, image_path);
	D = (float *)malloc(imageW*imageH*sizeof(float));
	for(r=0;r<imageH;r++){
		for(c=0;c<imageW;c++){
			D[r*imageW+c] = h_Src[r*imageW+c].x;
		}
	}

	N = imageW*imageH;

	// Threshold based on hash defined paramters
	for(i=0;i<N;i++){
		D[i] = EPSILON - abs(D[i] - THRESHOLD);
	}

	// Init phi to SDF
	init_phi();

	// Set up CUDA Timer
	cutCreateTimer(&Timer);
	cutCreateTimer(&ReInitTimer);

	cutStartTimer(Timer);

	// Allocate Memory on Device
	cudaMalloc((void**)&d_D,        sizeof(float)*imageW*imageH);
	cudaMalloc((void**)&d_phi,      sizeof(float)*imageW*imageH);
	cudaMalloc((void**)&d_phi1,         sizeof(float)*imageW*imageH);

	// Copy Host Thresholding Data to Device Memory
	cudaMemcpy(d_D, D,				sizeof(float)*imageW*imageH, cudaMemcpyHostToDevice);
	cudaMemcpy(d_phi1, phi, sizeof(float)*imageW*imageH, cudaMemcpyHostToDevice);

	// Init GL Window
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_ALPHA | GLUT_DOUBLE);
	glutInitWindowSize(imageW,imageH);
	glutInitWindowPosition(100,100);
	glutCreateWindow("GL Level Set Evolution");
	glClearColor(0.0,0.0,0.0,0.0);
	glutDisplayFunc(disp);
	glutMainLoop();

		cudaFree(d_D);
		cudaFree(d_phi1);
		cudaFree(d_phi);
}




//TODO Memory Malloc Free

//TODO Comment Code
