#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cutil.h>
#include <math.h>
#include <GL/glut.h>

#define IMAGE			"bigbrain.bmp"
#define MASK			"bigmask.bmp"


#define ITERATIONS   5000
#define THRESHOLD	 170
#define EPSILON		 35

#define ALPHA		 0.009
#define DT			 0.25

#define RITS		 50

float *phi, *phi1, *D, *contour;
uchar4 *h_Src, *h_Mask;
int imageW, imageH, N;



void LoadBMPFile(uchar4 **dst, int *width, int *height, const char *name);
void sedt2d(int *_d,unsigned char *_bimg,int _h,int _w);


int its=0;
unsigned int Timer = 0;
unsigned int ReInitTimer = 0;

int r;
int c;
int i;

void init_phi(){

	int *init;
	unsigned char *mask;
	const char *mask_path = MASK;
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


void update_phi(){

float dx;
float dxplus;
float dxminus;
float dxplusy;
float dxminusy;
float maxdxplus;
float maxminusdxminus;
float mindxplus;
float minminusdxminus;

float dy;
float dyplus;
float dyminus;
float dyplusx;
float dyminusx;
float maxdyplus;
float maxminusdyminus;
float mindyplus;
float minminusdyminus;

float gradphimax;
float gradphimin;

float nplusx;
float nplusy;
float nminusx;
float nminusy;
float curvature;

float F;
float gradphi;


int ind;



for(i=0;i<N;i++){
	phi1[i]=phi[i];
}

for(r=0;r<imageH;r++){
	for(c=0;c<imageW;c++){

		ind=r*imageW+c;

		if(c==0||c==imageW-1){dx=0;} else {dx=(phi1[ind+1]-phi1[ind-1])/2;}
		if(c==imageW-1){dxplus=0;} else {dxplus=(phi1[ind+1]-phi1[ind]);}
		if(c==0){dxminus=0;} else {dxminus=(phi1[ind]-phi1[ind-1]);}
		if(r==0||c==0||c==imageW-1){dxplusy=0;} else {dxplusy=(phi1[ind-imageW+1]-phi1[ind-imageW-1])/2;}
		if(r==imageH-1||c==0||c==imageW-1){dxminusy=0;} else {dxminusy=(phi1[ind+imageW+1]-phi1[ind+imageW-1])/2;}
		if(dxplus<0){maxdxplus=0;} else { maxdxplus= dxplus*dxplus; }
		if(-dxminus<0){maxminusdxminus=0;} else { maxminusdxminus= dxminus*dxminus; }
		if(dxplus>0){mindxplus=0;} else { mindxplus= dxplus*dxplus; }
		if(-dxminus>0){minminusdxminus=0;} else { minminusdxminus= dxminus*dxminus; }

		if(r==0||r==imageH-1){dy=0;} else {dy=(phi1[ind-imageW]-phi1[ind+imageW])/2;}
		if(r==0){dyplus=0;} else {dyplus=(phi1[ind-imageW]-phi1[ind]);}
		if(r==imageH-1){dyminus=0;} else {dyminus=(phi1[ind]-phi1[ind+imageW]);}
		if(r==0||c==imageW-1||r==imageH-1){dyplusx=0;} else {dyplusx=(phi1[ind-imageW+1]-phi1[ind+imageW+1])/2;}
		if(r==0||c==0||r==imageH-1){dyminusx=0;} else {dyminusx=(phi1[ind-imageW-1]-phi1[ind+imageW-1])/2;}
		if(dyplus<0){maxdyplus=0;} else { maxdyplus= dyplus*dyplus; }
		if(-dyminus<0){maxminusdyminus=0;} else { maxminusdyminus= dyminus*dyminus; }
		if(dyplus>0){mindyplus=0;} else { mindyplus= dyplus*dyplus; }
		if(-dyminus>0){minminusdyminus=0;} else { minminusdyminus= dyminus*dyminus; }

		gradphimax=sqrt((sqrt(maxdxplus+maxminusdxminus))*(sqrt(maxdxplus+maxminusdxminus))+(sqrt(maxdyplus+maxminusdyminus))*(sqrt(maxdyplus+maxminusdyminus)));
		gradphimin=sqrt((sqrt(mindxplus+minminusdxminus))*(sqrt(mindxplus+minminusdxminus))+(sqrt(mindyplus+minminusdyminus))*(sqrt(mindyplus+minminusdyminus)));
		nplusx= dxplus / sqrt(1.192092896e-07F + (dxplus*dxplus) + ((dyplusx + dy)*(dyplusx + dy)*0.25) );
		nplusy= dyplus / sqrt(1.192092896e-07F + (dyplus*dyplus) + ((dxplusy + dx)*(dxplusy + dx)*0.25) );
		nminusx= dxminus / sqrt(1.192092896e-07F + (dxminus*dxminus) + ((dyminusx + dy)*(dyminusx + dy)*0.25) );
		nminusy= dyminus / sqrt(1.192092896e-07F + (dyminus*dyminus) + ((dxminusy + dx)*(dxminusy + dx)*0.25) );
		curvature= ((nplusx-nminusx)+(nplusy-nminusy))/2;
		
		F = (-ALPHA * D[ind]) + ((1-ALPHA) * curvature);
		if(F>0) {gradphi=gradphimax;} else {gradphi=gradphimin;}
		phi[ind]=phi1[ind] + (DT * F * gradphi);
	}
}



}



void disp(void){
	
	
	glClear(GL_COLOR_BUFFER_BIT);

	
	update_phi();
	

	its++;

	if(its<ITERATIONS){
		glutPostRedisplay();
		
		if(its%50==0){
			
			printf("Iteration %3d Total Time: %3.2f ReInit Time: %3.2f\n", its, 0.001*cutGetTimerValue(Timer), 0.001*cutGetTimerValue(ReInitTimer));
			
			cutStartTimer(ReInitTimer); // ReInit Timer Start
			
			reinit_phi(); // ReInit

			glDrawPixels(imageW, imageH, GL_GREEN, GL_FLOAT, phi);
			glutSwapBuffers();
			cutStopTimer(ReInitTimer); // ReInit Timer Stop
		}

	} else {
		
		printf("Iteration %3d Total Time: %3.2f ReInit Time: %3.2f\n", its, 0.001*cutGetTimerValue(Timer), 0.001*cutGetTimerValue(ReInitTimer));

		glDrawPixels(imageW, imageH, GL_GREEN, GL_FLOAT, phi);
		glutSwapBuffers();



	}
	
}

int main(int argc, char** argv){

	const char *image_path = IMAGE;
	
	//TODO : declare ALL variables here

	LoadBMPFile(&h_Src, &imageW, &imageH, image_path);
	D = (float *)malloc(imageW*imageH*sizeof(float));

	//printf("Input Image\n");
	for(r=0;r<imageH;r++){
		for(c=0;c<imageW;c++){
			D[r*imageW+c] = h_Src[r*imageW+c].x;
			/*printf("%3.0f ", D[r*imageW+c]);*/
		}
		//printf("\n");
	}

	N = imageW*imageH;

	for(i=0;i<N;i++){
		D[i] = EPSILON - abs(D[i] - THRESHOLD);
	}

	//printf("Speed Function\n");	
	//for(int r=0;r<imageH;r++){
	//	for(int c=0;c<imageW;c++){
	//		printf("%3.0f ", D[r*imageW+c]);
	//	}
	//	printf("\n");
	//}

	// Set up CUDA Timer
	cutCreateTimer(&Timer);
	cutCreateTimer(&ReInitTimer);

	cutStartTimer(Timer);

	init_phi();
	if((contour=(float *)malloc(imageW*imageH*sizeof(float)))==NULL)printf("Contour\n");
	if((phi1=(float *)malloc(imageW*imageH*sizeof(float)))==NULL)printf("GRADPHI\n");
//update_phi();

		  // GL initialisation
		  glutInit(&argc, argv);
		  glutInitDisplayMode(GLUT_ALPHA | GLUT_DOUBLE);
		  glutInitWindowSize(imageW,imageH);
		  glutInitWindowPosition(100,100);
		  glutCreateWindow("GL Level Set Evolution");
		  glClearColor(0.0,0.0,0.0,0.0);


		  glutDisplayFunc(disp);
		  glutMainLoop();

	//printf("phi+1\n");
	//for(int r=0;r<imageH;r++){
	//	for(int c=0;c<imageW;c++){
	//		printf("%6.3f ", phi[r*imageW+c]);
	//	}
	//	printf("\n");
	//}
}


//TODO Memory Malloc Free

//TODO Timer

//TODO Comment Code
