#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <cuda_runtime.h>
#include <cutil.h>
#include <math.h>
#include <GL/glut.h>

#define IMAGE			"liver.bmp"

#define ITERATIONS   5000
#define THRESHOLD	 170
#define EPSILON		 35

#define ALPHA		 0.009
#define DT			 0.25

#define RITS		 50

float *phi, *D, *contour;
uchar4 *h_Src, *h_Mask;
int imageW, imageH, N;
unsigned char *mask;


void LoadBMPFile(uchar4 **dst, int *width, int *height, const char *name);
void sedt2d(int *_d,unsigned char *_bimg,int _h,int _w);


int its=0;
unsigned int Timer = 0;


void init_phi(){

	const char *mask_path = "mask.bmp";

	//printf("Init Mask\n");
	LoadBMPFile(&h_Mask, &imageW, &imageH, mask_path);
	mask = (unsigned char *)malloc(imageW*imageH*sizeof(unsigned char));

	for(int r=0;r<imageH;r++){
		for(int c=0;c<imageW;c++){
			mask[r*imageW+c] = (h_Mask[r*imageW+c].x)/255;
			//printf("%3d ", mask[r*imageW+c]);
		}
		//printf("\n");
	}

	int *init;
	if((init=(int *)malloc(imageW*imageH*sizeof(int)))==NULL)printf("ME_INIT\n");
	if((phi=(float *)malloc(imageW*imageH*sizeof(float)))==NULL)printf("ME_PHI\n");
	sedt2d(init,mask,imageH,imageW);



	//printf("sdf of init mask\n");
	for(int r=0;r<imageH;r++){
		for(int c=0;c<imageW;c++){
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

	unsigned char *reinit;
	reinit=(unsigned char *)malloc(imageW*imageH*sizeof(unsigned char));
	int *intphi;
	if((intphi=(int *)malloc(imageW*imageH*sizeof(int)))==NULL)printf("ME_INIT\n");
	
	for(int i=0;i<N;i++){
		if(phi[i]<0){
			phi[i]=1;
		} else {
			phi[i]=0;
		}
		reinit[i]=(int)phi[i];
	}


	sedt2d(intphi,reinit,imageH,imageW);

	printf("reinit\n");
	for(int r=0;r<imageH;r++){
		for(int c=0;c<imageW;c++){
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

	float *ptr2phi;

	float *dx, *ptr2dx;
	if((dx=(float *)malloc(imageW*imageH*sizeof(float)))==NULL)printf("ME_DX\n");
	ptr2dx=dx;
	ptr2phi=phi;
	for(int r=0;r<imageH;r++){
		*ptr2dx=0;
		for(int c=0;c<imageW-2;c++){
			ptr2phi++;
			ptr2dx++;
			*ptr2dx = (*(ptr2phi+1) - *(ptr2phi-1)) /2 ;
		}
		ptr2dx++;
		*ptr2dx=0;
		ptr2dx++;
		ptr2phi++;
		if(r!=(imageH-1))ptr2phi++;
	}

	float *dxplus, *ptr2dxplus;
	if((dxplus=(float *)malloc(imageW*imageH*sizeof(float)))==NULL)printf("ME_DX+\n");
	for(int i=0;i<N;i++)dxplus[i]=phi[i];
	ptr2dxplus=dxplus;
	for(int r=0;r<imageH;r++){
		for(int c=0;c<imageW-1;c++){
			*ptr2dxplus = *(ptr2dxplus+1) - *ptr2dxplus;
			ptr2dxplus++;
		}
		*ptr2dxplus=0;
		ptr2dxplus++;
	}

	float *dxminus, *ptr2dxminus;
	if((dxminus=(float *)malloc(imageW*imageH*sizeof(float)))==NULL)printf("ME_DX-\n");
	for(int i=0;i<N;i++)dxminus[i]=phi[i];
	ptr2dxminus=dxminus+N-1;
	for(int r=0;r<imageH;r++){
		for(int c=0;c<imageW-1;c++){
			*ptr2dxminus = *ptr2dxminus - *(ptr2dxminus-1);
			ptr2dxminus--;
		}
		*ptr2dxminus=0;
		ptr2dxminus--;
	
	}

	float *dxplusy, *ptr2dxplusy;
	if((dxplusy=(float *)malloc(imageW*imageH*sizeof(float)))==NULL)printf("ME_dxplusy\n");
	ptr2dxplusy=dxplusy;
	ptr2phi=phi;
	for(int c=0;c<imageW;c++){
		*ptr2dxplusy = 0;
		ptr2dxplusy++;
	}
	for(int r=0;r<imageH-1;r++){
		*ptr2dxplusy=0;
		for(int c=0;c<imageW-2;c++){
			ptr2phi++;
			ptr2dxplusy++;
			*ptr2dxplusy = (*(ptr2phi+1) - *(ptr2phi-1)) /2 ;
		}
		ptr2dxplusy++;
		*ptr2dxplusy=0;
		ptr2dxplusy++;
		ptr2phi++;
		if(r!=(imageH-2))ptr2phi++;
	}

	float *dxminusy, *ptr2dxminusy;
	if((dxminusy=(float *)malloc(imageW*imageH*sizeof(float)))==NULL)printf("ME_dxminusy\n");
	ptr2dxminusy=dxminusy;
	ptr2phi=phi+imageW;
	for(int r=0;r<imageH-1;r++){
		*ptr2dxminusy=0;
		for(int c=0;c<imageW-2;c++){
			ptr2phi++;
			ptr2dxminusy++;
			*ptr2dxminusy = (*(ptr2phi+1) - *(ptr2phi-1) );
		}
		ptr2dxminusy++;
		*ptr2dxminusy=0;
		ptr2dxminusy++;
		ptr2phi++;
		if(r!=(imageH-2))ptr2phi++;
	}
	for(int c=0;c<imageW;c++){
		*ptr2dxminusy = 0;
		ptr2dxminusy++;
	}

	float *maxdxplus, *ptr2maxdxplus;
	if((maxdxplus=(float *)malloc(imageW*imageH*sizeof(float)))==NULL)printf("ME_MDX+\n");
	for(int i=0;i<N;i++)maxdxplus[i]=dxplus[i];
	ptr2maxdxplus = maxdxplus;
	for(int i=0;i<N;i++){
		if (*ptr2maxdxplus < 0) {
			*ptr2maxdxplus = 0;
		} else {
			*ptr2maxdxplus *= *ptr2maxdxplus;
		}
		ptr2maxdxplus++;
	}

	float *maxminusdxminus, *ptr2maxminusdxminus;
	if((maxminusdxminus=(float *)malloc(imageW*imageH*sizeof(float)))==NULL)printf("ME_-MDX-\n");
	for(int i=0;i<N;i++)maxminusdxminus[i]=-dxminus[i];
	ptr2maxminusdxminus = maxminusdxminus;
	for(int i=0;i<N;i++){
		if (*ptr2maxminusdxminus < 0) {
			*ptr2maxminusdxminus = 0;
		} else {
			*ptr2maxminusdxminus *= *ptr2maxminusdxminus;
		}
		ptr2maxminusdxminus++;
	}

	float *mindxplus, *ptr2mindxplus;
	if((mindxplus=(float *)malloc(imageW*imageH*sizeof(float)))==NULL)printf("ME_MDX+\n");
	for(int i=0;i<N;i++)mindxplus[i]=dxplus[i];
	ptr2mindxplus = mindxplus;
	for(int i=0;i<N;i++){
		if (*ptr2mindxplus > 0) {
			*ptr2mindxplus = 0;
		} else {
			*ptr2mindxplus *= *ptr2mindxplus;
		}
		ptr2mindxplus++;
	}

	float *minminusdxminus, *ptr2minminusdxminus;
	if((minminusdxminus=(float *)malloc(imageW*imageH*sizeof(float)))==NULL)printf("ME_-MDX-\n");
	for(int i=0;i<N;i++)minminusdxminus[i]=-dxminus[i];
	ptr2minminusdxminus = minminusdxminus;
	for(int i=0;i<N;i++){
		if (*ptr2minminusdxminus > 0) {
			*ptr2minminusdxminus = 0;
		} else {
			*ptr2minminusdxminus *= *ptr2minminusdxminus;
		}
		ptr2minminusdxminus++;
	}

	//////// DY ////////

	float *dy, *ptr2dy;
	if((dy=(float *)malloc(imageW*imageH*sizeof(float)))==NULL)printf("ME_dy\n");
	ptr2dy=dy;
	ptr2phi=phi;
	for(int c=0;c<imageW;c++){
		*ptr2dy = 0;
		ptr2dy++;
		ptr2phi++;
	}
	for(int r=0;r<imageH-2;r++){
		for(int c=0;c<imageW;c++){
			*ptr2dy = (*(ptr2phi-imageW) - *(ptr2phi+imageW))/2;
			ptr2dy++;
			ptr2phi++;
		}
	}
	for(int c=0;c<imageW;c++){
		*ptr2dy = 0;
		ptr2dy++;
		ptr2phi++;
	}

	float *dyplus, *ptr2dyplus;
	if((dyplus=(float *)malloc(imageW*imageH*sizeof(float)))==NULL)printf("ME_dy+\n");
	for(int i=0;i<N;i++)dyplus[i]=phi[i];
	ptr2dyplus=dyplus+N-1;
	for(int r=0;r<imageH-1;r++){
		for(int c=0;c<imageW;c++){
			*ptr2dyplus = *(ptr2dyplus-imageW) - *ptr2dyplus;
			ptr2dyplus--;
		}
	}
	for(int c=0;c<imageW;c++){
		*ptr2dyplus = 0;
		ptr2dyplus--;
	}

	float *dyminus, *ptr2dyminus;
	if((dyminus=(float *)malloc(imageW*imageH*sizeof(float)))==NULL)printf("ME_dy+\n");
	for(int i=0;i<N;i++)dyminus[i]=phi[i];
	ptr2dyminus=dyminus;
	for(int r=0;r<imageH-1;r++){
		for(int c=0;c<imageW;c++){
			*ptr2dyminus = *ptr2dyminus - *(ptr2dyminus+imageW);
			ptr2dyminus++;
		}
	}
	for(int c=0;c<imageW;c++){
		*ptr2dyminus = 0;
		ptr2dyminus++;
	}

	float *dyplusx, *ptr2dyplusx;
	if((dyplusx=(float *)malloc(imageW*imageH*sizeof(float)))==NULL)printf("ME_dyplusx\n");
	ptr2dyplusx=dyplusx;
	ptr2phi=phi;
	for(int c=0;c<imageW;c++){
		*ptr2dyplusx = 0;
		ptr2dyplusx++;
		ptr2phi++;
	}
	for(int r=0;r<imageH-2;r++){
		for(int c=0;c<imageW-1;c++){
			ptr2phi++;
			*ptr2dyplusx = (*(ptr2phi-imageW) - *(ptr2phi+imageW))/2;
			ptr2dyplusx++;
		}
		*ptr2dyplusx=0;
		ptr2dyplusx++;
		ptr2phi++;
	}
	for(int c=0;c<imageW;c++){
		*ptr2dyplusx = 0;
		ptr2dyplusx++;
	}

	float *dyminusx, *ptr2dyminusx;
	if((dyminusx=(float *)malloc(imageW*imageH*sizeof(float)))==NULL)printf("ME_dyminusx\n");
	ptr2dyminusx=dyminusx;
	ptr2phi=phi;
	for(int c=0;c<imageW;c++){
		*ptr2dyminusx = 0;
		ptr2dyminusx++;
		ptr2phi++;
	}
	for(int r=0;r<imageH-2;r++){
		*ptr2dyminusx=0;
		for(int c=0;c<imageW-1;c++){
			ptr2dyminusx++;
			*ptr2dyminusx = (*(ptr2phi-imageW) - *(ptr2phi+imageW))/2;
			ptr2phi++;
		}
		ptr2dyminusx++;
		ptr2phi++;
	}
	for(int c=0;c<imageW;c++){
		*ptr2dyminusx = 0;
		ptr2dyminusx++;
	}

	float *maxdyplus, *ptr2maxdyplus;
	if((maxdyplus=(float *)malloc(imageW*imageH*sizeof(float)))==NULL)printf("ME_Mdy+\n");
	for(int i=0;i<N;i++)maxdyplus[i]=dyplus[i];
	ptr2maxdyplus = maxdyplus;
	for(int i=0;i<N;i++){
		if (*ptr2maxdyplus < 0) {
			*ptr2maxdyplus = 0;
		} else {
			*ptr2maxdyplus *= *ptr2maxdyplus;
		}
		ptr2maxdyplus++;
	}

	float *maxminusdyminus, *ptr2maxminusdyminus;
	if((maxminusdyminus=(float *)malloc(imageW*imageH*sizeof(float)))==NULL)printf("ME_-Mdy-\n");
	for(int i=0;i<N;i++)maxminusdyminus[i]=-dyminus[i];
	ptr2maxminusdyminus = maxminusdyminus;
	for(int i=0;i<N;i++){
		if (*ptr2maxminusdyminus < 0) {
			*ptr2maxminusdyminus = 0;
		} else {
			*ptr2maxminusdyminus *= *ptr2maxminusdyminus;
		}
		ptr2maxminusdyminus++;
	}

	float *mindyplus, *ptr2mindyplus;
	if((mindyplus=(float *)malloc(imageW*imageH*sizeof(float)))==NULL)printf("ME_Mdy+\n");
	for(int i=0;i<N;i++)mindyplus[i]=dyplus[i];
	ptr2mindyplus = mindyplus;
	for(int i=0;i<N;i++){
		if (*ptr2mindyplus > 0) {
			*ptr2mindyplus = 0;
		} else {
			*ptr2mindyplus *= *ptr2mindyplus;
		}
		ptr2mindyplus++;
	}

	float *minminusdyminus, *ptr2minminusdyminus;
	if((minminusdyminus=(float *)malloc(imageW*imageH*sizeof(float)))==NULL)printf("ME_-Mdy-\n");
	for(int i=0;i<N;i++)minminusdyminus[i]=-dyminus[i];
	ptr2minminusdyminus = minminusdyminus;
	for(int i=0;i<N;i++){
		if (*ptr2minminusdyminus > 0) {
			*ptr2minminusdyminus = 0;
		} else {
			*ptr2minminusdyminus *= *ptr2minminusdyminus;
		}
		ptr2minminusdyminus++;
	}

		float *gradphimax;
		if((gradphimax=(float *)malloc(imageW*imageH*sizeof(float)))==NULL)printf("GRADPHIMAX\n");
		for(int i=0;i<N;i++){
			gradphimax[i]=sqrt((sqrt(maxdxplus[i]+maxminusdxminus[i]))*(sqrt(maxdxplus[i]+maxminusdxminus[i]))+(sqrt(maxdyplus[i]+maxminusdyminus[i]))*(sqrt(maxdyplus[i]+maxminusdyminus[i])));
		}

		float *gradphimin;
		if((gradphimin=(float *)malloc(imageW*imageH*sizeof(float)))==NULL)printf("GRADPHImin\n");
		for(int i=0;i<N;i++){
			gradphimin[i]=sqrt((sqrt(mindxplus[i]+minminusdxminus[i]))*(sqrt(mindxplus[i]+minminusdxminus[i]))+(sqrt(mindyplus[i]+minminusdyminus[i]))*(sqrt(mindyplus[i]+minminusdyminus[i])));
		}

			float *nplusx;
			if((nplusx=(float *)malloc(imageW*imageH*sizeof(float)))==NULL)printf("nplusx\n");
			for(int i=0;i<N;i++){
				nplusx[i]= dxplus[i] / sqrt(FLT_EPSILON + (dxplus[i]*dxplus[i]) + ((dyplusx[i] + dy[i])*(dyplusx[i] + dy[i])*0.25) );
			}
			float *nplusy;
			if((nplusy=(float *)malloc(imageW*imageH*sizeof(float)))==NULL)printf("nplusy\n");
			for(int i=0;i<N;i++){
				nplusy[i]= dyplus[i] / sqrt(FLT_EPSILON + (dyplus[i]*dyplus[i]) + ((dxplusy[i] + dx[i])*(dxplusy[i] + dx[i])*0.25) );
			}			
			
			float *nminusx;
			if((nminusx=(float *)malloc(imageW*imageH*sizeof(float)))==NULL)printf("nminusx\n");
			for(int i=0;i<N;i++){
				nminusx[i]= dxminus[i] / sqrt(FLT_EPSILON + (dxminus[i]*dxminus[i]) + ((dyminusx[i] + dy[i])*(dyminusx[i] + dy[i])*0.25) );
			}			
			
			float *nminusy;
			if((nminusy=(float *)malloc(imageW*imageH*sizeof(float)))==NULL)printf("nminusy\n");
			for(int i=0;i<N;i++){
				nminusy[i]= dyminus[i] / sqrt(FLT_EPSILON + (dyminus[i]*dyminus[i]) + ((dxminusy[i] + dx[i])*(dxminusy[i] + dx[i])*0.25) );
			}

			float *curvature, *ptr2curvature;
			if((curvature=(float *)malloc(imageW*imageH*sizeof(float)))==NULL)printf("curvature\n");
			for(int i=0;i<N;i++){
				curvature[i]= ((nplusx[i]-nminusx[i])+(nplusy[i]-nminusy[i])/2);
			}
	

		float *F, *ptr2F, *ptr2D;
		if((F=(float *)malloc(imageW*imageH*sizeof(float)))==NULL)printf("F\n");
		ptr2F=F;
		ptr2D=D;
		ptr2curvature=curvature;
		for(int i=0;i<N;i++){
			*ptr2F = -(ALPHA * (*ptr2D)) + ( (1-ALPHA) * (*ptr2curvature));
			ptr2F++;
			ptr2D++;
			ptr2curvature++;
		}

		float *gradphi, *ptr2gradphi, *ptr2gradphimax, *ptr2gradphimin;
		if((gradphi=(float *)malloc(imageW*imageH*sizeof(float)))==NULL)printf("GRADPHI\n");
		ptr2gradphi=gradphi;
		ptr2gradphimax=gradphimax;
		ptr2gradphimin=gradphimin;
		ptr2F=F;
		for(int i=0; i<N; i++){
			if(*ptr2F>0){
				*ptr2gradphi = *ptr2gradphimax;
			} else {
				*ptr2gradphi = *ptr2gradphimin;
			}
			ptr2gradphi++;
			ptr2F++;
			ptr2gradphimax++;
			ptr2gradphimin++;
		}

		ptr2phi=phi;
		ptr2gradphi=gradphi;
		ptr2F=F;
		for(int i=0; i<N; i++){
			*ptr2phi = *ptr2phi + DT * (*ptr2F) * (*ptr2gradphi);
			ptr2phi++;
			ptr2gradphi++;
			ptr2F++;
		}




//printf("Freeing Memory\n");




free(dx);
free(dxplus);
free(dxminus);
free(dxminusy);
free(maxdxplus);
free(maxminusdxminus);
free(mindxplus);
free(minminusdxminus);
//
free(dy);
free(dyplus);
free(dyminus);
free(dyminusx);
free(maxdyplus);
free(maxminusdyminus);
free(mindyplus);
free(minminusdyminus);
//
free(gradphi);
free(gradphimax);
free(gradphimin);
//
free(nplusx);
free(nplusy);
free(nminusx);
free(nminusy);
free(curvature);

free(F);

}




int main(int argc, char** argv){

	const char *image_path = IMAGE;

	LoadBMPFile(&h_Src, &imageW, &imageH, image_path);
	D = (float *)malloc(imageW*imageH*sizeof(float));

	//printf("Input Image\n");
	for(int r=0;r<imageH;r++){
		for(int c=0;c<imageW;c++){
			D[r*imageW+c] = h_Src[r*imageW+c].x;
			/*printf("%3.0f ", D[r*imageW+c]);*/
		}
		//printf("\n");
	}

	N = imageW*imageH;

	float *ptr2D;
	ptr2D=D;
	for(int i=0;i<N;i++){
		*ptr2D = EPSILON - abs(*ptr2D - THRESHOLD);
		ptr2D++;
	}

	init_phi();

	// Set up CUDA Timer
	cutCreateTimer(&Timer);
	cutStartTimer(Timer);

	for(its=0;its<ITERATIONS;its++){
		update_phi();
		if(its%50==0)printf("Iteration %3d Total Time: %3.2f\n", its, 0.001*cutGetTimerValue(Timer));
	}
}