#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <cutil.h>
#include <math.h>


#define max(x,y)    ((x>y) ? x : y )
#define min(x,y)    ((x<y) ? x : y )

#define DT	0.15

char *volumeFilename, *maskFilename;
int	ITERATIONS, THRESHOLD, EPSILON;
float alpha;

float *phi, *D, *contour;
size_t size;
unsigned char *input,*output;
int imageW, imageH, imageD, N;

int its=0;
unsigned int Timer = 0;
unsigned int IterationTimer = 0;

int i,j,k;



unsigned char* loadRawUchar(char *filename, size_t size){

	FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error opening file '%s'\n", filename);
        return 0;
    }

	unsigned char *data = (unsigned char *) malloc(size);
	size_t read = fread(data, 1, size, fp);
	fclose(fp);

    printf("Read '%s', %d bytes\n", filename, read);

    return data;
}

float *loadMask(char *filename, size_t size){

	FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error opening file '%s'\n", filename);
        return 0;
    }

	float *data = (float *) malloc(size*sizeof(float));
	size_t read = fread(data, 1, size, fp);
	fclose(fp);

    printf("Read '%s', %d bytes\n", filename, read);

    return data;
}

void *writeoutput(unsigned char *data, size_t size){

	char *outputFilename= "output.raw";
	FILE *fp = fopen(outputFilename, "w");
    if (!fp) {
        fprintf(stderr, "Error opening file '%s'\n", outputFilename);
        return 0;
    }

	size_t write = fwrite(data, 1, size, fp);
	fclose(fp);

    printf("Write '%s', %d bytes\n", outputFilename, write);

    return 0;
}

void update_phi(){

	float dx,dy,dz;
	float dxplus, dyplus, dzplus, dxminus, dyminus, dzminus;
	float dxplusy, dxminusy, dxplusz, dxminusz, dyplusx, dyminusx, dyplusz, dyminusz, dzplusx, dzminusx, dzplusy, dzminusy;

	float gradphimax, gradphimin, nplusx, nplusy, nplusz, nminusx, nminusy, nminusz, curvature;
	float F, gradphi;

	float *phi1;

	int ind, jOFF, kOFF;

	jOFF=imageW;
	kOFF=imageW*imageH;

	phi1=(float *)malloc(imageW*imageH*imageD*sizeof(float));
	for(i=0;i<N;i++){
		phi1[i]=phi[i];
	}

	for(i=0;i<imageW;i++){
		for(j=0;j<imageH;j++){
			for(k=0;k<imageD;k++){

				ind=i+j*imageW+k*imageW*imageH;

				if(i==0||i==imageW-1){dx=0;} else {dx=(phi1[ind+1]-phi1[ind-1])/2;}
				if(j==0||j==imageH-1){dy=0;} else {dy=(phi1[ind-imageW]-phi1[ind+imageW])/2;}
				if(k==0||k==imageD-1){dz=0;} else {dz=(phi1[ind+kOFF]-phi1[ind-kOFF])/2;}

				if(i==imageW-1){dxplus=0;} else {dxplus=(phi1[ind+1]-phi1[ind]);}
				if(j==0){dyplus=0;} else {dyplus=(phi1[ind-imageW]-phi1[ind]);}
				if(k==imageD-1){dzplus=0;} else {dzplus=(phi1[ind+kOFF]-phi1[ind]);}
				if(i==0){dxminus=0;} else {dxminus=(phi1[ind]-phi1[ind-1]);}
				if(j==imageH-1){dyminus=0;} else {dyminus=(phi1[ind]-phi1[ind+imageW]);}
				if(k==0){dzminus=0;} else {dzminus=(phi1[ind]-phi1[ind-kOFF]);}

				if(i==0||i==imageW-1||j==0){dxplusy=0;} else {dxplusy=(phi1[ind-imageW+1]-phi1[ind-imageW-1])/2;}
				if(i==0||i==imageW-1||j==imageH-1){dxminusy=0;} else {dxminusy=(phi1[ind+imageW+1]-phi1[ind+imageW-1])/2;}
				if(i==0||i==imageW-1||k==imageD-1) {dxplusz=0;} else {dxplusz=(phi1[ind+kOFF+1]-phi1[ind+kOFF-1])/2;}
				if(i==0||i==imageW-1||k==0) {dxminusz=0;} else {dxminusz=(phi1[ind-kOFF+1]-phi1[ind-kOFF-1])/2;}
				if(j==0||j==imageH-1||i==imageW-1){dyplusx=0;} else {dyplusx=(phi1[ind-imageW+1]-phi1[ind+imageW+1])/2;}
				if(j==0||j==imageH-1||i==0){dyminusx=0;} else {dyminusx=(phi1[ind-imageW-1]-phi1[ind+imageW-1])/2;}
				if(j==0||j==imageH-1||k==imageD-1) {dyplusz=0;} else {dyplusz=(phi1[ind+kOFF-jOFF]-phi1[ind+kOFF+jOFF])/2;}
				if(j==0||j==imageH-1||k==0) {dyminusz=0;} else {dyminusz=(phi1[ind-kOFF-jOFF]-phi1[ind-kOFF+jOFF])/2;}
				if(k==0||k==imageD-1||i==imageW-1) {dzplusx=0;} else {dzplusx=(phi1[ind+1+kOFF]-phi1[ind+1-kOFF])/2;}
				if(k==0||k==imageD-1||i==0) {dzminusx=0;} else {dzminusx=(phi1[ind-1+kOFF]-phi1[ind-1-kOFF])/2;}
				if(k==0||k==imageD-1||j==0) {dzplusy=0;} else {dzplusy=(phi1[ind-jOFF+kOFF]-phi1[ind-jOFF-kOFF])/2;}
				if(k==0||k==imageD-1||j==imageH-1) {dzminusy=0;} else {dzminusy=(phi1[ind+jOFF+kOFF]-phi1[ind+jOFF-kOFF])/2;}


				gradphimax=sqrt((sqrt(max(dxplus,0)*max(dxplus,0)+max(-dxminus,0)*max(-dxminus,0)))*(sqrt(max(dxplus,0)*max(dxplus,0)+max(-dxminus,0)*max(-dxminus,0)))
					+(sqrt(max(dyplus,0)*max(dyplus,0)+max(-dyminus,0)*max(-dyminus,0)))*(sqrt(max(dyplus,0)*max(dyplus,0)+max(-dyminus,0)*max(-dyminus,0)))
					+(sqrt(max(dzplus,0)*max(dzplus,0)+max(-dzminus,0)*max(-dzminus,0)))*(sqrt(max(dzplus,0)*max(dzplus,0)+max(-dzminus,0)*max(-dzminus,0))));

				gradphimin=sqrt((sqrt(min(dxplus,0)*min(dxplus,0)+min(-dxminus,0)*min(-dxminus,0)))*(sqrt(min(dxplus,0)*min(dxplus,0)+min(-dxminus,0)*min(-dxminus,0)))
					+(sqrt(min(dyplus,0)*min(dyplus,0)+min(-dyminus,0)*min(-dyminus,0)))*(sqrt(min(dyplus,0)*min(dyplus,0)+min(-dyminus,0)*min(-dyminus,0)))
					+(sqrt(min(dzplus,0)*min(dzplus,0)+min(-dzminus,0)*min(-dzminus,0)))*(sqrt(min(dzplus,0)*min(dzplus,0)+min(-dzminus,0)*min(-dzminus,0))));

				nplusx= dxplus / sqrt(1.192092896e-07F + (dxplus*dxplus) + ((dyplusx + dy)*(dyplusx + dy)*0.25) + ((dzplusx + dz)*(dzplusx + dz)*0.25));
				nplusy= dyplus / sqrt(1.192092896e-07F + (dyplus*dyplus) + ((dxplusy + dx)*(dxplusy + dx)*0.25) + ((dzplusy + dz)*(dzplusy + dz)*0.25));
				nplusz= dzplus / sqrt(1.192092896e-07F + (dzplus*dzplus) + ((dxplusz + dz)*(dxplusz + dz)*0.25) + ((dyplusz + dy)*(dyplusz + dy)*0.25));

				nminusx= dxminus / sqrt(1.192092896e-07F + (dxminus*dxminus) + ((dyminusx + dy)*(dyminusx + dy)*0.25) + ((dzminusx + dz)*(dzminusx + dz)*0.25));
				nminusy= dyminus / sqrt(1.192092896e-07F + (dyminus*dyminus) + ((dxminusy + dx)*(dxminusy + dx)*0.25) + ((dzminusy + dz)*(dzminusy + dz)*0.25));
				nminusz= dzminus / sqrt(1.192092896e-07F + (dzminus*dzminus) + ((dxminusz + dz)*(dxminusz + dz)*0.25) + ((dyminusz + dy)*(dyminusz + dy)*0.25));

				curvature= ((nplusx-nminusx)+(nplusy-nminusy)+(nplusz-nminusz))/2;

				F = (-alpha * D[ind]) + ((1-alpha) * curvature);
				if(F>0) {gradphi=gradphimax;} else {gradphi=gradphimin;}
				phi[ind]=phi1[ind] + (DT * F * gradphi);

			}
		}
	}


	free(phi1);

}



int main(int argc, char** argv){

if(argc<9){
		printf("Too few command line arguments specified. Example: Seg -volume=brain_181_217_181.raw -mask=phi.raw -xsize=181 -ysize=217 -zsize=181 -iterations=1000 -threshold=150 -epsilon=50 -alpha=0.01\n");
		exit(0);
	}

	cutGetCmdLineArgumentstr( argc, (const char**) argv, "volume", &volumeFilename);
	cutGetCmdLineArgumentstr( argc, (const char**) argv, "mask", &maskFilename);
	cutGetCmdLineArgumenti( argc, (const char**) argv, "xsize", &imageW);
	cutGetCmdLineArgumenti( argc, (const char**) argv, "ysize", &imageH);
	cutGetCmdLineArgumenti( argc, (const char**) argv, "zsize", &imageD);
	cutGetCmdLineArgumenti( argc, (const char**) argv, "iterations", &ITERATIONS);
	cutGetCmdLineArgumenti( argc, (const char**) argv, "threshold", &THRESHOLD);
	cutGetCmdLineArgumenti( argc, (const char**) argv, "epsilon", &EPSILON);
	cutGetCmdLineArgumentf( argc, (const char**) argv, "alpha", &alpha);

	N=imageW*imageH*imageD;
	input = loadRawUchar(volumeFilename, N);
	phi = loadMask(maskFilename, N);

	if((D = (float *)malloc(imageW*imageH*imageD*sizeof(float)))==NULL)printf("ME_D\n");
	for(i=0;i<N;i++){
		D[i] = EPSILON - abs(input[i] - THRESHOLD);
	}

	// Set up CUDA Timer
	cutCreateTimer(&Timer);
	cutCreateTimer(&IterationTimer);

	cutStartTimer(Timer);

	for(its=0;its<ITERATIONS;its++){
		update_phi();
		if(its%10==0)printf("Iteration %3d Total Time: %3.2f ReInit Time: %3.2f\n", its, 0.001*cutGetTimerValue(Timer), 0.001*cutGetTimerValue(IterationTimer));
	}

	if((output = (unsigned char *) malloc(size))==NULL)printf("ME_OUTPUT\n");
	for(i=0;i<N;i++){
		if(phi[i]>0){output[i]=0;} else { output[i]=255; }
	}

	writeoutput(output, size);



}

