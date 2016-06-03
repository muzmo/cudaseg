
#define ALPHA		 0.02
#define DT			 0.1

#define max(x,y)    ((x>y) ? x : y )
#define min(x,y)    ((x<y) ? x : y )

__global__ void updatephi( float *d_phi, float *d_phi1, float *d_D, int imageW, int imageH, int imageD, int pitch)
{

	float dx,dy,dz;
	float dxplus, dyplus, dzplus, dxminus, dyminus, dzminus;
	float dxplusy, dxminusy, dxplusz, dxminusz, dyplusx, dyminusx, dyplusz, dyminusz, dzplusx, dzminusx, dzplusy, dzminusy;

	float gradphimax, gradphimin, nplusx, nplusy, nplusz, nminusx, nminusy, nminusz, curvature;
	float F, gradphi;

	int ind, jOFF, kOFF;

	int i= blockIdx.x * blockDim.x + threadIdx.x;
	int j= blockIdx.y * blockDim.y + threadIdx.y;

	jOFF=pitch;
	kOFF=pitch*imageH;

	if(i<imageW&&j<imageH){

		ind=i+j*jOFF;

		for(int k=0;k<imageD;k++){

		if(i==0||i==imageW-1){dx=0;} else {dx=(d_phi1[ind+1   ]-d_phi1[ind-1   ])/2;}
		if(j==0||j==imageH-1){dy=0;} else {dy=(d_phi1[ind-jOFF]-d_phi1[ind+jOFF])/2;}
		if(k==0||k==imageD-1){dz=0;} else {dz=(d_phi1[ind+kOFF]-d_phi1[ind-kOFF])/2;}

		if(i==imageW-1){dxplus=0;}   else {dxplus =(d_phi1[ind+1   ]-d_phi1[ind     ]);}
		if(j==0){dyplus=0;}		     else {dyplus =(d_phi1[ind-jOFF]-d_phi1[ind     ]);}
		if(k==imageD-1){dzplus=0;}   else {dzplus =(d_phi1[ind+kOFF]-d_phi1[ind     ]);}
		if(i==0){dxminus=0;}         else {dxminus=(d_phi1[ind     ]-d_phi1[ind-1   ]);}
		if(j==imageH-1){dyminus=0;}  else {dyminus=(d_phi1[ind     ]-d_phi1[ind+jOFF]);}
		if(k==0){dzminus=0;}         else {dzminus=(d_phi1[ind     ]-d_phi1[ind-kOFF]);}

		if(i==0||i==imageW-1||j==0){dxplusy=0;}			 else {dxplusy =(d_phi1[ind-jOFF+1   ]-d_phi1[ind-jOFF-1   ])/2;}
		if(i==0||i==imageW-1||j==imageH-1){dxminusy=0;}  else {dxminusy=(d_phi1[ind+jOFF+1   ]-d_phi1[ind+jOFF-1   ])/2;}
		if(i==0||i==imageW-1||k==imageD-1) {dxplusz=0;}  else {dxplusz =(d_phi1[ind+kOFF+1   ]-d_phi1[ind+kOFF-1   ])/2;}
		if(i==0||i==imageW-1||k==0) {dxminusz=0;}		 else {dxminusz=(d_phi1[ind-kOFF+1   ]-d_phi1[ind-kOFF-1   ])/2;}
		if(j==0||j==imageH-1||i==imageW-1){dyplusx=0;}   else {dyplusx =(d_phi1[ind-jOFF+1   ]-d_phi1[ind+jOFF+1   ])/2;}
		if(j==0||j==imageH-1||i==0){dyminusx=0;}		 else {dyminusx=(d_phi1[ind-jOFF-1   ]-d_phi1[ind+jOFF-1   ])/2;}
		if(j==0||j==imageH-1||k==imageD-1) {dyplusz=0;}  else {dyplusz =(d_phi1[ind+kOFF-jOFF]-d_phi1[ind+kOFF+jOFF])/2;}
		if(j==0||j==imageH-1||k==0) {dyminusz=0;}		 else {dyminusz=(d_phi1[ind-kOFF-jOFF]-d_phi1[ind-kOFF+jOFF])/2;}
		if(k==0||k==imageD-1||i==imageW-1) {dzplusx=0;}  else {dzplusx =(d_phi1[ind+1+kOFF   ]-d_phi1[ind+1-kOFF   ])/2;}
		if(k==0||k==imageD-1||i==0) {dzminusx=0;}		 else {dzminusx=(d_phi1[ind-1+kOFF   ]-d_phi1[ind-1-kOFF   ])/2;}
		if(k==0||k==imageD-1||j==0) {dzplusy=0;}		 else {dzplusy =(d_phi1[ind-jOFF+kOFF]-d_phi1[ind-jOFF-kOFF])/2;}
		if(k==0||k==imageD-1||j==imageH-1) {dzminusy=0;} else {dzminusy=(d_phi1[ind+jOFF+kOFF]-d_phi1[ind+jOFF-kOFF])/2;}


		gradphimax=sqrt((sqrt(max(dxplus,0)*max(dxplus,0)+max(-dxminus,0)*max(-dxminus,0)))*(sqrt(max(dxplus,0)*max(dxplus,0)+max(-dxminus,0)*max(-dxminus,0)))
			+(sqrt(max(dyplus,0)*max(dyplus,0)+max(-dyminus,0)*max(-dyminus,0)))*(sqrt(max(dyplus,0)*max(dyplus,0)+max(-dyminus,0)*max(-dyminus,0)))
			+(sqrt(max(dzplus,0)*max(dzplus,0)+max(-dzminus,0)*max(-dzminus,0)))*(sqrt(max(dzplus,0)*max(dzplus,0)+max(-dzminus,0)*max(-dzminus,0))));

		gradphimin=sqrt((sqrt(min(dxplus,0)*min(dxplus,0)+min(-dxminus,0)*min(-dxminus,0)))*(sqrt(min(dxplus,0)*min(dxplus,0)+min(-dxminus,0)*min(-dxminus,0)))
			+(sqrt(min(dyplus,0)*min(dyplus,0)+min(-dyminus,0)*min(-dyminus,0)))*(sqrt(min(dyplus,0)*min(dyplus,0)+min(-dyminus,0)*min(-dyminus,0)))
			+(sqrt(min(dzplus,0)*min(dzplus,0)+min(-dzminus,0)*min(-dzminus,0)))*(sqrt(min(dzplus,0)*min(dzplus,0)+min(-dzminus,0)*min(-dzminus,0))));

		nplusx = dxplus / sqrt(1.192092896e-07F + (dxplus*dxplus) + ((dyplusx + dy)*(dyplusx + dy)*0.25) + ((dzplusx + dz)*(dzplusx + dz)*0.25));
		nplusy = dyplus / sqrt(1.192092896e-07F + (dyplus*dyplus) + ((dxplusy + dx)*(dxplusy + dx)*0.25) + ((dzplusy + dz)*(dzplusy + dz)*0.25));
		nplusz = dzplus / sqrt(1.192092896e-07F + (dzplus*dzplus) + ((dxplusz + dz)*(dxplusz + dz)*0.25) + ((dyplusz + dy)*(dyplusz + dy)*0.25));

		nminusx=dxminus / sqrt(1.192092896e-07F + (dxminus*dxminus) + ((dyminusx + dy)*(dyminusx + dy)*0.25) + ((dzminusx + dz)*(dzminusx + dz)*0.25));
		nminusy=dyminus / sqrt(1.192092896e-07F + (dyminus*dyminus) + ((dxminusy + dx)*(dxminusy + dx)*0.25) + ((dzminusy + dz)*(dzminusy + dz)*0.25));
		nminusz=dzminus / sqrt(1.192092896e-07F + (dzminus*dzminus) + ((dxminusz + dz)*(dxminusz + dz)*0.25) + ((dyminusz + dy)*(dyminusz + dy)*0.25));

		curvature= ((nplusx-nminusx)+(nplusy-nminusy)+(nplusz-nminusz))/2;

		F = (-ALPHA * d_D[ind]) + ((1-ALPHA) * curvature);
		if(F>0) {gradphi=gradphimax;} else {gradphi=gradphimin;}
		d_phi[ind]=d_phi1[ind] + (DT * F * gradphi);

		ind += kOFF;
		
		}
	}
}