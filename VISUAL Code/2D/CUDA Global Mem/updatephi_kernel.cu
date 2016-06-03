
#define ALPHA		 0.007
#define DT			 0.2

#define max(x,y)    ((x>y) ? x : y )
#define min(x,y)    ((x<y) ? x : y )

__global__ void updatephi( float *d_phi, float *d_phi1, float *d_D, int imageW, int imageH)

{
	int c= blockIdx.x * blockDim.x + threadIdx.x;
	int r= blockIdx.y * blockDim.y + threadIdx.y;
	int ind= r*imageW+c;

	if(ind<imageW*imageH){

		float dx,dxplus,dxminus,dxplusy,dxminusy;

		float dy, dyplus,dyminus,dyplusx,dyminusx;

		float gradphimax, gradphimin, nplusx, nplusy, nminusx, nminusy, curvature;
		float F, gradphi;

		if(c==0||c==imageW-1){dx=0;} else {dx=(d_phi1[ind+1]-d_phi1[ind-1])/2;}
		if(c==imageW-1){dxplus=0;} else {dxplus=(d_phi1[ind+1]-d_phi1[ind]);}
		if(c==0){dxminus=0;} else {dxminus=(d_phi1[ind]-d_phi1[ind-1]);}
		if(r==0||c==0||c==imageW-1){dxplusy=0;} else {dxplusy=(d_phi1[ind-imageW+1]-d_phi1[ind-imageW-1])/2;}
		if(r==imageH-1||c==0||c==imageW-1){dxminusy=0;} else {dxminusy=(d_phi1[ind+imageW+1]-d_phi1[ind+imageW-1])/2;}

		if(r==0||r==imageH-1){dy=0;} else {dy=(d_phi1[ind-imageW]-d_phi1[ind+imageW])/2;}
		if(r==0){dyplus=0;} else {dyplus=(d_phi1[ind-imageW]-d_phi1[ind]);}
		if(r==imageH-1){dyminus=0;} else {dyminus=(d_phi1[ind]-d_phi1[ind+imageW]);}
		if(r==0||c==imageW-1||r==imageH-1){dyplusx=0;} else {dyplusx=(d_phi1[ind-imageW+1]-d_phi1[ind+imageW+1])/2;}
		if(r==0||c==0||r==imageH-1){dyminusx=0;} else {dyminusx=(d_phi1[ind-imageW-1]-d_phi1[ind+imageW-1])/2;}

		gradphimax=sqrt((sqrt(max(dxplus,0)*max(dxplus,0)+max(-dxminus,0)*max(-dxminus,0)))*(sqrt(max(dxplus,0)*max(dxplus,0)+max(-dxminus,0)*max(-dxminus,0)))
					   +(sqrt(max(dyplus,0)*max(dyplus,0)+max(-dyminus,0)*max(-dyminus,0)))*(sqrt(max(dyplus,0)*max(dyplus,0)+max(-dyminus,0)*max(-dyminus,0))));
		
		gradphimin=sqrt((sqrt(min(dxplus,0)*min(dxplus,0)+min(-dxminus,0)*min(-dxminus,0)))*(sqrt(min(dxplus,0)*min(dxplus,0)+min(-dxminus,0)*min(-dxminus,0)))
					   +(sqrt(min(dyplus,0)*min(dyplus,0)+min(-dyminus,0)*min(-dyminus,0)))*(sqrt(min(dyplus,0)*min(dyplus,0)+min(-dyminus,0)*min(-dyminus,0))));
		
		nplusx= dxplus / sqrt(1.192092896e-07F + (dxplus*dxplus) + ((dyplusx + dy)*(dyplusx + dy)*0.25) );
		nplusy= dyplus / sqrt(1.192092896e-07F + (dyplus*dyplus) + ((dxplusy + dx)*(dxplusy + dx)*0.25) );
		nminusx= dxminus / sqrt(1.192092896e-07F + (dxminus*dxminus) + ((dyminusx + dy)*(dyminusx + dy)*0.25) );
		nminusy= dyminus / sqrt(1.192092896e-07F + (dyminus*dyminus) + ((dxminusy + dx)*(dxminusy + dx)*0.25) );
		curvature= ((nplusx-nminusx)+(nplusy-nminusy))/2;
		
		F = (-ALPHA * d_D[ind]) + ((1-ALPHA) * curvature);
		if(F>0) {gradphi=gradphimax;} else {gradphi=gradphimin;}
		d_phi[ind]=d_phi1[ind] + (DT * F * gradphi);
	
}
}
		



