#define ALPHA		 0.02
#define DT			 0.1

#define max(x,y)    ((x>y) ? x : y )
#define min(x,y)    ((x<y) ? x : y )

#define INDEX(i,j,j_off)  (i +__mul24(j,j_off))

#define BLOCKDIM_X	 32
#define BLOCKDIM_Y	 4
#define BLOCKDIM_Z	 1

__global__ void updatephi( float *d_phi, float *d_phi1, float *d_D, int imageW, int imageH, int imageD, int pitch)
{

	float dx,dy,dz;
	float dxplus, dyplus, dzplus, dxminus, dyminus, dzminus;
	float dxplusy, dxminusy, dxplusz, dxminusz, dyplusx, dyminusx, dyplusz, dyminusz, dzplusx, dzminusx, dzplusy, dzminusy;

	float gradphimax, gradphimin, nplusx, nplusy, nplusz, nminusx, nminusy, nminusz, curvature;
	float F, gradphi;

	int   indg, indg_h, indg0;
	int   i, j, k, ind, ind_h, halo, active;

	int IOFF  = 1;
	int JOFF  = (BLOCKDIM_X+2);
	int KOFF  = (BLOCKDIM_X+2)*(BLOCKDIM_Y+2);

	__shared__ float s_data[3*(BLOCKDIM_X+2)*(BLOCKDIM_Y+2)];

	k    =  threadIdx.y*BLOCKDIM_X + threadIdx.x;
	halo = k < 2*(BLOCKDIM_X+BLOCKDIM_Y+2);

	if (halo) {
		if (threadIdx.y<2) {               // y-halos (coalesced)
			i = threadIdx.x;
			j = threadIdx.y*(BLOCKDIM_Y+1) - 1;
		}
		else {                             // x-halos (not coalesced)
			i = (k%2)*(BLOCKDIM_X+1) - 1;
			j =  k/2 - BLOCKDIM_X - 1;
		}

		ind_h  = INDEX(i+1,j+1,BLOCKDIM_X+2)+KOFF;

		i      = INDEX(i,blockIdx.x,BLOCKDIM_X);   // global indices
		j      = INDEX(j,blockIdx.y,BLOCKDIM_Y);
		indg_h = INDEX(i,j,pitch);

		halo   =  (i>=0) && (i<imageW) && (j>=0) && (j<imageH);
	}

	//
	// then set up indices for main block
	//

	i    = threadIdx.x;
	j    = threadIdx.y;
	ind  = INDEX(i+1,j+1,BLOCKDIM_X+2) ;

	i    = INDEX(i,blockIdx.x,BLOCKDIM_X);     // global indices
	j    = INDEX(j,blockIdx.y,BLOCKDIM_Y);
	indg = INDEX(i,j,pitch);

	active = (i<imageW) && (j<imageH);

	//
	// read initial plane of u1 array
	//

	if (active) s_data[ind+KOFF+KOFF] = d_phi1[indg];
	if (halo) s_data[ind_h+KOFF+KOFF] = d_phi1[indg_h];

	

	for(int k=0;k<imageD;k++){

		if (active) {
			indg0 = indg;
			indg  = INDEX(indg,imageH,pitch);
			s_data[ind-KOFF] = s_data[ind];
			s_data[ind]      = s_data[ind+KOFF];
			if (k<imageD-1)
				s_data[ind+KOFF] = d_phi1[indg];
		}

		if (halo) {
			indg_h = INDEX(indg_h,imageH,pitch);
			s_data[ind_h-KOFF] = s_data[ind_h];
			s_data[ind_h]      = s_data[ind_h+KOFF];
			if (k<imageD-1)
				s_data[ind_h+KOFF] = d_phi1[indg_h];
		}

		

		if (active) {

			if(i==0||i==imageW-1){dx=0;} else {dx=(s_data[ind+IOFF]-s_data[ind-IOFF])/2;}
			if(j==0||j==imageH-1){dy=0;} else {dy=(s_data[ind-JOFF]-s_data[ind+JOFF])/2;}
			if(k==0||k==imageD-1){dz=0;} else {dz=(s_data[ind+KOFF]-s_data[ind-KOFF])/2;}

			if(i==imageW-1){dxplus=0;}   else {dxplus =(s_data[ind+IOFF]-s_data[ind     ]);}
			if(j==0){dyplus=0;}		     else {dyplus =(s_data[ind-JOFF]-s_data[ind     ]);}
			if(k==imageD-1){dzplus=0;}   else {dzplus =(s_data[ind+KOFF]-s_data[ind     ]);}
			if(i==0){dxminus=0;}         else {dxminus=(s_data[ind     ]-s_data[ind-IOFF]);}
			if(j==imageH-1){dyminus=0;}  else {dyminus=(s_data[ind     ]-s_data[ind+JOFF]);}
			if(k==0){dzminus=0;}         else {dzminus=(s_data[ind     ]-s_data[ind-KOFF]);}

			if(i==0||i==imageW-1||j==0){dxplusy=0;}			 else {dxplusy =(s_data[ind-JOFF+IOFF]-s_data[ind-JOFF-IOFF])/2;}
			if(i==0||i==imageW-1||j==imageH-1){dxminusy=0;}  else {dxminusy=(s_data[ind+JOFF+IOFF]-s_data[ind+JOFF-IOFF])/2;}
			if(i==0||i==imageW-1||k==imageD-1) {dxplusz=0;}  else {dxplusz =(s_data[ind+KOFF+IOFF]-s_data[ind+KOFF-IOFF])/2;}
			if(i==0||i==imageW-1||k==0) {dxminusz=0;}		 else {dxminusz=(s_data[ind-KOFF+IOFF]-s_data[ind-KOFF-IOFF])/2;}
			if(j==0||j==imageH-1||i==imageW-1){dyplusx=0;}   else {dyplusx =(s_data[ind-JOFF+IOFF]-s_data[ind+JOFF+IOFF])/2;}
			if(j==0||j==imageH-1||i==0){dyminusx=0;}		 else {dyminusx=(s_data[ind-JOFF-IOFF]-s_data[ind+JOFF-IOFF])/2;}
			if(j==0||j==imageH-1||k==imageD-1) {dyplusz=0;}  else {dyplusz =(s_data[ind+KOFF-JOFF]-s_data[ind+KOFF+JOFF])/2;}
			if(j==0||j==imageH-1||k==0) {dyminusz=0;}		 else {dyminusz=(s_data[ind-KOFF-JOFF]-s_data[ind-KOFF+JOFF])/2;}
			if(k==0||k==imageD-1||i==imageW-1) {dzplusx=0;}  else {dzplusx =(s_data[ind+IOFF+KOFF]-s_data[ind+IOFF-KOFF])/2;}
			if(k==0||k==imageD-1||i==0) {dzminusx=0;}		 else {dzminusx=(s_data[ind-IOFF+KOFF]-s_data[ind-IOFF-KOFF])/2;}
			if(k==0||k==imageD-1||j==0) {dzplusy=0;}		 else {dzplusy =(s_data[ind-JOFF+KOFF]-s_data[ind-JOFF-KOFF])/2;}
			if(k==0||k==imageD-1||j==imageH-1) {dzminusy=0;} else {dzminusy=(s_data[ind+JOFF+KOFF]-s_data[ind+JOFF-KOFF])/2;}


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

			F = (-ALPHA * d_D[indg0]) + ((1-ALPHA) * curvature);
			if(F>0) {gradphi=gradphimax;} else {gradphi=gradphimin;}
			d_phi[indg0]=s_data[ind] + (DT * F * gradphi);
		}

		__syncthreads();

	}

}