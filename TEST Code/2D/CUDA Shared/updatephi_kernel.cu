
#define ALPHA		 0.003
#define DT			 0.2

#define max(x,y)    ((x>y) ? x : y )
#define min(x,y)    ((x<y) ? x : y )

#define INDEX(i,j,j_off)  (i +__mul24(j,j_off))

#define BLOCKDIM_X	 16
#define BLOCKDIM_Y	 32

__global__ void updatephi( float *d_phi, float *d_phi1, float *d_D, int imageW, int imageH, int pitch)

{
	int c= blockIdx.x * blockDim.x + threadIdx.x;
	int r= blockIdx.y * blockDim.y + threadIdx.y;
	int ind= r*pitch+c;

	int   indg, indg_h, indg0;
	int   i, j, k, ind2, ind_h, halo, active;

	__shared__ float s_data[(BLOCKDIM_X+2)*(BLOCKDIM_Y+2)];

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

		ind_h  = INDEX(i+1,j+1,BLOCKDIM_X+2);

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
	ind2  = INDEX(i+1,j+1,BLOCKDIM_X+2) ;

	i    = INDEX(i,blockIdx.x,BLOCKDIM_X);     // global indices
	j    = INDEX(j,blockIdx.y,BLOCKDIM_Y);
	indg = INDEX(i,j,pitch);

	active = (i<imageW) && (j<imageH);

	//
	// read initial plane of u1 array
	//

	if (active) s_data[ind2] = d_phi1[indg];
	if (halo) s_data[ind_h] = d_phi1[indg_h];

	__syncthreads();

	
	if(active){

		if(r<imageW&&c<imageH){

			float dx,dxplus,dxminus,dxplusy,dxminusy;

			float dy, dyplus,dyminus,dyplusx,dyminusx;

			float gradphimax, gradphimin, nplusx, nplusy, nminusx, nminusy, curvature;
			float F, gradphi;

			if(c==0||c==imageW-1){dx=0;} else {dx=(s_data[ind2+1]-s_data[ind2-1])/2;}
			if(c==imageW-1){dxplus=0;} else {dxplus=(s_data[ind2+1]-s_data[ind2]);}
			if(c==0){dxminus=0;} else {dxminus=(s_data[ind2]-s_data[ind2-1]);}
			if(r==0||c==0||c==imageW-1){dxplusy=0;} else {dxplusy=(s_data[ind2-(BLOCKDIM_X+2)+1]-s_data[ind2-(BLOCKDIM_X+2)-1])/2;}
			if(r==imageH-1||c==0||c==imageW-1){dxminusy=0;} else {dxminusy=(s_data[ind2+(BLOCKDIM_X+2)+1]-s_data[ind2+(BLOCKDIM_X+2)-1])/2;}

			if(r==0||r==imageH-1){dy=0;} else {dy=(s_data[ind2-(BLOCKDIM_X+2)]-s_data[ind2+(BLOCKDIM_X+2)])/2;}
			if(r==0){dyplus=0;} else {dyplus=(s_data[ind2-(BLOCKDIM_X+2)]-s_data[ind2]);}
			if(r==imageH-1){dyminus=0;} else {dyminus=(s_data[ind2]-s_data[ind2+(BLOCKDIM_X+2)]);}
			if(r==0||c==imageW-1||r==imageH-1){dyplusx=0;} else {dyplusx=(s_data[ind2-(BLOCKDIM_X+2)+1]-s_data[ind2+(BLOCKDIM_X+2)+1])/2;}
			if(r==0||c==0||r==imageH-1){dyminusx=0;} else {dyminusx=(s_data[ind2-(BLOCKDIM_X+2)-1]-s_data[ind2+(BLOCKDIM_X+2)-1])/2;}

			gradphimax=sqrt((sqrt(max(dxplus,0)*max(dxplus,0)+max(-dxminus,0)*max(-dxminus,0)))*(sqrt(max(dxplus,0)*max(dxplus,0)+max(-dxminus,0)*max(-dxminus,0)))
				+(sqrt(max(dyplus,0)*max(dyplus,0)+max(-dyminus,0)*max(-dyminus,0)))*(sqrt(max(dyplus,0)*max(dyplus,0)+max(-dyminus,0)*max(-dyminus,0))));

			gradphimin=sqrt((sqrt(min(dxplus,0)*min(dxplus,0)+min(-dxminus,0)*min(-dxminus,0)))*(sqrt(min(dxplus,0)*min(dxplus,0)+min(-dxminus,0)*min(-dxminus,0)))
				+(sqrt(min(dyplus,0)*min(dyplus,0)+min(-dyminus,0)*min(-dyminus,0)))*(sqrt(min(dyplus,0)*min(dyplus,0)+min(-dyminus,0)*min(-dyminus,0))));

			nplusx= dxplus / sqrt(1.192092896e-07F + (dxplus*dxplus) + ((dyplusx + dy)*(dyplusx + dy)*0.25) );
			nplusy= dyplus / sqrt(1.192092896e-07F + (dyplus*dyplus) + ((dxplusy + dx)*(dxplusy + dx)*0.25) );
			nminusx= dxminus / sqrt(1.192092896e-07F + (dxminus*dxminus) + ((dyminusx + dy)*(dyminusx + dy)*0.25) );
			nminusy= dyminus / sqrt(1.192092896e-07F + (dyminus*dyminus) + ((dxminusy + dx)*(dxminusy + dx)*0.25) );
			curvature= ((nplusx-nminusx)+(nplusy-nminusy))/2;

			F = (-ALPHA * d_D[indg]) + ((1-ALPHA) * curvature);
			if(F>0) {gradphi=gradphimax;} else {gradphi=gradphimin;}
			d_phi[indg]=s_data[ind2] + (DT * F * gradphi);

		}

		__syncthreads();

			


	}
}




