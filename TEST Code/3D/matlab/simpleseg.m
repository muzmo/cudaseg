function seg = simpleseg(I,init_mask,max_its,E,T)

%-- Create a signed distance map (SDF) from mask
phi = mask2phi(init_mask);

%main loop
for its = 1:max_its
    
    alpha=0.05;
    D = E - abs(I - T);
    K = get_curvature(phi);
    F = -alpha*D + (1-alpha)*K;
    
    dxplus=shiftR(phi)-phi;
    dyplus=shiftU(phi)-phi;
    dzplus=shiftZ(phi)-phi;
    dxminus=phi-shiftL(phi);
    dyminus=phi-shiftD(phi);
    dzminus=phi-shiftminusZ(phi);    

    gradphimax_x = sqrt(max(dxplus,0).^2+max(-dxminus,0).^2);
    gradphimin_x = sqrt(min(dxplus,0).^2+min(-dxminus,0).^2);
    gradphimax_y = sqrt(max(dyplus,0).^2+max(-dyminus,0).^2);
    gradphimin_y = sqrt(min(dyplus,0).^2+min(-dyminus,0).^2);
    gradphimax_z = sqrt(max(dzplus,0).^2+max(-dzminus,0).^2);
    gradphimin_z = sqrt(min(dzplus,0).^2+min(-dzminus,0).^2);    
    
    
    gradphimax = sqrt((gradphimax_x.^2)+(gradphimax_y.^2)+(gradphimax_z.^2));
    gradphimin = sqrt((gradphimin_x.^2)+(gradphimin_y.^2)+(gradphimin_z.^2));
    
    gradphi=(F>0).*(gradphimax) + (F<0).*(gradphimin);
    
    %stability CFL
    %dt = .5/max(max(max(abs(F.*gradphi))));
    dt=0.1;
    
    %evolve the curve
    phi = phi + dt.*(F).*gradphi;
    
    %reinitialise distance funciton every 50 iterations
    if(mod(its,50) == 0)
        phi=bwdist(phi<0)-bwdist(phi>0);
    end
    
    %intermediate output
    if(mod(its,20) == 0)
        showCurveAndPhi(I,phi,its);
        %subplot(2,2,4); surf(phi)
    end
end

%showCurveAndPhi(I,phi,its);

%make mask from SDF
seg = phi<=0; %-- Get mask from levelset

%-- whole matrix derivatives
function shift = shiftD(M)
shift = [ M(1,:,:) ; M(1:size(M,1)-1,:,:)];

function shift = shiftL(M)
shift = [ M(:,2:size(M,2),:) M(:,size(M,2),:) ];

function shift = shiftR(M)
shift = [ M(:,1,:) M(:,1:size(M,2)-1,:) ];

function shift = shiftU(M)
shift = [ M(2:size(M,2),:,:) ; M(size(M,2),:,:) ];

function shift = shiftZ(M)
M(:,:,1)=M(:,:,1);
M(:,:,2:end)=M(:,:,1:end-1);
shift=M;

function shift = shiftminusZ(M)
M(:,:,end)=M(:,:,end);
M(:,:,1:end-1)=M(:,:,2:end);
shift=M;


function curvature=get_curvature(phi)
dx=(shiftR(phi)-shiftL(phi))/2;
dy=(shiftU(phi)-shiftD(phi))/2;
dz=(shiftZ(phi)-shiftminusZ(phi))/2;
dxplus=shiftR(phi)-phi;
dyplus=shiftU(phi)-phi;
dzplus=shiftZ(phi)-phi;
dxminus=phi-shiftL(phi);
dyminus=phi-shiftD(phi);
dzminus=phi-shiftminusZ(phi); 
dxplusy =(shiftU(shiftR(phi))-shiftU(shiftL(phi)))/2;
dxminusy=(shiftD(shiftR(phi))-shiftD(shiftL(phi)))/2;
dxplusz =(shiftR(shiftZ(phi))-shiftL(shiftZ(phi)))/2;
dxminusz=(shiftR(shiftminusZ(phi))-shiftL(shiftminusZ(phi)))/2;
dyplusx =(shiftR(shiftU(phi))-shiftR(shiftD(phi)))/2;
dyminusx=(shiftL(shiftU(phi))-shiftL(shiftD(phi)))/2;
dyplusz =(shiftU(shiftZ(phi))-shiftD(shiftZ(phi)))/2;
dyminusz=(shiftU(shiftminusZ(phi))-shiftD(shiftminusZ(phi)))/2;
dzplusx =(shiftZ(shiftR(phi))-shiftminusZ(shiftR(phi)))/2;
dzminusx=(shiftZ(shiftL(phi))-shiftminusZ(shiftL(phi)))/2;
dzplusy =(shiftZ(shiftU(phi))-shiftminusZ(shiftU(phi)))/2;
dzminusy=(shiftZ(shiftD(phi))-shiftminusZ(shiftD(phi)))/2;


nplusx = dxplus./sqrt(eps+(dxplus.^2 )+((dyplusx+dy )/2).^2 + ((dzplusx+dz)/2).^2);

nplusy = dyplus./sqrt(eps+(dyplus.^2 )+((dxplusy+dx )/2).^2 + ((dzplusy+dz)/2).^2);

nplusz = dzplus./sqrt(eps+(dzplus.^2 )+((dxplusz+dx )/2).^2 + ((dyplusz+dy)/2).^2);

nminusx= dxminus./sqrt(eps+(dxminus.^2)+((dyminusx+dy)/2).^2 + ((dzminusx+dz)/2).^2);

nminusy= dyminus./sqrt(eps+(dyminus.^2)+((dxminusy+dx)/2).^2 + ((dzminusy+dz)/2).^2);

nminusz= dzminus./sqrt(eps+(dzminus.^2)+((dxminusz+dx)/2).^2 + ((dyminusz+dy)/2).^2);

curvature=((nplusx-nminusx)+(nplusy-nminusy)+(nplusz-nminusz))/2;

%---------------------------------------------------------------------
%-- AUXILIARY FUNCTIONS ----------------------------------------------
%---------------------------------------------------------------------

%-- Displays the image with curve superimposed
function showCurveAndPhi(I, phi, i)
figure(1);
% subplot(2,2,3); title('Evolution'); view(3);
% cla;
% isosurface(phi<0);title([num2str(i) ' Iterations']); camlight; lighting gouraud; drawnow;
figure(4);
imagesc(I(:,:,10));
hold on
contour(phi(:,:,10),[0 0],'r');

%-- converts a mask to a SDF
function phi = mask2phi(init_a)
phi=bwdist(init_a)-bwdist(1-init_a);
