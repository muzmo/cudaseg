%region_seg(I,init_mask,max_its,E,T)
%coded by hormuz mostofi


%I = imread('liver.bmp');               %-- load the image
%m = zeros(size(I,1),size(I,2));          %-- create initial mask
%m= imread('mask.bmp');
%I = imresize(I,0.2);
%m = imresize(m,0.2);

%m=zeros(30,30,30);m(10:20,10:20,10:20)=1;
%I=zeros(30,30,30);I(13:16,13:16,13:16)=1;
cla;
res=3;

load mristack.mat;
[XI,YI,ZI] = meshgrid(0:res:size(mristack,1), 0:res:size(mristack,2), 0:1:size(mristack,3));
mristack=cast(mristack,'double');
I = interp3(mristack,XI,YI,ZI);I(isnan(I))=0;
m = zeros(size(I,1),size(I,2),size(I,3));
m(size(I,1)/3:2*size(I,1)/3,size(I,2)/3:2*size(I,2)/3,size(I,3)/3:2*size(I,3)/3)=1;

subplot(2,2,1); slice(I,15,5,5); title('Input Image');view(3);
subplot(2,2,2);

isosurface(m); title('Initial Mask'); view(3); %camlight; lighting gouraud;


seg = simpleseg(I, m, 400, 30, 100); %-- Run segmentationm, set last parameter = 205

subplot(2,2,1);cla; isosurface(seg); camlight; lighting gouraud; title('Final Mask of phi<=0');