/*(C) 2007 Timothy B. Terriberry*/
/*Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions
  are met:

  - Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  - Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  - Neither the name of the Xiph.org Foundation nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.*/
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>

/*Computes the signed Euclidean distance transform of a binary image in two
   dimensions.
  Regions inside the shape have negative distance, and regions outside have
   positive distance (_d[i,j]<0 if and only if _bimg[i,j] is non-zero).
  The result is symmetric with respect to "inside" and "outside" the shape.
  That is, if bimg1[i,j]==!bimg2[i,j] for all i, j, then the resulting
   distance transforms d1 and d2 obey the relation d1[i,j]==-d2[i,j].
  We use the distance to the boundary of the shape, which is equivalent to
   the following procedure:
   a) Compute a new boundary image b, twice the size of the original, where
    b[i,j] is zero if i and j are both even, and non-zero elsewhere if and
    only if the set of values _bimg[y>>1,x>>1] for the 8-connected neighbors
    (x,y) of (i,j) contain both zero and non-zero values.
   b) Compute db[i,j], the normal, unsigned distance transform of the boundary
    image, b[i,j].
   c) Negate the result inside the shape and downsample by setting
    d[i,j]=_bimg[i,j]?-db[i<<1,j<<1]:db[i<<1,j<<1].
  _d:    Returns the signed squared distance in half-pixel squared units.
         That is, the Euclidean distance to the boundary of the binary shape is
          0.5*sqrt(fabs(_d[i,j])).
         If there is _no_ boundary in the image, i.e., _bimg[i,j] is zero
          everywhere resp. non-zero everywhere, then the returned distance is
          INT_MAX resp. INT_MIN everywhere.
  _bimg: The binary image to compute the distance transform over.
         A pixel is "inside" the shape if _bimg[i,j] is non-zero, and
          "outside" otherwise.
  _h:    The number of rows in _bimg.
  _w:    The number of columns in _bimg.*/

void sedt2d(int *_d,unsigned char *_bimg,int _h,int _w){
  int *dd;
  int *f;
  int *v;
  int *z;
  int  i;
  int  j;
  int  k;
  int  n;
  /*We use an adaptation of the O(n) algorithm from \cite{FH04}.
    Our adaptation computes the positive and negative transform values
     simultaneously in a single pass.
    We could use the orignal algorithm on a doubled image as described in our
     documentation, but choose instead to operate directly on the original
     image, which makes for more complex code, but vastly reduced memory usage.
  @TECHREPORT{
    author="Pedro F. Felzenszwalb and Daniel P. Huttenlocher",
    title="Distance Transforms of Sampled Functions",
    number="TR2004-1963",
    institution="Cornell Computing and Information Science",
    year=2004
  }*/
  if(_h<=0||_w<=0)return;
  /*We do not strictly need this temporary image, but using it allows us to
     read out of it untransposed in the second pass, which is much nicer to the
     cache.
    We still have to write a transposed image in each pass, but writing
     introduces fewer stalls than reading.*/
  dd=(int *)malloc(_h*_w*sizeof(*dd));
  n=_h>_w?_h:_w;
  v=(int *)malloc(n*sizeof(*v));
  z=(int *)malloc((n+1)*sizeof(*z));
  f=(int *)malloc(_h*sizeof(*f));
  /*First compute the signed distance transform along the X axis.*/
  for(i=0;i<_h;i++){
    k=-1;
    /*At this stage, every transition contributes a parabola to the final
       envelope, and the intersection point _must_ lie between the vertices, so
       there's no need to worry about deletion or bounds checking.*/
    for(j=1;j<_w;j++)if(!_bimg[i*_w+j-1]!=!_bimg[i*_w+j]){
      int q;
      int s;
      q=(j<<1)-1;
      s=k<0?0:(v[k]+q>>2)+1;
      v[++k]=q;
      z[k]=s;
    }
    /*Now, go back and compute the distances to each parabola.
      If there were _no_ parabolas, then fill the row with +/- infinity.*/
    /*This is equivalent to dd[j*_h+i]=_bimg[i*_w+j]?INT_MIN:INT_MAX;*/
    if(k<0)for(j=0;j<_w;j++)dd[j*_h+i]=INT_MAX+!!_bimg[i*_w+j];
    else{
      int zk;
      z[k+1]=_w;
      j=k=0;
      do{
        int d1;
        int d2;
        d1=(j<<1)-v[k];
        d2=d1*d1;
        d1=d1+1<<2;
        zk=z[++k];
        for(;;){
          /*This is equivalent to dd[j*_h+i]=_bimg[i*_w+j]?-d2:d2;*/
          dd[j*_h+i]=d2-(d2<<1&-!!_bimg[i*_w+j]);
          if(++j>=zk)break;
          d2+=d1;
          d1+=8;
        }
      }
      while(zk<_w);
    }
  }
  /*Now extend the signed distance transform along the Y axis.
    This part of the code heavily depends on good branch prediction to be
     fast.*/
  for(j=0;j<_w;j++){
    int psign;
    /*v2 is not used uninitialzed, despite what your compiler may think.*/
    int v2;
    int q2;
    k=-1;
    /*Choose an initial value of psign that ensures there's no transition.
      This is the reason for the _h<=0 test at the start of the function.*/
    psign=dd[j*_h+0]<0;
    for(i=0,q2=1;i<_h;i++){
      int sign;
      int d;
      d=dd[j*_h+i];
      sign=d<0;
      /*If the sign changes, we've found a boundary point, and place a parabola
         of height 0 there.*/
      if(sign!=psign){
        int q;
        int s;
        q=(i<<1)-1;
        if(k<0)s=0;
        else for(;;){
          s=q2-v2-f[k];
          /*This is subtle: if q==v[k], then the parabolas never intersect, but
             our test is correct anyway, because f[k] is always positive.*/
          if(s>0){
            s=s/(q-v[k]<<2)+1;
            if(s>z[k])break;
          }
          else s=0;
          if(--k<0)break;
          v2=v[k]*v[k];
        }
        v[++k]=q;
        f[k]=0;
        z[k]=s;
        v2=q2;
      }
      /*We test for INT_MIN and INT_MAX together by adding +1 or -1 depending
         on the sign of d and checking if it retains that sign.
        If we have a finite point, we place up to three parabolas around it at
         height abs(d).
        There's no need to distinguish between the envelope inside and outside
         the shape, as a parabola of height 0 will always lie between them.*/
      if(sign==d-sign+!sign<0){
        int fq;
        int q;
        int s;
        int t;
        fq=abs(d);
        q=(i<<1)-1;
        if(k<0){
          s=0;
          t=1;
        }
        else for(;;){
          t=(q+1-v[k])*(q+1-v[k])+f[k]-fq;
          /*If the intersection point occurs to the left of the vertex, we will
             add all three parabolas, so we compute the intersection with the
             left-most parabola here.*/
          if(t>0){
            /*This is again subtle: if q==v[k], then we will take this branch
               whenever f[k]>=fq.
              The parabolas then intersect everywhere (when f[k]==fq) or
               nowhere (when f[k]>fq).
              However, in either case s<=0, and so we skip the division by zero
               below and delete the previous parabola.
              This relies on the fact that we ensure z[k] is never negative.*/
            s=q2-v2+fq-f[k];
            s=s<=0?0:s/(q-v[k]<<2)+1;
          }
          /*Otherwise we only add the right-most, so we compute that
             intersection point.
            (q+1)'s intersection point must lie even farther to the right than
             q's, so there is no needs to boundary check against 0.*/
          else s=(q2+(i<<3)-v2+fq-f[k])/(q+2-v[k]<<2)+1;
          if(s>z[k]||--k<0)break;
          v2=v[k]*v[k];
        }
        if(t>0){
          /*We only add the left-most parabola if it affects at least one
             pixel to prevent overrunning our arrays (e.g., consider the case
             _h==1).*/
          if(s<i){
            v[++k]=q;
            f[k]=fq;
            z[k]=s;
          }
          /*The center parabola will always span the interval [i,i+1), since
             the left and right parabolas are better outside of it.*/
          v[++k]=q+1;
          f[k]=fq;
          z[k]=i;
          s=i+1;
        }
        /*We only add the right-most parabola if it affects at least one pixel,
           to prevent overrunning our arrays (e.g., consider the case _h==1).
          This also conveniently ensures that z[k] is never larger than _h.*/
        if(s<_h){
          v[++k]=q+2;
          f[k]=fq;
          z[k]=s;
          v2=q2+(i<<3);
        }
      }
      psign=sign;
      q2+=i<<3;
    }
    /*Now, go back and compute the distances to each parabola.*/
    if(k<0){
      /*If there were _no_ parabolas, then the shape is uniform, and we've
         already filled it with +/- infinity in the X pass, so there's no need
         to examine the rest of the columns.
        Just copy the whole thing to the output image.*/
      memcpy(_d,dd,_w*_h*sizeof(*_d));
      break;
    }
    else{
      int zk;
      z[k+1]=_h;
      i=k=0;
      do{
        int d2;
        int d1;
        d1=(i<<1)-v[k];
        d2=d1*d1+f[k];
        d1=d1+1<<2;
        zk=z[++k];
        for(;;){
          /*This is equivalent to _d[i*_w+j]=dd[j*_h+i]<0?-d2:d2;*/
          _d[i*_w+j]=d2-(d2<<1&-(dd[j*_h+i]<0));
          if(++i>=zk)break;
          d2+=d1;
          d1+=8;
        }
      }
      while(zk<_h);
    }
  }
  free(f);
  free(z);
  free(v);
  free(dd);
}