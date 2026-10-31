"""Experimental double-precision GPU visibility traversal over a CPU-built BVH."""
import time

import numpy as np

KERNEL = r'''
__device__ bool blocked(const double* pts, const int* faces, const int* order,
 const double* boxes, const int* nodes, const double* o, const double* d, double limit) {
 int stack[64], sp=1; stack[0]=0;
 while(sp) {
  int n=stack[--sp]; double lo=0, hi=limit;
  for(int k=0;k<3;k++) {
   double a=boxes[n*6+k], b=boxes[n*6+3+k];
   if(fabs(d[k])<1e-30) { if(o[k]<a || o[k]>b) {hi=-1;break;} }
   else { double x=(a-o[k])/d[k], y=(b-o[k])/d[k];
    lo=fmax(lo,fmin(x,y)); hi=fmin(hi,fmax(x,y)); }
  }
  if(hi<lo)continue;
  const int* node=nodes+4*n;
  if(node[1]>0) {
   for(int j=node[0]; j<node[0]+node[1];j++) {
    int f=order[j]; const int* ids=faces+f*3;
    const double* a=pts+ids[0]*3; const double* b=pts+ids[1]*3; const double* c=pts+ids[2]*3;
    double e[3],g[3],p[3],s[3],q[3];
    for(int k=0;k<3;k++){e[k]=b[k]-a[k];g[k]=c[k]-a[k];s[k]=o[k]-a[k];}
    p[0]=d[1]*g[2]-d[2]*g[1];p[1]=d[2]*g[0]-d[0]*g[2];p[2]=d[0]*g[1]-d[1]*g[0];
    double det=e[0]*p[0]+e[1]*p[1]+e[2]*p[2]; if(fabs(det)<1e-24)continue;
    double u=(s[0]*p[0]+s[1]*p[1]+s[2]*p[2])/det; if(u<0 || u>1)continue;
    q[0]=s[1]*e[2]-s[2]*e[1];q[1]=s[2]*e[0]-s[0]*e[2];q[2]=s[0]*e[1]-s[1]*e[0];
    double v=(d[0]*q[0]+d[1]*q[1]+d[2]*q[2])/det; if(v<0 || u+v>1)continue;
    double t=(g[0]*q[0]+g[1]*q[1]+g[2]*q[2])/det; if(t>=0 && t<=limit)return true;
   }
  } else { stack[sp++]=node[2];stack[sp++]=node[3]; }
 }
 return false;
}
extern "C" __global__ void visible(const double* pts,const int* faces,const int* order,
 const double* boxes,const int* nodes,const double* centers,const double* normals,
 const double* dirs,double epsilon,double length,int begin,int end,unsigned char* out) {
 int i=begin+blockDim.x*blockIdx.x+threadIdx.x;if(i>=end)return;
 double o[3],d[3];for(int k=0;k<3;k++){d[k]=normals[i*3+k];o[k]=centers[i*3+k]+epsilon*d[k];}
 // The normal ray ends at center + length * normal in the CPU reference.
 if(!blocked(pts,faces,order,boxes,nodes,o,d,length-epsilon)){out[i]=1;return;}
 out[i]=0;
}
'''


def visibility_gpu(points, faces, centers, normals, directions, epsilon, length,
                   preference="system", cancel=None, progress=lambda *args: None):
    from depth_adaptive_qem import cp, select_cuda_device
    if cp is None:
        raise RuntimeError("CUDA runtime unavailable")
    started = time.perf_counter()
    device = select_cuda_device(preference)
    name = cp.cuda.runtime.getDeviceProperties(device)["name"]
    if isinstance(name, bytes):
        name = name.decode()
    triangles = points[faces]
    index_centers = triangles.mean(axis=1)
    mins, maxs = triangles.min(axis=1), triangles.max(axis=1)
    boxes, nodes, order = [], [], []
    completed = 0

    def build(ids):
        nonlocal completed
        if cancel is not None and cancel.is_set():
            raise RuntimeError("Surface extraction cancelled.")
        n = len(nodes)
        low, high = mins[ids].min(axis=0), maxs[ids].max(axis=0)
        boxes.append(np.r_[low, high])
        nodes.append([0, 0, 0, 0])
        if len(ids) <= 12:
            nodes[n] = [len(order), len(ids), 0, 0]
            order.extend(ids)
            completed += len(ids)
            if completed % 2048 < 12:
                progress("Building ray index", completed, len(faces))
        else:
            axis = int(np.argmax(np.ptp(index_centers[ids], axis=0)))
            middle = len(ids) // 2
            ids = ids[np.argpartition(index_centers[ids, axis], middle)]
            left, right = build(ids[:middle]), build(ids[middle:])
            nodes[n] = [0, 0, left, right]
        return n

    progress("Building ray index", 0, len(faces))
    build(np.arange(len(faces), dtype=np.int32))
    progress("Building ray index", len(faces), len(faces))
    kernel = cp.RawKernel(KERNEL, "visible", options=("--std=c++11",))
    progress("Preparing GPU kernel", 0, 1)
    kernel.compile()
    progress("Preparing GPU kernel", 1, 1)
    arrays = [cp.asarray(np.ascontiguousarray(a, dtype=d)) for a, d in (
        (points, np.float64), (faces, np.int32), (order, np.int32), (boxes, np.float64),
        (nodes, np.int32), (centers, np.float64), (normals, np.float64), (directions, np.float64))]
    out = cp.empty(len(centers), dtype=cp.uint8)
    for begin in range(0, len(centers), 8192):
        if cancel is not None and cancel.is_set():
            raise RuntimeError("Surface extraction cancelled.")
        end = min(begin + 8192, len(centers))
        kernel(((end - begin + 127) // 128,), (128,),
               (*arrays, np.float64(epsilon), np.float64(length), np.int32(begin), np.int32(end), out))
        cp.cuda.Stream.null.synchronize()
        progress(f"Surface probes: GPU {device} ({name})", end, len(centers))
    codes = cp.asnumpy(out).astype(np.int8)
    result = np.where(codes == 2, -1, codes).astype(np.int8)
    return result, {"device": f"GPU {device} ({name})", "gpu_pipeline_seconds": time.perf_counter() - started}
