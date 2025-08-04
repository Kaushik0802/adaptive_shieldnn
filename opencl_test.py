import numpy as np
import pyopencl as cl
import time

if __name__ == '__main__':
	platform = cl.get_platforms()[0]
	device = platform.get_devices()[2]
	breakpoint()
	context = cl.Context([device])

	program = cl.Program(context, """
		#define denom(xi,beta,sigma,lr,rBar) (4*pow(3*lr*pow(sigma, 2)*sin(beta) + lr*pow(sigma, 2)*sin(beta - 2*xi) - 6*lr*(-1 + \
			sigma)*sigma*sin(beta - (3*xi)/2) + 4*lr*(2 - 4*sigma + 3*pow(sigma, 2))*sin(beta - xi) + 2*(rBar - 5*lr*(-1 + sigma))*sigma*sin(beta - \
			xi/2) - 2*rBar*sigma*sin(beta + xi/2), 3)*pow(2*lr*pow(1 - sigma + sigma*cos(xi/2), 2)*sin(beta - xi) - rBar*sigma*cos(beta)*sin(xi/2) \
			+ lr*sigma*cos(beta - xi)*(1 - sigma + sigma*cos(xi/2))*sin(xi/2), 3))
		__kernel void make_table(
			float sigma,
			float lr,
			float rBar,
			float epsXi,
			float lowerExtentXi,
			float epsBeta,
			float lowerExtentBeta,
			int xiMultStart,
			int xiMultEnd,
			int betaMultStart,
			int betaMultEnd,
			__global float *result
		)
	    {
			int gid = get_global_id(0);
			if(gid >= xiMultEnd - xiMultStart) return;
			float localTemp;
			float localMin = 1000000.;
			for(int k = betaMultStart; k <= betaMultEnd; k++) {
				localTemp = fabs(denom(lowerExtentXi + (gid + xiMultStart)*epsXi, lowerExtentBeta + (k)*epsBeta, sigma, lr, rBar));
				localMin = localTemp < localMin ? localTemp : localMin;
			}
			result[gid] = localMin;
	    }
		""").build()

	d3bnd = np.float32(65055280686.85207)
	epsXi = np.float32(1/(d3bnd/np.sqrt(2)))
	epsBeta = np.float32(1/(d3bnd/np.sqrt(2)))
	lowerExtentXi = np.float32(1.128537566697665)
	lowerExtentBeta = np.float32(-0.4636476090008061)
	xiMultStart = np.int32(0)
	xiMultEnd =   np.int32(4000)
	betaMultStart = np.int32(0)
	betaMultEnd = np.int32(400)
	lr = np.float32(2)
	rBar = np.float32(3.5)
	sigma = np.float32(0.48)

	start_time = time.time()

	results = np.zeros(xiMultEnd-xiMultStart,np.float32)
	mem_flags = cl.mem_flags
	destination_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, results.nbytes)

	queue = cl.CommandQueue(context)
	kernel = program.make_table
	kernel.set_scalar_arg_dtypes([     np.float32, np.float32, np.float32, np.float32, np.float32,    np.float32, np.float32,      np.int32,    np.int32,  np.int32,      np.int32,    None])
	kernel(queue, results.shape, None, sigma,      lr,         rBar,       epsXi,      lowerExtentXi, epsBeta,    lowerExtentBeta, xiMultStart, xiMultEnd, betaMultStart, betaMultEnd, destination_buf)

	cl.enqueue_copy(queue, results, destination_buf)

	print("Elapsed time: " + str(time.time() - start_time))

