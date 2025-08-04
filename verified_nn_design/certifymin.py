import numpy as np
import pyopencl as cl
import multiprocessing.dummy as multiprocessing
from multiprocessing.dummy import Pool, Queue, Manager
import multiprocessing as multip
import time

oclStruct = None


def initContexts(_ocStruct,):
	global oclStruct
	oclStruct = _ocStruct


class CertifyMinimum:
	def __init__(self,lr=2,rBar=3.5,sigma=0.48,betaMax=np.arctan(0.5*np.tan(np.pi/4)),cfun = "", dbnd = 10000, chunkSize=2000, initGrid = 0.000001, refineFactor = 45, margin=0.9):
		self.lr = np.float64(lr)
		self.rBar = np.float64(rBar)
		self.sigma = np.float64(sigma)
		self.betaMax = np.float64(betaMax)
		self.d3bnd = np.float64(dbnd)
		self.chunk = chunkSize
		self.cfun = cfun
		self.margin = np.float64(margin)
		self.initGrid = np.float64(initGrid)
		self.refine = np.float64(refineFactor)
		self.guaranteedMin = np.inf
		self.minCertified = False
		self.platform = cl.get_platforms()[0]
		self.devices = self.platform.get_devices()
		self.cpus = [ self.devices[k] for k in np.nonzero([1 if d.type == 2 else 0 for d in self.devices])[0].tolist()]
		self.gpus = [ self.devices[k] for k in np.nonzero([1 if d.type == 4 and (d.vendor == 'AMD' or d.vendor == 'NVIDIA Corporation') else 0  for d in self.devices])[0].tolist()]
		# For debugging purporses, to get multiple "gpus" on my laptop:
		# self.gpus = self.cpus + self.gpus
		# breakpoint()
	
	def exportDict(self):
		return { \
				'lr': self.lr, \
				'rBar': self.rBar, \
				'sigma': self.sigma, \
				'betaMax': self.betaMax, \
				'dbnd': self.d3bnd, \
				'chunk': self.chunk, \
				'cfun': self.cfun, \
				'margin': self.margin, \
				'initGrid': self.initGrid, \
				'refine': self.refine, \
				'guaranteedMin' : self.guaranteedMin, \
				'minCertified': self.minCertified \
			}

	def setInitGrid(self,initGrid = 0.000001):
		self.initGrid = np.float64(initGrid)

	def setRefinementFactor(self, refineFactor = 75):
		self.refine = np.float64(refineFactor)

	def verifyByTwoLevelAdaptive(self,xiMin,xiMax):
		# i = 17
		# d3bnd = dd.deriv3DenomBoundsLambdas[i](sigma, lr, rBar)
		# denom = dd.deriv3DenomLambdas[i]
		# initGrid = 0.000001
		betaDim = int((2*self.betaMax/self.initGrid)+1)
		if np.floor(betaDim) != betaDim:
			betaDim = int(np.floor(betaDim)) + 1
			betaIncludeFinal = 1
		else:
			betaDim = int(np.floor(betaDim))
			betaIncludeFinal = 0
		xiDim = int(((xiMax-xiMin)/self.initGrid)+1)
		if np.floor(xiDim) != xiDim:
			xiDim = int(np.floor(xiDim)) + 1
			xiIncludeFinal = 1
		else:
			xiDim = int(np.floor(xiDim))
			xiIncludeFinal = 0
		# fGrid = np.ones((betaDim,4),dtype=np.float64)

		# quadCount = 0
		# xiTempOld = xiMin
		# overallMin = np.inf
		k = 1
		manager = Manager()
		openCLStruct = {
			"device_Contexts"        : [cl.Context([dev]) for dev in self.gpus], \
			"coarse_Device_Buffers"  : [], \
			"fine_Device_Buffers"    : [], \
			"device_Queues"          : [], \
			"overallMin"             : manager.Queue(), \
			"thread_Lock"            : multiprocessing.Lock(), \
			"free_gpus"              : [None for k in range(len(self.gpus))]
		}
		# Create one OpenCL command queue for each GPU:
		openCLStruct["device_Queues"] = [ \
											cl.CommandQueue( \
												openCLStruct["device_Contexts"][j], \
												device=self.gpus[j] \
											) \
											for j in range(len(self.gpus)) \
										]
		# Create one "coarse"-level device buffer on each GPU:
		# (This will store the output of the getWorkQuadrants function in processXiSlice)
		openCLStruct["coarse_Device_Buffers"] =  [ \
													cl.Buffer( \
														openCLStruct["device_Contexts"][j], \
														cl.mem_flags.READ_WRITE , \
														betaDim*4*8 \
													) \
													for j in range(len(self.gpus)) \
												]
		# Create one "fine"-level device buffer on each GPU:
		# (This will store the output of the processSingleQuadrantSequence function in processXiSlice)
		openCLStruct["fine_Device_Buffers"] =  [ \
													cl.Buffer( \
														openCLStruct["device_Contexts"][j], \
														cl.mem_flags.READ_WRITE , \
														betaDim*8 \
													) \
													for j in range(len(self.gpus)) \
												]
		openCLStruct["getWorkQuadrants_programs"] =  [ \
								cl.Program(openCLStruct["device_Contexts"][j], """
									#define chunk """ + str(int(self.chunk)) + """
									#define margin """ + str(self.margin) + """
									""" + self.cfun + """
									__kernel void make_table(
										double xiLow,
										double xiHigh,
										double betaMax,
										double initGrid,
										int betaDim,
										double lr,
										double rBar,
										double sigma,
										double d3bnd,
										__global double *result
									)
									{
										double fValOldC1,fValOldC2,fValNewC1,fValNewC2;
										double betaTemp,betaTempOld;
										int offset;
										int gid = get_global_id(0);
										if(gid > betaDim/chunk) return;
										betaTempOld = (gid == 0) ? -betaMax : ((gid*chunk==betaDim-1) ? betaMax : -betaMax + (gid*chunk - 1)*initGrid);
										fValOldC1 = fabs(denom(xiLow,betaTempOld,sigma,lr,rBar));
										fValOldC2 = fabs(denom(xiHigh,betaTempOld,sigma,lr,rBar));
										result[0+2] = 0;
										result[0+3] = fValOldC1;
										for(int k = (gid==0)?1:0; k < min(chunk,(betaDim)-gid*chunk); k++) {
											if(k + gid*chunk == betaDim - 1) {
												betaTemp = betaMax;
											} else {
												betaTemp = (k + gid*chunk)*initGrid - betaMax;
											}
											fValNewC1 = fabs(denom(xiLow, betaTemp,sigma,lr,rBar));
											fValNewC2 = fabs(denom(xiHigh,betaTemp,sigma,lr,rBar));
											if(
												margin*fValOldC1 <= d3bnd*sqrt(2.)*(initGrid) ||
												margin*fValOldC2 <= d3bnd*sqrt(2.)*(initGrid) ||
												margin*fValNewC1 <= d3bnd*sqrt(2.)*(initGrid) ||
												margin*fValNewC2 <= d3bnd*sqrt(2.)*(initGrid)
											) {
												offset = 4*(k + gid*chunk);
												result[offset]   = betaTempOld;
												result[offset+1] = betaTemp;
												result[offset+2] = 1;
												result[offset+3] = min(min(fValNewC1,fValNewC2),min(fValOldC1,fValOldC2));
											} else {
												offset = 4*(k + gid*chunk);
												result[offset+2] = 0;
												result[offset+3] = fValOldC1;
											}
											fValOldC1 = fValNewC1;
											fValOldC2 = fValNewC2;
											betaTempOld = betaTemp;
										}
									}
									""").build() \
								for j in range(len(self.gpus)) \
						]
		openCLStruct["processSingleQuadrantSequence_programs"] =  [ \
								cl.Program(openCLStruct["device_Contexts"][j], """
									#define refine """ + str(int(self.refine)) + """
									#define margin """ + str(self.margin) + """
									""" + self.cfun + """
									__kernel void make_table2(
										double xiLow,
										double xiHigh,
										int idxLow,
										int idxHigh,
										double initGrid,
										double lr,
										double rBar,
										double sigma,
										double d3bnd,
										__global double *table,
										__global double *result
									)
									{
										double betaLow, betaHigh, overallMin, temp;
										double tempXi,tempBeta;

										int gid = get_global_id(0);
										if(gid+idxLow > idxHigh) return;

										betaLow = table[4*(gid+idxLow)];
										betaHigh = table[4*(gid+idxLow)+1];

										overallMin = fabs(denom(xiLow, betaLow, sigma,lr,rBar));
										for(int i = 0; i <= refine; i++) {
											for(int j = 0; j <= refine; j++) {
												tempXi = xiLow + (xiHigh-xiLow)*i/refine;
												tempBeta = betaLow + (betaHigh-betaLow)*j/refine;
												temp = fabs(denom(tempXi, tempBeta, sigma,lr,rBar));
												overallMin = min(overallMin,temp);
											}
										}

										if(margin*overallMin <= d3bnd*sqrt(2.)*(initGrid/refine)) {
											result[gid] = 0;
										} else {
											result[gid] = overallMin;
										}

									}
									""").build() \
								for j in range(len(self.gpus)) \
						]
		# Force OpenCL to allocate the device buffers "coarse_Device_Buffers"/"fine_Device_Buffers" immediately:
		# (Default behavior is not specified: the driver may allocate immediately or defer allocation)
		[ \
			cl.enqueue_migrate_mem_objects( \
				openCLStruct["device_Queues"][j], \
				[openCLStruct["coarse_Device_Buffers"][j]] \
			) \
			for j in range(len(self.gpus)) \
		]
		[ \
			cl.enqueue_migrate_mem_objects( \
				openCLStruct["device_Queues"][j], \
				[openCLStruct["fine_Device_Buffers"][j]] \
			) \
			for j in range(len(self.gpus)) \
		]
		# Enqueue barriers to force the above migrations to complete before proceeding:
		# (i.e. the cl.enqueue_barrier call blocks until the supplied queue is empty)
		[ \
			cl.enqueue_barrier( \
				openCLStruct["device_Queues"][j], \
			) \
			for j in range(len(self.gpus)) \
		]
		
		# breakpoint()
		totalMins = 0
		# Create a (thread) pool, with one thread for each gpu:
		pool = Pool(processes=len(self.gpus),initializer=initContexts,initargs=(openCLStruct,))

		while k < xiDim:
			# time_start = time.time()

			xiArgs = [ 	(k-1) *self.initGrid + xiMin, \
						(k)   *self.initGrid + xiMin  ]
			
			if k == xiDim-1 and xiIncludeFinal == 1:
				xiArgs[1] = xiMax


			# The main thread blocks here until a GPU is free; this implements a primitive polling syncronization.
			# Afterwards,freeGPU specifies a GPU that is available for computation.
			freeGPU = -1
			while freeGPU < 0:
				# No lock is needed here, since openCLStruct["free_gpus"] is set only by the main thread
				for gpuStatus in [(j,openCLStruct["free_gpus"][j]) for j in range(len(self.gpus))]:
					if type(gpuStatus[1])==multip.pool.AsyncResult:
						if gpuStatus[1].ready():
							if not gpuStatus[1].get():
								self.minCertified = False
								return
							else:
								freeGPU = gpuStatus[0]
								openCLStruct["free_gpus"][freeGPU] = None
								break
					else:
						freeGPU = gpuStatus[0]
						openCLStruct["free_gpus"][freeGPU] = None
						break
				while not openCLStruct["overallMin"].empty():
					self.guaranteedMin = np.min([self.guaranteedMin,openCLStruct["overallMin"].get()])
					totalMins = totalMins + 1
				time.sleep(0.0001)


			# breakpoint()
			
			# Schedule this xi slice on the free gpu; store the return value, an AsyncResult object,
			# back to openCLStruct["free_gpus"], so we can check on the status of the task
			openCLStruct["free_gpus"][freeGPU] = pool.apply_async( \
													processXiSlice, \
													( \
														freeGPU, \
														self.cfun, \
														self.chunk, \
														self.refine, \
														np.float64(xiArgs[0]), \
														np.float64(xiArgs[1]), \
														self.betaMax, \
														self.initGrid, \
														np.int32(betaDim), \
														self.lr, \
														self.rBar,\
														self.sigma, \
														self.d3bnd, \
														self.margin \
													) \
												)
			
			# breakpoint()


			# print(str(time.time()-time_start) + ' seconds elapsed')

			# xiTempOld = xiTemp
			k = k + 1
		# Once every task is scheduled, wait for the remaining processes to complete
		for gpuStatus in [(j,openCLStruct["free_gpus"][j]) for j in range(len(self.gpus))]:
			if type(gpuStatus[1])==multip.pool.AsyncResult:
				if not gpuStatus[1].get():
					self.minCertified = False
					return
				else:
					freeGPU = gpuStatus[0]
					openCLStruct["free_gpus"][freeGPU] = None
					break
		pool.close()
		pool.join()
		# No need to lock here, because all of the threads should be done processing:
		while not openCLStruct["overallMin"].empty():
			self.guaranteedMin = np.min([self.guaranteedMin,openCLStruct["overallMin"].get()])
			totalMins = totalMins + 1
		# print('Received ' + str(totalMins) + ' total min values...')
		# # self.guaranteedMin = oclStruct["overallMin"]
		# print('Guaranteed Min is : ' + str(self.guaranteedMin))
		self.minCertified = True

def processXiSlice(gpu, cfun, chunk, refine, xiLow, xiHigh, betaMax, initGrid, betaDim, lr, rBar, sigma, d3bnd, margin):

	# print('Thread instantiated to use GPU: ' + str(gpu) + '!')
	
	coarseTable = getWorkQuadrants(	gpu, \
									cfun, \
									chunk, \
									np.float64(xiLow), \
									np.float64(xiHigh), \
									betaMax, \
									initGrid, \
									betaDim, \
									lr, \
									rBar, \
									sigma, \
									d3bnd, \
									margin \
								)

	coarseTable = coarseTable.reshape((betaDim,4))
	overallMin = np.min(coarseTable[:,3])

	if np.count_nonzero(coarseTable[:,2]) > 0:
		# Find all of the contiguous stretches of quadrants that need to be reprocessed
		idx = 0
		locs = np.nonzero(np.convolve(coarseTable[:,2],[1,-1]))[0]
		while idx < len(locs):
			# print("Window: (" + str(locs[idx]) + "," + str(locs[idx+1]-1) + ")")
			results = processSingleQuadrantSequence(	gpu, \
														cfun, \
														refine, \
														np.float64(xiLow), \
														np.float64(xiHigh), \
														np.int32(locs[idx]), \
														np.int32(locs[idx+1]-1), \
														initGrid, \
														betaDim, \
														lr, \
														rBar, \
														sigma, \
														d3bnd, \
														margin \
													)
			# breakpoint()
			if np.count_nonzero(results) < len(results):
				return False
			# All the threads write to openCLStruct["overallMin"], so be sure to lock it
			# with oclStruct["thread_Lock"]:
			# 	# print(np.min([overallMin,np.min(results)]))
			# 	oclStruct["overallMin"] = np.min([oclStruct["overallMin"],overallMin,np.min(results)])
			overallMin = np.min([overallMin,np.min(results)])
			idx = idx + 2
	
	oclStruct["overallMin"].put(overallMin)

	return True

def getWorkQuadrants(gpu, cfun,chunk,xiLow,xiHigh,betaMax,initGrid,betaDim,lr,rBar,sigma,d3bnd, margin):

	program = oclStruct['getWorkQuadrants_programs'][gpu]

	start_time = time.time()

	results = np.ones(betaDim*4,np.float64)

	destination_buf = oclStruct["coarse_Device_Buffers"][gpu]
	# # breakpoint()
	queue = oclStruct["device_Queues"][gpu]
	kernel = program.make_table
	kernel.set_scalar_arg_dtypes([     np.float64, np.float64, np.float64, np.float64,  np.int32,    np.float64, np.float64,   np.float64,  np.float64,   None])
	kernel(queue, (2**int(np.ceil(np.log2(results.shape[0]))),), None, xiLow,      xiHigh,     betaMax,    initGrid,   betaDim,     lr,         rBar,         sigma,       d3bnd,        destination_buf)

	cl.enqueue_copy(queue, results, destination_buf)

	print("Elapsed time (getWorkQuadrants): " + str(time.time() - start_time))

	return results

def processSingleQuadrantSequence(gpu, cfun, refine, xiLow, xiHigh, idxLow, idxHigh, initGrid, betaDim, lr, rBar, sigma, d3bnd, margin):
	
	program = oclStruct['processSingleQuadrantSequence_programs'][gpu]

	start_time = time.time()

	results = np.ones(idxHigh-idxLow+1,np.float64)

	output_buf = oclStruct["fine_Device_Buffers"][gpu]
	# Note: no need to initialize table_buf: it should be set by a call to getWorkQuadrants
	table_buf = oclStruct["coarse_Device_Buffers"][gpu]

	# # breakpoint()
	queue = oclStruct["device_Queues"][gpu]
	
	kernel = program.make_table2
	kernel.set_scalar_arg_dtypes([                                                 np.float64, np.float64, np.int32, np.int32, np.float64,  np.float64, np.float64,   np.float64,  np.float64,   None,            None])
	kernel(queue, (2**int(np.ceil(np.log2(results.shape[0]))),), None, xiLow,      xiHigh,     idxLow,   idxHigh,  initGrid,    lr,         rBar,         sigma,       d3bnd,        table_buf, output_buf)

	cl.enqueue_copy(queue, results, output_buf)

	print("Elapsed time (processSingleQuadrantSequence): " + str(time.time() - start_time))

	return results

# if __name__ == '__main__':

# 	res  = getWorkQuadrants(1.128537566697665,1.1285385666976648,0.4636476090008061,0.000001,927297,2,3.5,0.48,65055280686.85207)
# 	# breakpoint()
