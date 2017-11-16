
/******************************************************************************
* PROGRAM: se_kernel
* PURPOSE: This is a collection of functions which is used to optimize the 
* 	speed of seed extension step in BWA MEM procedure.
*
*
* NAME: Vuong Pham-Duy.
	College student.
*       Faculty of Computer Science and Technology.
*       Ho Chi Minh University of Technology, Viet Nam.
*       vuongpd95@gmail.com
*
* DATE: 5/10/2017
*
******************************************************************************/

#include <stdint.h>
#include "se_kernel.h"
#include "bwamem.h"
#include "bntseq.h"
#include "utils.h"

extern "C" static void *cuda_process(void *shared, int step, void *_data);

// Support functions
#define gpuErrchk(ans) { \
	gpuAssert((ans), __FILE__, __LINE__); \
}
void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", \
			cudaGetErrorString(code), file, line);
		if (abort) 
			exit(code);
	}
}

// Constant variables, tested, usable and can apply to all const needed variables.
__constant__ mem_opt_t d_opt;
__constant__ bwaidx_t d_idx;

__global__ 
void mem_process_seqs_kernel(ktp_data_t *data) {
	
}
static void *cuda_process(void *shared, int step, void *_data)
{
	ktp_aux_t *aux = (ktp_aux_t*)shared;
	ktp_data_t *data = (ktp_data_t*)_data;
	int i, j;
	if (step == 0) {
		const mem_opt_t *opt = aux->opt;
		ktp_data_t *ret;
		int64_t size = 0;
		ret = (ktp_data_t*)calloc(opt->cuda_num_thread, sizeof(ktp_data_t));
		for(j = 0; j < opt->cuda_num_thread; j++) {
			ret[j].seqs = bseq_read(aux->actual_chunk_size, &ret[j].n_seqs, aux->ks, aux->ks2);		
			if (ret[j].seqs == 0) {
				if (j == 0) {
					free(ret);
					return 0;
				} else {
					ret = (ktp_data_t*)realloc(ret, j * sizeof(ktp_data_t));
					break;
				}
			}
		}
		if (!aux->copy_comment) {
			for(j = 0; j < opt->cuda_num_thread; j++) {
				for (i = 0; i < ret[j].n_seqs; ++i) {
					free(ret[j].seqs[i].comment);
					ret[j].seqs[i].comment = 0;
				}
			}
		}
		for(j = 0; j < opt->cuda_num_thread; j++) {		
			for (i = 0; i < ret[j].n_seqs; ++i) 
				size += ret[j].seqs[i].l_seq;
		}
		if (bwa_verbose >= 3)
			fprintf(stderr, "[M::%s] read %d sequences (%ld bp)...\n", __func__, ret->n_seqs, (long)size);
		return ret;
	} else if (step == 1) {
		const mem_opt_t *opt = aux->opt;
		const bwaidx_t *idx = aux->idx;
		// push the above variables to CUDA symbols
		gpuErrchk(cudaMemcpyToSymbol(d_opt, opt, sizeof(mem_opt_t), 0, cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpyToSymbol(d_idx, idx, sizeof(bwaidx_t), 0, cudaMemcpyHostToDevice));
		// push data to CUDA shared memory
		ktp_data_t *d_data;
		gpuErrchk(cudaMalloc(&d_data, opt->cuda_num_thread * sizeof(ktp_data_t)));
		gpuErrchk(cudaMemcpy(d_data, data, opt->cuda_num_thread * sizeof(ktp_data_t), cudaMemcpyHostToDevice));		
		// push the original mem_process_seqs to mem_process_seqs_kernel
		mem_process_seqs_kernel<<<1, 1>>>(d_data);
		// check if the kernel is running fine
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			fprintf(stderr, "[M::%s] error while running mem_process_seqs_kernel: %s\n", \
				__func__, \
				cudaGetErrorString(err));
			exit(1);
		}
		// get the data back from the GPUs
		gpuErrchk(cudaMemcpy(data, d_data, opt->cuda_num_thread * sizeof(ktp_data_t),cudaMemcpyDeviceToHost));
		return data;
	} else if (step == 2) {
		for (j = 0; j < aux->opt->cuda_num_thread; j++) {
			for (i = 0; i < data[j].n_seqs; ++i) {
				if (data[j].seqs[i].sam) 
					err_fputs(data[j].seqs[i].sam, stdout);
				free(data[j].seqs[i].name); 
				free(data[j].seqs[i].comment);
				free(data[j].seqs[i].seq); 
				free(data[j].seqs[i].qual); 
				free(data[j].seqs[i].sam);
			}
			free(data[j].seqs); 
		}
		free(data);
		return 0;
	}
	return 0;
}

