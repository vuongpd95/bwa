/******************************************************************************
* PROGRAM: se_kernel
* PURPOSE: This is a collection of functions which is used to optimize the 
* 	speed of seed extension step in BWA MEM procedure.
*
*
* NAME: Vuong Pham-Duy.
	College student.
*       Faculty of Computer Science and Engineering.
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
		if (opt->flag & MEM_F_SMARTPE) {
			bseq1_t *sep[2 * opt->cuda_num_thread]; // data[x] use sep[2x, 2x+1]
			int n_sep[2 * opt->cuda_num_thread]; // data[x] use n_sep[2x, 2x+1]
			// TODO: Make the next line into CUDA kernel.
			// bseq_classify(data->n_seqs, data->seqs, n_sep, sep);
			for(j = 0; j < opt->cuda_num_thread; j++) {
				mem_opt_t tmp_opt = *opt;
				if(n_sep[2 * j]) {
					tmp_opt.flag &= ~MEM_F_PE;
					// TODO: Break it down
					mem_process_seqs(&tmp_opt, idx->bwt, idx->bns, idx->pac, aux->n_processed, \
						n_sep[2 * j], sep[2 * j + 1], 0);
					for (i = 0; i < n_sep[2 * j]; ++i)
						data->seqs[sep[2 * j][i].id].sam = sep[2 * j + 1][i].sam;
				}
				if(n_sep[2 * j + 1] {
					tmp_opt.flag |= MEM_F_PE;
					// TODO: Break it down
					mem_process_seqs(&tmp_opt, idx->bwt, idx->bns, idx->pac, aux->n_processed + n_sep[0], \
						n_sep[2 * j + 1], sep[2 * j + 1], aux->pes0);
					for (i = 0; i < n_sep[2 * j + 1]; ++i)
						data->seqs[sep[2 * j + 1][i].id].sam = sep[2 * j + 1][i].sam;
				}
				free(sep[2 * j]); free(2 * j + sep[1]);
			}
		} else {
			for(j = 0; j < opt->cuda_num_thread; j++) {
				// TODO: Break it down
				mem_process_seqs(opt, idx->bwt, idx->bns, idx->pac, aux->n_processed, \
					data[j]->n_seqs, data[j]->seqs, aux->pes0);
				aux->n_processed += data[j]->n_seqs;
			}
		}
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

