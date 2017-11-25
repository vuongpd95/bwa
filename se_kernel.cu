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
#include "se_kernel.h"

extern "C" void cuda_mem_process_seqs(const mem_opt_t *opt, const bwt_t *bwt, \
	const bntseq_t *bns, const uint8_t *pac, int64_t n_processed, int n, \
	bseq1_t *seqs, const mem_pestat_t *pes0);

mem_chain_v chain_mem_core(const mem_opt_t *opt, const bwt_t *bwt, \
	const bntseq_t *bns, const uint8_t *pac, int l_seq, char *seq, \
	void *buf);

mem_alnreg_v sort_dedup_patch_core(const mem_opt_t *opt, const bntseq_t *bns, \
	const uint8_t *pac, char *seq, mem_alnreg_v *regs);

void chn_mem(void *data, int i, int tid);

void mem_sort_dedup_patch(void *data, int i, int tid);

// Support functions
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

// Constant variables, tested, usable and can apply to all const needed 
// variables.
__constant__ mem_opt_t d_opt;
__constant__ bwaidx_t d_bwt;
__constant__ bntseq_t d_bns;
__constant__ uint8_t d_pac;

void cuda_mem_process_seqs(const mem_opt_t *opt, const bwt_t *bwt, \
	const bntseq_t *bns, const uint8_t *pac, int64_t n_processed, int n, \
	bseq1_t *seqs, const mem_pestat_t *pes0)
{
	worker_t w;
	mem_pestat_t pes[4];
	double ctime, rtime;
	int i;
	ctime = cputime(); rtime = realtime();

	w.regs = (mem_alnreg_v*)malloc(n * sizeof(mem_alnreg_v));
	w.opt = opt;
	w.bwt = bwt;
	w.bns = bns;
	w.pac = pac;
	w.seqs = seqs;
	w.n_processed = n_processed;
	w.pes = &pes[0];

	if (opt->cuda_num_threads > 0) {
		w.aux = (smem_aux_t**)malloc(opt->n_threads * sizeof(smem_aux_t));
		for (i = 0; i < opt->n_threads; ++i)
			w.aux[i] = smem_aux_init();
		// Chaining mem
		kt_for(opt->n_threads, chn_mem, &w, \
			(opt->flag&MEM_F_PE)? n>>1 : n);
		
		// TODO: CUDA function goes here
		cudaMemcpyToSymbol(&d_opt, opt, sizeof(mem_opt_t));
		cudaMemcpyToSymbol(&d_bwt, bwt, sizeof(bwaidx_t));
		cudaMemcpyToSymbol(&d_bns, bns, sizeof(bntseq_t));
		cudaMemcpyToSymbol(&d_pac, pac, sizeof(uint8_t));
		
		if (opt->flag & MEM_F_PE) {
							
		} else {

		}
		// mem sort and delete duplicated patch 
		kt_for(opt->n_threads, mem_sort_dedup_patch, &w, \
			(opt->flag&MEM_F_PE)? n>>1 : n);
		for (i = 0; i < opt->n_threads; ++i)
			smem_aux_destroy(w.aux[i]);
		free(w.aux);
		goto w2;
	}
	
	w.aux = (smem_aux_t**)malloc(opt->n_threads * sizeof(smem_aux_t));
	for (i = 0; i < opt->n_threads; ++i)
		w.aux[i] = smem_aux_init();
	// find mapping positions
	kt_for(opt->n_threads, worker1, &w, (opt->flag&MEM_F_PE)? n>>1 : n); 
	for (i = 0; i < opt->n_threads; ++i)
		smem_aux_destroy(w.aux[i]);
	free(w.aux);

w2:
	if (opt->flag&MEM_F_PE) { // infer insert sizes if not provided
		// if pes0 != NULL, set the insert-size distribution as pes0		
		if (pes0) {
			memcpy(pes, pes0, 4 * sizeof(mem_pestat_t));
		} else {
			// otherwise, infer the insert size distribution from 
			// data
			mem_pestat(opt, bns->l_pac, n, w.regs, pes);
		}
	}
	// generate alignment
	kt_for(opt->n_threads, worker2, &w, (opt->flag&MEM_F_PE)? n>>1 : n);
	free(w.regs);
	if (bwa_verbose >= 3)
		fprintf(stderr, "[M::%s] Processed %d reads in %.3f CPU sec, \
			%.3f real sec\n", __func__, n, cputime() - ctime, \
			realtime() - rtime);
}

void chn_mem(void *data, int i, int tid)
{
	worker_t *w = (worker_t*)data;
	if (!(w->opt->flag&MEM_F_PE)) {
		if (bwa_verbose >= 4) 
			printf("=====> Processing read '%s' <=====\n", \
			w->seqs[i].name);
		w->chns[i] = chain_mem_core(w->opt, w->bwt, w->bns, w->pac, \
			w->seqs[i].l_seq, w->seqs[i].seq, w->aux[tid]);
	} else {
		if (bwa_verbose >= 4) 
			printf("=====> Processing read '%s'/1 <=====\n", \
				w->seqs[i<<1|0].name);
		w->chns[i<<1|0] = chain_mem_core(w->opt, w->bwt, w->bns, \
			w->pac, w->seqs[i<<1|0].l_seq, w->seqs[i<<1|0].seq,\
			w->aux[tid]);
		if (bwa_verbose >= 4) 
			printf("=====> Processing read '%s'/2 <=====\n", \
				w->seqs[i<<1|1].name);
		w->chns[i<<1|1] = chain_mem_core(w->opt, w->bwt, w->bns, \
			w->pac, w->seqs[i<<1|1].l_seq, w->seqs[i<<1|1].seq, \
			w->aux[tid]);
	}
}

mem_chain_v chain_mem_core(const mem_opt_t *opt, const bwt_t *bwt, \
	const bntseq_t *bns, const uint8_t *pac, int l_seq, char *seq, \
	void *buf) {

	int i;
	mem_chain_v chn;

	for (i = 0; i < l_seq; ++i) 
		// convert to 2-bit encoding if we have not done so
		seq[i] = seq[i] < 4? seq[i] : nst_nt4_table[(int)seq[i]];

	chn = mem_chain(opt, bwt, bns, l_seq, (uint8_t*)seq, buf);
	chn.n = mem_chain_flt(opt, chn.n, chn.a);
	mem_flt_chained_seeds(opt, bns, pac, l_seq, (uint8_t*)seq, chn.n, chn.a);
	if (bwa_verbose >= 4) mem_print_chain(bns, &chn);
	
	return chn;	
}

void mem_sort_dedup_patch(void *data, int i, int tid) {
	worker_t *w = (worker_t*)data;
	if (!(w->opt->flag&MEM_F_PE)) {
		w->regs[i] = sort_dedup_patch_core(w->opt, w->bns, w->pac, \
			w->seqs[i].seq, &w->regs[i]);
	} else { 
		w->regs[i<<1|0] = sort_dedup_patch_core(w->opt, w->bns, \
			w->pac, w->seqs[i<<1|0].seq, &w->regs[i<<1|0]);

		w->regs[i<<1|1] = sort_dedup_patch_core(w->opt, w->bns, \
			w->pac, w->seqs[i<<1|1].seq, &w->regs[i<<1|1]);
	}
}

mem_alnreg_v sort_dedup_patch_core(const mem_opt_t *opt, const bntseq_t *bns, \
	const uint8_t *pac, char *seq, mem_alnreg_v *regs) {
	
	int i;
	regs->n = mem_sort_dedup_patch(opt, bns, pac, (uint8_t*)seq, regs->n, \
		regs->a);
	if (bwa_verbose >= 4) {
		err_printf("* %ld chains remain after removing duplicated chains\n", \
			regs->n);
		for (i = 0; i < regs->n; ++i) {
			mem_alnreg_t *p = &regs->a[i];
			printf("** %d, [%d,%d) <=> [%ld,%ld)\n", \
				p->score, p->qb, p->qe, (long)p->rb, (long)p->re);
		}
	}
	for (i = 0; i < regs->n; ++i) {
		mem_alnreg_t *p = &regs->a[i];
		if (p->rid >= 0 && bns->anns[p->rid].is_alt)
			p->is_alt = 1;
	}
	return (*regs);	
}
