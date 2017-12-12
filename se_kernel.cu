/******************************************************************************
* PROGRAM: se_kernel
* PURPOSE: This is a collection of functions which is intened to optimize the 
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
void cuda_seed_extension (const mem_opt_t *opt, const bntseq_t *bns, \
	const uint8_t *pac, int n, worker_t *w);

void print_seq(int n, int *l_seq);
void print_chns(int n, mem_chain_v *f_chns, mem_chain_t *f_a, int *i_a);
void print_bns_pac(int64_t l_pac, int32_t n_seqs);

CUDA_KSORT_INIT(64,  uint64_t, ks_lt_generic)

// Support functions
void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "[M::%s] GPUassert: %s %s %d\n", __func__, \
			cudaGetErrorString(code), file, line);
		if (abort) 
			exit(code);
	}
}

void print_mem_info()
{
	size_t free_byte;
	size_t total_byte;
	gpuErrchk(cudaMemGetInfo(&free_byte, &total_byte));

	double free_db = (double)free_byte;
	double total_db = (double)total_byte;
	double used_db = total_db - free_db;

	fprintf(stderr, "[M::%s] GPU memory usage: used = %.2f MB, free = %.2f MB, total = %.2f MB\n", \
		__func__, used_db/ONE_MBYTE, free_db/ONE_MBYTE, total_db/ONE_MBYTE);
}

void cuda_mem_process_seqs(const mem_opt_t *opt, const bwt_t *bwt, \
	const bntseq_t *bns, const uint8_t *pac, int64_t n_processed, int n, \
	bseq1_t *seqs, const mem_pestat_t *pes0)
{
	worker_t w;
	mem_pestat_t pes[4];
	double ctime, rtime;
	ctime = cputime(); rtime = realtime();
	int i;
	w.regs = (mem_alnreg_v*)malloc(n * sizeof(mem_alnreg_v));
	w.chns = (mem_chain_v*)malloc(n * sizeof(mem_chain_v));
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
			(opt->flag & MEM_F_PE)? n >> 1 : n);

		// Perform seed extension
		cuda_seed_extension(opt, bns, pac, n, &w);		

		// mem sort and delete duplicated patch 
		kt_for(opt->n_threads, mem_sort_dedup_patch, &w, \
			(opt->flag & MEM_F_PE)? n >> 1 : n);
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
		fprintf(stderr, "[M::%s] Processed %d reads in %.3f CPU sec, %.3f real sec\n", __func__, n, cputime() - ctime, \
			realtime() - rtime);
}

void chn_mem(void *data, int i, int tid)
{
	worker_t *w = (worker_t*)data;
	if (!(w->opt->flag&MEM_F_PE)) {
		w->chns[i] = chain_mem_core(w->opt, w->bwt, w->bns, w->pac, \
			w->seqs[i].l_seq, w->seqs[i].seq, w->aux[tid]);
	} else {
		w->chns[i<<1|0] = chain_mem_core(w->opt, w->bwt, w->bns, \
			w->pac, w->seqs[i<<1|0].l_seq, w->seqs[i<<1|0].seq,\
			w->aux[tid]);
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
	for (i = 0; i < regs->n; ++i) {
		mem_alnreg_t *p = &regs->a[i];
		if (p->rid >= 0 && bns->anns[p->rid].is_alt)
			p->is_alt = 1;
	}
	return (*regs);	
}

__device__ 
mem_chain_v* get_mem_chain_v(int n, mem_chain_v *f_chns, \
		mem_chain_t *f_a, int *i_a, mem_seed_t *seeds, \
		int *i_seeds, int i) {
	if(i < 0 || i >= n) 
		return get_mem_chain_v(n, f_chns, f_a, i_a, seeds, i_seeds, \
					(i >= n) ? (n - 1) : 0);
	mem_chain_v *ret;
	int j, first_a, first_seed, acc_seeds;
	// int k;
	first_a = i_a[i];
	first_seed = i_seeds[i];
	acc_seeds = 0;
	/*
	ret = (mem_chain_v*)malloc(sizeof(mem_chain_v));
	assert(ret != NULL);
	ret->n = f_chns[i].n;
	ret->m = f_chns[i].m;
	ret->a = (mem_chain_t*)malloc(ret->n * sizeof(mem_chain_t));
	assert(!(ret->n != 0 && ret->a == NULL));
	*/
	ret = &f_chns[i];
	if (ret->n > 0) ret->a = &f_a[first_a];
	else ret->a = NULL;
	for(j = 0; j < ret->n; j++) {
		ret->a[j].seeds = &seeds[first_seed + acc_seeds];
		acc_seeds += ret->a[j].n;
	}
	/*
	for(j = 0; j < ret->n; j++) {
		ret->a[j].n = f_a[first_a + j].n;
		ret->a[j].m = f_a[first_a + j].m;
		ret->a[j].first = f_a[first_a + j].first;
		ret->a[j].rid = f_a[first_a + j].rid;
		ret->a[j].w = f_a[first_a + j].w;
		ret->a[j].kept = f_a[first_a + j].kept;
		ret->a[j].is_alt = f_a[first_a + j].is_alt;
		ret->a[j].frac_rep = f_a[first_a + j].frac_rep;
		ret->a[j].pos = f_a[first_a + j].pos;
		ret->a[j].seeds = (mem_seed_t*)malloc(ret->a[j].n * sizeof(mem_seed_t));
		assert(!(ret->a[j].n != 0 && ret->a[j].seeds == NULL));
		if (ret->a[j].seeds != NULL)
			memcpy(ret->a[j].seeds, &seeds[first_seed + acc_seeds], \
					ret->a[j].n * sizeof(mem_seed_t));
		ret->a[j].seeds = &seeds[first_seed + acc_seeds];
		acc_seeds += ret->a[j].n;
	}
	*/
	return ret;
}

__device__
void free_mem_chain_v(mem_chain_v **p) {
	int i;
	for(i = 0; i < (*p)->n; i++) {
		free((*p)->a[i].seeds);	
	}
	free((*p)->a);
	free(*p);
}

__device__
void cuda_mem_chain2aln(const mem_opt_t *opt, const bntseq_t *bns, const uint8_t *pac, int l_query, const uint8_t *query, \
		const mem_chain_t *c, mem_alnreg_v *av);

__device__ 
mem_alnreg_v* cuda_mem_align1_core(mem_opt_t *opt, bntseq_t *bns, uint8_t *pac, \
	int l_seq, uint8_t *seq, mem_chain_v *f_chns, \
	mem_chain_t *f_a, int *i_a, mem_seed_t *seeds, int *i_seeds,
	int i, int n) {
	mem_alnreg_v *regs;
	mem_chain_v *chn;
	int j;
	regs = (mem_alnreg_v*)malloc(sizeof(mem_alnreg_v));
	assert(regs != NULL);
	chn = get_mem_chain_v(n, f_chns, f_a, i_a, seeds, i_seeds, i);
	cuda_kv_init(*regs);
	for (j = 0; j < chn->n; ++j) {
		mem_chain_t *p;
		p = &(chn->a[j]);
		cuda_mem_chain2aln(opt, bns, pac, l_seq, seq, p, regs);
	}
	// free_mem_chain_v(&chn);
	
	return regs;
}

__global__ 
void extension_kernel(int n, mem_opt_t *opt, int *l_seq, uint8_t *seq, int *i_seq, \
	mem_chain_v *f_chns, mem_chain_t *f_a, int *i_a, mem_seed_t *seeds, int *i_seeds, \
	int64_t l_pac, int32_t n_seqs, bntann1_t *anns, uint8_t *pac, \
	int *na, mem_alnreg_v **avs, mem_alnreg_v *fav) {
	// opt, bns, pac, l_seq, (uint8_t*)seq, chns.a[...]
	// return regs

	// opt, bns, pac are used all
	// use only 1 chain and return 1 regs for each of the execution
	// use only one l_seq and seq[...] for each of the execution
	// opt is constant so dont have to worry about it
	// pac can be used right away
	// Convert bns to desired form first
	bntseq_t bns;
	bns.l_pac = l_pac;
	bns.n_seqs = n_seqs;
	bns.anns = anns;
	int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
	/*if (thread_idx == 0 && cnt == 93) {
		for(int j = 0; j < n; j++) {
			printf("f_chns[%d]: n = %lu, m = %lu.\n", j, f_chns[j].n, f_chns[j].m);
			for(int k = 0; k < f_chns[j].n; k++) {
				printf("	a[%d]: n = %d, m = %d.\n", k, f_a[i_a[j] + k].n, f_a[i_a[j] + k].m);
			}
		}
	}*/
	int i = thread_idx;
	int cuda_num_threads = opt->cuda_num_threads;
	if(opt->flag & MEM_F_PE) {
		int alt_n;
		alt_n = n >> 1;
		for(;;) {
			if (i >= alt_n) break;
			avs[i<<1|0] = cuda_mem_align1_core(opt, &bns, pac, \
				l_seq[i<<1|0], &seq[i_seq[i<<1|0]], \
				f_chns, f_a, i_a, seeds, i_seeds, i<<1|0, n);
			atomicAdd(na, avs[i<<1|0]->n);
			fav[i<<1|0].n = avs[i<<1|0]->n;
			fav[i<<1|0].m = avs[i<<1|0]->m;

			avs[i<<1|1] = cuda_mem_align1_core(opt, &bns, pac, \
				l_seq[i<<1|1], &seq[i_seq[i<<1|1]], \
				f_chns, f_a, i_a, seeds, i_seeds, i<<1|1, n);
			atomicAdd(na, avs[i<<1|1]->n);
			fav[i<<1|1].n = avs[i<<1|1]->n;
			fav[i<<1|1].m = avs[i<<1|1]->m;

			i += cuda_num_threads;
		}
	} else {
		for(;;) {
			if (i >= n) break;
			avs[i] = cuda_mem_align1_core(opt, &bns, pac, \
				l_seq[i], &seq[i_seq[i]], f_chns, f_a, i_a, \
				seeds, i_seeds, i, n);
			atomicAdd(na, avs[i]->n);
			fav[i].n = avs[i]->n;
			fav[i].m = avs[i]->m;
			i += cuda_num_threads;
		}
	}
}

__global__ 
void ret_kernel(int n, mem_opt_t *opt, mem_alnreg_v **avs, int* av_ia, mem_alnreg_t *av_a) {
	int thread_idx, i, j, size, cuda_num_threads;
	cuda_num_threads = opt->cuda_num_threads;
	thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
	i = thread_idx;
	for(;;) {
		if(i >= n) break;
		j = av_ia[i];
		size = avs[i]->n;
		if (avs[i]-> a != NULL) memcpy(&av_a[j], avs[i]->a, size * sizeof(mem_alnreg_t));
		i += cuda_num_threads;
	}
}

void cuda_seed_extension(const mem_opt_t *opt, const bntseq_t *bns, \
	const uint8_t *pac, int n, worker_t *w) {
	
	// f_... means flattened_..., the new structure doesn't have pointers 
	// Use w->seqs[...].l_seq =====> l_seq[...]
	// Use w->seqs[...].seq =======> seq[...] + idx: di_seq[...]
	// Use w->chns[...] ===========> df_chns[...], df_a[...] + idx: di_a[...], d_seeds[...] + idx: di_seeds[...]
	// Use opt, pac ===============> d_opt, d_pac
	// Use bns ====================> l_pac, d_anns[...]
	// Use pac[l_pac/4+1]
	// cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024 * ONE_MBYTE);
	double ctime, rtime;
	ctime = cputime(); rtime = realtime();

	int i, j, k;
	int n_a, n_seeds;

	bntann1_t *d_anns;

	int *l_seq, *dl_seq, sl_seq, *i_seq, *di_seq;
	uint8_t *seq, *d_seq;

	mem_chain_v *chns; // used to replace w->chns
	bseq1_t *seqs; // used to replace w->seqs
	mem_opt_t *d_opt;
	mem_chain_v *f_chns, *df_chns; // flat version of chns (mem_chain_v)
	mem_chain_t *f_a, *df_a; // flat version of a (mem_chain_t)
	int *i_a, *di_a; // starting index of different chns's mem_chain_t
	mem_seed_t *seeds, *d_seeds; // contains all seeds (mem_seed_t)
	int *i_seeds, *di_seeds; // starting index of different chns's seeds. (mem_seed_t)
	uint8_t *d_pac; // Just device variable of the host's pac variable

	n_a = 0; 
	n_seeds = 0;
	sl_seq = 0;
	l_seq = (int*)malloc(n * sizeof(int));
	i_seq = (int*)malloc(n * sizeof(int));
	chns = w->chns;
	seqs = w->seqs;
	for(i = 0; i < n; i++) {
		l_seq[i] = seqs[i].l_seq;
		sl_seq += l_seq[i];
		n_a += chns[i].n;
		for(j = 0; j < chns[i].n; j++) {
			n_seeds += chns[i].a[j].n;
		}
	}
	seq = (uint8_t*)malloc(sl_seq * sizeof(uint8_t));
	int acc_seq;
	acc_seq = 0;
	for(i = 0; i < n; i++) {
		for(j = 0; j < l_seq[i]; j++) {
			seq[acc_seq + j] = w->seqs[i].seq[j];		
		}
		i_seq[i] = acc_seq;
		acc_seq += l_seq[i];
	}
	
	f_chns = (mem_chain_v*)malloc(n * sizeof(mem_chain_v));
	f_a = (mem_chain_t*)malloc(n_a * sizeof(mem_chain_t));
	seeds = (mem_seed_t*)malloc(n_seeds * sizeof(mem_seed_t));
	i_a = (int*)malloc(n * sizeof(int));
	i_seeds = (int*)malloc(n * sizeof(int));
                                                        
	int acc_a, acc_seeds;                             
	acc_a = 0; acc_seeds = 0;

	for(i = 0; i < n; i++) {
		i_seeds[i] = acc_seeds;
		f_chns[i].n = chns[i].n;
		f_chns[i].m = chns[i].m;
		for(j = 0; j < chns[i].n; j++) {
			// int n, m, first, rid;
			// uint32_t w:29, kept:2, is_alt:1;
			// float frac_rep;
			// int64_t pos;
			mem_chain_t *tmp;
			tmp = &chns[i].a[j];
			f_a[acc_a + j].n = tmp->n;
			f_a[acc_a + j].m = tmp->m;
			f_a[acc_a + j].first = tmp->first;
			f_a[acc_a + j].rid = tmp->rid;
			f_a[acc_a + j].w = tmp->w;
			f_a[acc_a + j].kept = tmp->kept;
			f_a[acc_a + j].is_alt = tmp->is_alt;
			f_a[acc_a + j].frac_rep = tmp->frac_rep;
			f_a[acc_a + j].pos = tmp->pos;
			for(k = 0; k < chns[i].a[j].n; k++) {
				// int64_t rbeg;
				// int32_t qbeg, len;
				// int score;
				mem_seed_t *tmp0;
				tmp0 = &chns[i].a[j].seeds[k];
				// seeds[acc_seeds + k].rbeg = tmp0->rbeg;
				// seeds[acc_seeds + k].qbeg = tmp0->qbeg;
				// seeds[acc_seeds + k].len = tmp0->len;
				// seeds[acc_seeds + k].score = tmp0->score;
				memcpy(&seeds[acc_seeds + k], tmp0, sizeof(mem_seed_t));
			}
			acc_seeds += chns[i].a[j].n;		
		}
		i_a[i] = acc_a;
		acc_a += chns[i].n;
	}

	if (bwa_verbose >= 3)
		fprintf(stderr, "[M::%s] Flattening structures takes %.3f CPU sec, %.3f real sec\n", __func__, cputime() - ctime, \
				realtime() - rtime);

	ctime = cputime(); rtime = realtime();
	gpuErrchk(cudaDeviceSetLimit(cudaLimitMallocHeapSize, FIXED_HEAP * ONE_MBYTE));
	if (bwa_verbose >= 3)
		fprintf(stderr, "[M::%s] First CUDA call lasts %.3f CPU sec, %.3f real sec\n", __func__, cputime() - ctime, \
			realtime() - rtime);

	gpuErrchk(cudaMalloc(&d_opt, sizeof(mem_opt_t)));

	ctime = cputime(); rtime = realtime();
	gpuErrchk(cudaMalloc(&d_anns, bns->n_seqs * sizeof(bntann1_t)));
	gpuErrchk(cudaMalloc(&dl_seq, n * sizeof(int)));
	gpuErrchk(cudaMalloc(&d_seq, sl_seq * sizeof(uint8_t)));

	gpuErrchk(cudaMalloc(&df_chns, n * sizeof(mem_chain_v)));
	gpuErrchk(cudaMalloc(&df_a, n_a * sizeof(mem_chain_t)));
	gpuErrchk(cudaMalloc(&d_seeds, n_seeds * sizeof(mem_seed_t)));

	gpuErrchk(cudaMalloc(&di_seq, n * sizeof(int)));
	gpuErrchk(cudaMalloc(&di_a, n * sizeof(int)));
	gpuErrchk(cudaMalloc(&di_seeds, n * sizeof(int)));

	gpuErrchk(cudaMalloc(&d_pac, (bns->l_pac/4+1) * sizeof(uint8_t)));

	gpuErrchk(cudaMemcpy(d_opt, opt, sizeof(mem_opt_t), \
			cudaMemcpyHostToDevice));

	gpuErrchk(cudaMemcpy(d_anns, bns->anns, bns->n_seqs * sizeof(bntann1_t), \
			cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dl_seq, l_seq, n * sizeof(int), \
			cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_seq, seq, sl_seq * sizeof(uint8_t), \
			cudaMemcpyHostToDevice));

	gpuErrchk(cudaMemcpy(df_chns, f_chns, n * sizeof(mem_chain_v), \
				cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(df_a, f_a, n_a * sizeof(mem_chain_t), \
				cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_seeds, seeds, n_seeds * sizeof(mem_seed_t), \
				cudaMemcpyHostToDevice));

	gpuErrchk(cudaMemcpy(di_seq, i_seq, n * sizeof(int), \
				cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(di_a, i_a, n * sizeof(int), \
				cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(di_seeds, i_seeds, n * sizeof(int), \
				cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_pac, pac, (bns->l_pac/4+1) * sizeof(uint8_t), \
				cudaMemcpyHostToDevice));
	
	mem_alnreg_v **d_avs; // Used to maintain connections between kernels
	int *h_av_na, *d_av_na; // Varibles which hold total number of mem_alnreg_t
	int *h_av_ia, *d_av_ia; // Arrays which hold starting index of mem_alnreg_t
			  // of different mem_alnreg_v
	mem_alnreg_v *h_fav, *d_fav; // Arrays which hold flat mem_alnreg_v
	mem_alnreg_t *h_av_a, *d_av_a; // Arrays which hold all mem_alnreg_t
	/*
	int r_n;
	if (opt->flag & MEM_F_PE) r_n = n >> 1;
	else r_n = n;
	*/
	h_fav = (mem_alnreg_v*)malloc(n * sizeof(mem_alnreg_v));
	gpuErrchk(cudaMalloc(&d_fav, n * sizeof(mem_alnreg_v)));

	h_av_na = (int*)malloc(sizeof(int));
	gpuErrchk(cudaMalloc(&d_av_na, sizeof(int)));
	gpuErrchk(cudaMemset(d_av_na, 0, sizeof(int)));

	gpuErrchk(cudaMalloc((void**)&d_avs, n * sizeof(mem_alnreg_v*)));

	// print_seq(n, l_seq);
	// print_chns(n, f_chns, f_a, i_a);
	// print_bns_pac(bns->l_pac, bns->n_seqs);
	int num_block;
	int t_block;
	if (opt->cuda_num_threads > THREAD_LIMIT_PER_BLOCK) {
		t_block = THREAD_LIMIT_PER_BLOCK;
		num_block = ceil(opt->cuda_num_threads / t_block);
	} else {
		t_block = ceil(opt->cuda_num_threads / WARP_SIZE) * WARP_SIZE;
		num_block = 1;
	}
	dim3 thread_per_block(t_block);

	extension_kernel<<<num_block, thread_per_block>>>(n, d_opt, dl_seq, d_seq, di_seq, \
		df_chns, df_a, di_a, d_seeds, di_seeds, bns->l_pac, bns->n_seqs, d_anns, \
		d_pac, d_av_na, d_avs, d_fav);

	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	gpuErrchk(cudaMemcpy(h_av_na, d_av_na, sizeof(int), \
			cudaMemcpyDeviceToHost));

	gpuErrchk(cudaMemcpy(h_fav, d_fav, n * sizeof(mem_alnreg_v), \
			cudaMemcpyDeviceToHost));
	
	h_av_a = (mem_alnreg_t*)malloc(*h_av_na * sizeof(mem_alnreg_t));
	h_av_ia = (int*)malloc(n * sizeof(int));
	
	int acc_av_a;
	acc_av_a = 0;
	for(i = 0; i < n; i++) {
		h_av_ia[i] = acc_av_a;
		acc_av_a += h_fav[i].n;	
	}

	gpuErrchk(cudaMalloc(&d_av_ia, n * sizeof(int)));
	gpuErrchk(cudaMalloc(&d_av_a, *h_av_na * sizeof(mem_alnreg_t)));

	gpuErrchk(cudaMemcpy(d_av_ia, h_av_ia, n * sizeof(int), \
			cudaMemcpyHostToDevice));

	ret_kernel<<<num_block, thread_per_block>>>(n, d_opt, d_avs, d_av_ia, d_av_a);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	gpuErrchk(cudaMemcpy(h_av_a, d_av_a, *h_av_na * sizeof(mem_alnreg_t), \
		cudaMemcpyDeviceToHost));

	// Convert from flat to original form
	for(i = 0; i < n; i++) {
		w->regs[i].n = h_fav[i].n;
		w->regs[i].m = h_fav[i].m;
		w->regs[i].a = (mem_alnreg_t*)malloc(h_fav[i].n * sizeof(mem_alnreg_t));
		memcpy(w->regs[i].a, &h_av_a[h_av_ia[i]], h_fav[i].n * sizeof(mem_alnreg_t));
	}
	/*
	cudaFree(d_opt);
	cudaFree(d_pac);
	cudaFree(d_anns);
	cudaFree(dl_seq);
	cudaFree(d_seq);
	cudaFree(df_chns);
	cudaFree(df_a);
	cudaFree(d_seeds);
	cudaFree(di_seq);
	cudaFree(di_a);
	cudaFree(di_seeds);
	free(l_seq);
	free(seq);
	free(f_chns);
	free(f_a);
	free(seeds);
	free(i_seq);
	free(i_a);
	free(i_seeds);

	cudaFree(d_avs);
	cudaFree(d_av_na);
	cudaFree(d_av_ia); 
	cudaFree(d_fav);
	cudaFree(d_av_a);

	free(h_av_ia);
	free(h_av_na);
	free(h_fav);
	free(h_av_a);
 	/* */
	gpuErrchk(cudaDeviceReset());
	fprintf(stderr, "[M::%s] CUDA kernels take %.3f CPU sec, %.3f real sec\n", __func__, cputime() - ctime, \
				realtime() - rtime);
}

/* Functions that mainly the same with original C source */
__device__ 
void *cuda_realloc(void *ptr, size_t old_size, size_t new_size)
{
	void *cuda_new;

	if (!ptr) {
		cuda_new = malloc(new_size);
		if (!cuda_new) return NULL;
	} else {
		if (old_size < new_size) {
			cuda_new = malloc(new_size);
			if (!cuda_new) return NULL;
			memcpy(cuda_new, ptr, old_size);
			free(ptr);
		} else {
			cuda_new = ptr;
		}
	}
	return cuda_new;
}

__device__
void *cuda_calloc(size_t count, size_t size)
{
	size_t alloc_size = count * size;
	void *cuda_new = malloc(alloc_size);

	if (cuda_new) {
		memset(cuda_new, 0, alloc_size);
		return cuda_new;
	}
	return NULL;
}

#define cuda_kv_pushp(type, v) ((((v).n == (v).m)?									\
			   ((v).m = ((v).m? (v).m<<1 : 2),								\
				((v).a = (type*)cuda_realloc((v).a, (v).n * sizeof(type), sizeof(type) * (v).m), 0))	\
			   : 0), &(v).a[(v).n++])
__device__
mem_alnreg_t* cuda_kv_pushp_v2 (mem_alnreg_v av) {
	mem_alnreg_t *ret;
	mem_alnreg_t *tmp;
	if (av.n == av.m) {
		av.m = (av.m)?(av.m << 1):2;
		tmp = (mem_alnreg_t*)cuda_realloc(av.a, av.n * sizeof(mem_alnreg_t), sizeof(mem_alnreg_t) * av.m);
		assert(tmp != NULL);
		av.a = tmp;
	} else {
		ret = 0;
	}
	ret = &(av.a[av.n++]);
	return ret;
}
__device__
int cal_max_gap(const mem_opt_t *opt, int qlen)
{
	int l_del = (int)((double)(qlen * opt->a - opt->o_del) / opt->e_del + 1.);
	int l_ins = (int)((double)(qlen * opt->a - opt->o_ins) / opt->e_ins + 1.);
	int l = l_del > l_ins? l_del : l_ins;
	l = l > 1? l : 1;
	return l < opt->w<<1? l : opt->w<<1;
}

__device__
int cuda_bns_pos2rid(const bntseq_t *bns, int64_t pos_f)
{
	int left, mid, right;
	if (pos_f >= bns->l_pac) return -1;
	left = 0; mid = 0; right = bns->n_seqs;
	while (left < right) { // binary search
		mid = (left + right) >> 1;
		if (pos_f >= bns->anns[mid].offset) {
			if (mid == bns->n_seqs - 1) break;
			if (pos_f < bns->anns[mid+1].offset) break; // bracketed
			left = mid + 1;
		} else right = mid;
	}
	return mid;
}

__device__
uint8_t *cuda_bns_get_seq(int64_t l_pac, const uint8_t *pac, int64_t beg, int64_t end, int64_t *len)
{
	uint8_t *seq = 0;
	if (end < beg) end ^= beg, beg ^= end, end ^= beg; // if end is smaller, swap
	if (end > l_pac<<1) end = l_pac<<1;
	if (beg < 0) beg = 0;
	if (beg >= l_pac || end <= l_pac) {
		int64_t k, l = 0;
		*len = end - beg;
		seq = (uint8_t*)malloc(end - beg);;
		assert(!((end - beg) != 0 && seq == NULL));
		if (beg >= l_pac) { // reverse strand
			int64_t beg_f = (l_pac<<1) - 1 - end;
			int64_t end_f = (l_pac<<1) - 1 - beg;
			for (k = end_f; k > beg_f; --k)
				seq[l++] = 3 - _cuda_get_pac(pac, k);
		} else { // forward strand
			for (k = beg; k < end; ++k)
				seq[l++] = _cuda_get_pac(pac, k);
		}
	} else *len = 0; // if bridging the forward-reverse boundary, return nothing
	return seq;
}

__device__
int64_t cuda_bns_depos(const bntseq_t *bns, int64_t pos, int *is_rev)
{
	return (*is_rev = (pos >= bns->l_pac))? (bns->l_pac<<1) - 1 - pos : pos;
}
__device__
uint8_t *cuda_bns_fetch_seq(const bntseq_t *bns, const uint8_t *pac, int64_t *beg, int64_t mid, int64_t *end, int *rid)
{
	int64_t far_beg, far_end, len;
	int is_rev;
	uint8_t *seq;

	if (*end < *beg) *end ^= *beg, *beg ^= *end, *end ^= *beg; // if end is smaller, swap
	assert(*beg <= mid && mid < *end);
	*rid = cuda_bns_pos2rid(bns, cuda_bns_depos(bns, mid, &is_rev));
	far_beg = bns->anns[*rid].offset;
	far_end = far_beg + bns->anns[*rid].len;
	if (is_rev) { // flip to the reverse strand
		int64_t tmp = far_beg;
		far_beg = (bns->l_pac<<1) - far_end;
		far_end = (bns->l_pac<<1) - tmp;
	}
	*beg = *beg > far_beg? *beg : far_beg;
	*end = *end < far_end? *end : far_end;
	seq = cuda_bns_get_seq(bns->l_pac, pac, *beg, *end, &len);
	if (seq == 0 || *end - *beg != len) {
			printf("[E::%s] begin=%ld, mid=%ld, end=%ld, len=%ld, seq=%p, rid=%d, far_beg=%ld, far_end=%ld\n",
					__func__, (long)*beg, (long)mid, (long)*end, (long)len, seq, *rid, (long)far_beg, (long)far_end);
	}
	assert(seq && *end - *beg == len); // assertion failure should never happen
	return seq;
}
__device__
int simple_cuda_abs(int c) {
	if(c < 0) return (-c);
	else return c;
}
__device__
int cuda_ksw_extend2(int qlen, const uint8_t *query, int tlen, const uint8_t *target, int m, const int8_t *mat, int o_del, int e_del, int o_ins, int e_ins, int w, int end_bonus, int zdrop, int h0, int *_qle, int *_tle, int *_gtle, int *_gscore, int *_max_off)
{
	eh_t *eh; // score array
	int8_t *qp; // query profile
	int i, j, k, oe_del = o_del + e_del, oe_ins = o_ins + e_ins, beg, end, max, max_i, max_j, max_ins, max_del, max_ie, gscore, max_off;
	assert(h0 > 0);
	// allocate memory
	qp = (int8_t*)malloc(qlen * m);
	assert(!((qlen * m) != 0 && qp == NULL));
	eh = (eh_t*)cuda_calloc(qlen + 1, 8);
	assert(!((qlen + 1) != 0 && eh == NULL));
	// generate the query profile
	for (k = i = 0; k < m; ++k) {
		const int8_t *p = &mat[k * m];
		for (j = 0; j < qlen; ++j) qp[i++] = p[query[j]];
	}
	// fill the first row
	eh[0].h = h0; eh[1].h = h0 > oe_ins? h0 - oe_ins : 0;
	for (j = 2; j <= qlen && eh[j-1].h > e_ins; ++j)
		eh[j].h = eh[j-1].h - e_ins;
	// adjust $w if it is too large
	k = m * m;
	for (i = 0, max = 0; i < k; ++i) // get the max score
		max = max > mat[i]? max : mat[i];
	max_ins = (int)((double)(qlen * max + end_bonus - o_ins) / e_ins + 1.);
	max_ins = max_ins > 1? max_ins : 1;
	w = w < max_ins? w : max_ins;
	max_del = (int)((double)(qlen * max + end_bonus - o_del) / e_del + 1.);
	max_del = max_del > 1? max_del : 1;
	w = w < max_del? w : max_del; // TODO: is this necessary?
	// DP loop
	max = h0, max_i = max_j = -1; max_ie = -1, gscore = -1;
	max_off = 0;
	beg = 0, end = qlen;
	for (i = 0; i < tlen; ++i) {
		int t, f = 0, h1, m = 0, mj = -1;
		int8_t *q = &qp[target[i] * qlen];
		// apply the band and the constraint (if provided)
		if (beg < i - w) beg = i - w;
		if (end > i + w + 1) end = i + w + 1;
		if (end > qlen) end = qlen;
		// compute the first column
		if (beg == 0) {
			h1 = h0 - (o_del + e_del * (i + 1));
			if (h1 < 0) h1 = 0;
		} else h1 = 0;
		for (j = beg; j < end; ++j) {
			// At the beginning of the loop: eh[j] = { H(i-1,j-1), E(i,j) }, f = F(i,j) and h1 = H(i,j-1)
			// Similar to SSE2-SW, cells are computed in the following order:
			//   H(i,j)   = max{H(i-1,j-1)+S(i,j), E(i,j), F(i,j)}
			//   E(i+1,j) = max{H(i,j)-gapo, E(i,j)} - gape
			//   F(i,j+1) = max{H(i,j)-gapo, F(i,j)} - gape
			eh_t *p = &eh[j];
			int h, M = p->h, e = p->e; // get H(i-1,j-1) and E(i-1,j)
			p->h = h1;          // set H(i,j-1) for the next row
			M = M? M + q[j] : 0;// separating H and M to disallow a cigar like "100M3I3D20M"
			h = M > e? M : e;   // e and f are guaranteed to be non-negative, so h>=0 even if M<0
			h = h > f? h : f;
			h1 = h;             // save H(i,j) to h1 for the next column
			mj = m > h? mj : j; // record the position where max score is achieved
			m = m > h? m : h;   // m is stored at eh[mj+1]
			t = M - oe_del;
			t = t > 0? t : 0;
			e -= e_del;
			e = e > t? e : t;   // computed E(i+1,j)
			p->e = e;           // save E(i+1,j) for the next row
			t = M - oe_ins;
			t = t > 0? t : 0;
			f -= e_ins;
			f = f > t? f : t;   // computed F(i,j+1)
		}
		eh[end].h = h1; eh[end].e = 0;
		if (j == qlen) {
			max_ie = gscore > h1? max_ie : i;
			gscore = gscore > h1? gscore : h1;
		}
		if (m == 0) break;
		if (m > max) {
			max = m, max_i = i, max_j = mj;
			max_off = max_off > simple_cuda_abs(mj - i)? max_off : simple_cuda_abs(mj - i);
		} else if (zdrop > 0) {
			if (i - max_i > mj - max_j) {
				if (max - m - ((i - max_i) - (mj - max_j)) * e_del > zdrop) break;
			} else {
				if (max - m - ((mj - max_j) - (i - max_i)) * e_ins > zdrop) break;
			}
		}
		// update beg and end for the next round
		for (j = beg; j < end && eh[j].h == 0 && eh[j].e == 0; ++j);
		beg = j;
		for (j = end; j >= beg && eh[j].h == 0 && eh[j].e == 0; --j);
		end = j + 2 < qlen? j + 2 : qlen;
		//beg = 0; end = qlen; // uncomment this line for debugging
	}
	free(eh); free(qp);
	if (_qle) *_qle = max_j + 1;
	if (_tle) *_tle = max_i + 1;
	if (_gtle) *_gtle = max_ie + 1;
	if (_gscore) *_gscore = gscore;
	if (_max_off) *_max_off = max_off;
	return max;
}
__device__
void cuda_mem_chain2aln(const mem_opt_t *opt, const bntseq_t *bns, const uint8_t *pac, int l_query, const uint8_t *query, \
		const mem_chain_t *c, mem_alnreg_v *av)
{
	int i, k, rid, max_off[2], aw[2]; // aw: actual bandwidth used in extension
	int64_t l_pac = bns->l_pac, rmax[2], tmp, max;
	max = 0;
	const mem_seed_t *s;
	uint8_t *rseq = 0;
	uint64_t *srt;

	if (c->n == 0) return;
	// get the max possible span
	rmax[0] = l_pac << 1;
	rmax[1] = 0;
	for (i = 0; i < c->n; ++i) {
		int64_t b, e;
		const mem_seed_t *t = &c->seeds[i];
		b = t->rbeg - (t->qbeg + cal_max_gap(opt, t->qbeg));
		e = t->rbeg + t->len + ((l_query - t->qbeg - t->len) + cal_max_gap(opt, l_query - t->qbeg - t->len));
		rmax[0] = rmax[0] < b? rmax[0] : b;
		rmax[1] = rmax[1] > e? rmax[1] : e;
		if (t->len > max) max = t->len;
	}
	// rmax[0] = rmax[0] > 0? rmax[0] : 0;
	if(rmax[0] <= 0) rmax[0] = 0;
	// rmax[1] = rmax[1] < l_pac<<1? rmax[1] : l_pac<<1;
	if (rmax[1] >= (l_pac << 1)) rmax[1] = l_pac << 1;

	if (rmax[0] < l_pac && l_pac < rmax[1]) { // crossing the forward-reverse boundary; then choose one side
		// this works because all seeds are guaranteed to be on the same strand
		if (c->seeds[0].rbeg < l_pac) rmax[1] = l_pac;
		else rmax[0] = l_pac;
	}
	// retrieve the reference sequence
	rseq = cuda_bns_fetch_seq(bns, pac, &rmax[0], c->seeds[0].rbeg, &rmax[1], &rid);
	assert(c->rid == rid);
	srt = (uint64_t*)malloc(c->n * 8);
	assert(!(c->n != 0 && srt == NULL));
	for (i = 0; i < c->n; ++i)
		srt[i] = (uint64_t)c->seeds[i].score<<32 | i;
	cuda_ks_introsort_64(c->n, srt);
	for (k = c->n - 1; k >= 0; --k) {
		mem_alnreg_t *a;
		s = &c->seeds[(uint32_t)srt[k]];

		for (i = 0; i < av->n; ++i) { // test whether extension has been made before
			mem_alnreg_t *p = &av->a[i];
			int64_t rd;
			int qd, w, max_gap;
			if (s->rbeg < p->rb || s->rbeg + s->len > p->re || s->qbeg < p->qb || s->qbeg + s->len > p->qe)
				continue; // not fully contained
			if (s->len - p->seedlen0 > .1 * l_query) continue; // this seed may give a better alignment
			// qd: distance ahead of the seed on query; rd: on reference
			qd = s->qbeg - p->qb; rd = s->rbeg - p->rb;
			max_gap = cal_max_gap(opt, qd < rd? qd : rd); // the maximal gap allowed in regions ahead of the seed
			w = max_gap < p->w? max_gap : p->w; // bounded by the band width
			if (qd - rd < w && rd - qd < w) break; // the seed is "around" a previous hit
			// similar to the previous four lines, but this time we look at the region behind
			qd = p->qe - (s->qbeg + s->len); rd = p->re - (s->rbeg + s->len);
			max_gap = cal_max_gap(opt, qd < rd? qd : rd);
			w = max_gap < p->w? max_gap : p->w;
			if (qd - rd < w && rd - qd < w) break;
		}
		if (i < av->n) { // the seed is (almost) contained in an existing alignment; further testing is needed to confirm it is not leading to a different aln
			for (i = k + 1; i < c->n; ++i) { // check overlapping seeds in the same chain
				const mem_seed_t *t;
				if (srt[i] == 0) continue;
				t = &c->seeds[(uint32_t)srt[i]];
				if (t->len < s->len * .95) continue; // only check overlapping if t is long enough;
				// TODO: more efficient by early stopping
				if (s->qbeg <= t->qbeg && s->qbeg + s->len - t->qbeg >= s->len>>2 && t->qbeg - s->qbeg != t->rbeg - s->rbeg)
					break;
				if (t->qbeg <= s->qbeg && t->qbeg + t->len - s->qbeg >= s->len>>2 && s->qbeg - t->qbeg != s->rbeg - t->rbeg)
					break;
			}
			if (i == c->n) { // no overlapping seeds; then skip extension
				srt[k] = 0; // mark that seed extension has not been performed
				continue;
			}
		}

		a = cuda_kv_pushp_v2(*av);
		memset(a, 0, sizeof(mem_alnreg_t));
		a->w = aw[0] = aw[1] = opt->w;
		a->score = a->truesc = -1;
		a->rid = c->rid;

		if (s->qbeg) { // left extension
			uint8_t *rs, *qs;
			int qle, tle, gtle, gscore;
			qs = (uint8_t*)malloc(s->qbeg);
			assert(!(s->qbeg != 0 && qs == NULL));
			for (i = 0; i < s->qbeg; ++i) qs[i] = query[s->qbeg - 1 - i];
			tmp = s->rbeg - rmax[0];
			rs = (uint8_t*)malloc(tmp);
			assert(!(tmp != 0 && rs == NULL));
			for (i = 0; i < tmp; ++i) rs[i] = rseq[tmp - 1 - i];
			for (i = 0; i < MAX_BAND_TRY; ++i) {
				int prev = a->score;
				aw[0] = opt->w << i;
				a->score = cuda_ksw_extend2(s->qbeg, qs, tmp, rs, 5, opt->mat, opt->o_del, opt->e_del, opt->o_ins, opt->e_ins, \
						aw[0], opt->pen_clip5, opt->zdrop, s->len * opt->a, &qle, &tle, &gtle, &gscore, &max_off[0]);
				if (a->score == prev || max_off[0] < (aw[0]>>1) + (aw[0]>>2)) break;
			}
			// check whether we prefer to reach the end of the query
			if (gscore <= 0 || gscore <= a->score - opt->pen_clip5) { // local extension
				a->qb = s->qbeg - qle, a->rb = s->rbeg - tle;
				a->truesc = a->score;
			} else { // to-end extension
				a->qb = 0, a->rb = s->rbeg - gtle;
				a->truesc = gscore;
			}
			free(qs); free(rs);
		} else a->score = a->truesc = s->len * opt->a, a->qb = 0, a->rb = s->rbeg;

		if (s->qbeg + s->len != l_query) { // right extension
			int qle, tle, qe, re, gtle, gscore, sc0 = a->score;
			qe = s->qbeg + s->len;
			re = s->rbeg + s->len - rmax[0];
			assert(re >= 0);
			for (i = 0; i < MAX_BAND_TRY; ++i) {
				int prev = a->score;
				aw[1] = opt->w << i;
				a->score = cuda_ksw_extend2(l_query - qe, query + qe, rmax[1] - rmax[0] - re, rseq + re, 5, opt->mat, \
						opt->o_del, opt->e_del, opt->o_ins, opt->e_ins, aw[1], opt->pen_clip3, opt->zdrop, sc0, \
						&qle, &tle, &gtle, &gscore, &max_off[1]);
				if (a->score == prev || max_off[1] < (aw[1]>>1) + (aw[1]>>2)) break;
			}
			// similar to the above
			if (gscore <= 0 || gscore <= a->score - opt->pen_clip3) { // local extension
				a->qe = qe + qle, a->re = rmax[0] + re + tle;
				a->truesc += a->score - sc0;
			} else { // to-end extension
				a->qe = l_query, a->re = rmax[0] + re + gtle;
				a->truesc += gscore - sc0;
			}
		} else a->qe = l_query, a->re = s->rbeg + s->len;
		// compute seedcov
		for (i = 0, a->seedcov = 0; i < c->n; ++i) {
			const mem_seed_t *t = &c->seeds[i];
			// seed fully contained
			if (t->qbeg >= a->qb && t->qbeg + t->len <= a->qe && t->rbeg >= a->rb && t->rbeg + t->len <= a->re)
				a->seedcov += t->len; // this is not very accurate, but for approx. mapQ, this is good enough
		}
		a->w = aw[0] > aw[1]? aw[0] : aw[1];
		a->seedlen0 = s->len;

		a->frac_rep = c->frac_rep;
	}
	free(srt); free(rseq);
}

void print_seq(int n, int *l_seq) {
	for(int i = 0; i < n; i++) {
		fprintf(stderr, "[M::%s] seq[%d], l = %d.\n", __func__, i, l_seq[i]);
	}
}

void print_chns(int n, mem_chain_v *f_chns, mem_chain_t *f_a, int *i_a) {
	for(int i = 0; i < n; i++) {
		fprintf(stderr, "[M::%s] chns[%d], n = %lu, m = %lu.\n", __func__, i, f_chns[i].n, f_chns[i].m);
		for(int j = 0; j < f_chns[i].n; j++) {
			fprintf(stderr, "[M::%s] 		chns[%d].a[%d], n = %d, m = %d.\n",\
					__func__, i, j, f_a[i_a[i] + j].n, f_a[i_a[i] + j].m);
		}
	}
}

void print_bns_pac(int64_t l_pac, int32_t n_seqs) {
	fprintf(stderr, "[M::%s] bns, n_seqs = %" PRId32 " | l_pac = %" PRId64 ".\n", __func__, n_seqs, l_pac);
}
