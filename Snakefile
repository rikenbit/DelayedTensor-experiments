from snakemake.utils import min_version

min_version("6.0.5")
container: 'docker://koki/delayedtensor-experiments:20210709'

ARITHMETICS = ["unfold", "modesum", "innerprod", "vec", "kronecker", "fold", "hosvd", "cp", "tucker", "mpca", "pvd", "einsum"]
METHODS = ["rtensor", "dense_delayedtensor", "sparse_delayedtensor"]
SIZES = ["1E7", "5E7", "1E8", "5E8"]

rule all:
	input:
		expand('plot/profmem/time/{a}.png', a=ARITHMETICS),
		expand('plot/profmem/memory/{a}.png', a=ARITHMETICS),
		expand('plot/benchmark/time/{a}.png', a=ARITHMETICS),
		expand('plot/benchmark/memory/{a}.png', a=ARITHMETICS),
		expand('plot/gnutime/time/{a}.png', a=ARITHMETICS),
		expand('plot/gnutime/memory/{a}.png', a=ARITHMETICS)

rule profmem:
	output:
		'profmem/{a}_{m}_{s}.RData'
	resources:
		mem_gb=50
	benchmark:
		'benchmarks/{a}_{m}_{s}.txt'
	log:
		'logs/{a}_{m}_{s}.log'
	shell:
		'/usr/bin/time -v src/profmem.sh {wildcards.a} {wildcards.m} {wildcards.s} {output} >& {log}'

rule plot_profmem_time:
	input:
		expand('profmem/{a}_{m}_{s}.RData',
			a=ARITHMETICS, m=METHODS, s=SIZES)
	output:
		'plot/profmem/time/{a}.png'
	resources:
		mem_gb=50
	benchmark:
		'benchmarks/plot_profmem_time_{a}.txt'
	log:
		'logs/plot_profmem_time_{a}.log'
	shell:
		'src/plot_profmem_time.sh {wildcards.a} {output} >& {log}'

rule plot_profmem_memory:
	input:
		expand('profmem/{a}_{m}_{s}.RData',
			a=ARITHMETICS, m=METHODS, s=SIZES)
	output:
		'plot/profmem/memory/{a}.png'
	resources:
		mem_gb=50
	benchmark:
		'benchmarks/plot_profmem_memory_{a}.txt'
	log:
		'logs/plot_profmem_memory_{a}.log'
	shell:
		'src/plot_profmem_memory.sh {wildcards.a} {output} >& {log}'

rule plot_benchmark_time:
	input:
		expand('profmem/{a}_{m}_{s}.RData',
			a=ARITHMETICS, m=METHODS, s=SIZES)
	output:
		'plot/benchmark/time/{a}.png'
	resources:
		mem_gb=50
	benchmark:
		'benchmarks/plot_benchmark_time_{a}.txt'
	log:
		'logs/plot_benchmark_time_{a}.log'
	shell:
		'src/plot_benchmark_time.sh {wildcards.a} {output} >& {log}'

rule plot_benchmark_memory:
	input:
		expand('profmem/{a}_{m}_{s}.RData',
			a=ARITHMETICS, m=METHODS, s=SIZES)
	output:
		'plot/benchmark/memory/{a}.png'
	resources:
		mem_gb=50
	benchmark:
		'benchmarks/plot_benchmark_memory_{a}.txt'
	log:
		'logs/plot_benchmark_memory_{a}.log'
	shell:
		'src/plot_benchmark_memory.sh {wildcards.a} {output} >& {log}'

rule plot_gnutime_time:
	input:
		expand('profmem/{a}_{m}_{s}.RData',
			a=ARITHMETICS, m=METHODS, s=SIZES)
	output:
		'plot/gnutime/time/{a}.png'
	resources:
		mem_gb=50
	benchmark:
		'benchmarks/plot_gnutime_time_{a}.txt'
	log:
		'logs/plot_gnutime_time_{a}.log'
	shell:
		'src/plot_gnutime_time.sh {wildcards.a} {output} >& {log}'

rule plot_gnutime_memory:
	input:
		expand('profmem/{a}_{m}_{s}.RData',
			a=ARITHMETICS, m=METHODS, s=SIZES)
	output:
		'plot/gnutime/memory/{a}.png'
	resources:
		mem_gb=50
	benchmark:
		'benchmarks/plot_gnutime_memory_{a}.txt'
	log:
		'logs/plot_gnutime_memory_{a}.log'
	shell:
		'src/plot_gnutime_memory.sh {wildcards.a} {output} >& {log}'
