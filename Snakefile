from snakemake.utils import min_version

min_version("6.0.5")
container: 'docker://koki/delayedtensor-experiments:20210707'

ARITHMETICS = ["unfold", "modesum", "innerprod", "vec", "kronecker", "fold", "hosvd", "cp", "tucker", "mpca", "pvd", "einsum"]
METHODS = ["rtensor", "dense_delayedtensor", "sparse_delayedtensor"]
SIZES = ["1E2", "1E3", "1E4", "1E5", "1E6", "1E7", "1E8", "1E9"]

rule all:
	input:
		expand('plot/time/{a}.png', a=ARITHMETICS),
		expand('plot/memory/{a}.png', a=ARITHMETICS)

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
		'src/profmem.sh {wildcards.a} {wildcards.m} {wildcards.s} {output} || true >& {log}'

rule plot_time:
	input:
		expand('profmem/{a}_{m}_{s}.RData',
			a=ARITHMETICS, m=METHODS, s=SIZES)
	output:
		'plot/time/{a}.png'
	resources:
		mem_gb=50
	benchmark:
		'benchmarks/plot_time_{a}.txt'
	log:
		'logs/plot_time_{a}.log'
	shell:
		'src/plot_time.sh {wildcards.a} {output} >& {log}'

rule plot_memory:
	input:
		expand('profmem/{a}_{m}_{s}.RData',
			a=ARITHMETICS, m=METHODS, s=SIZES)
	output:
		'plot/memory/{a}.png'
	resources:
		mem_gb=50
	benchmark:
		'benchmarks/plot_memory_{a}.txt'
	log:
		'logs/plot_memory_{a}.log'
	shell:
		'src/plot_memory.sh {wildcards.a} {output} >& {log}'
