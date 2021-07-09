source("src/functions.R")

arithmetic <- commandArgs(trailingOnly=TRUE)[1]
outfile <- commandArgs(trailingOnly=TRUE)[2]

plot_profmem_memory(arithmetic, outfile)