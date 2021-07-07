source("src/functions.R")

arithmetic <- commandArgs(trailingOnly=TRUE)[1]
method <- commandArgs(trailingOnly=TRUE)[2]
size <- commandArgs(trailingOnly=TRUE)[3]
outfile <- commandArgs(trailingOnly=TRUE)[4]

profmem_xxx(arithmetic, method, size, outfile)