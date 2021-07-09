# Package Loading
library("DelayedTensor")
library("DelayedRandomArray")
library("rTensor")
library("ggplot2")
library("scales")
library("RColorBrewer")
library("viridis")
library("matter")
library("einsum")

setAutoBlockSize(size=1E+8)
setVerbose(TRUE)

#######################################################
# All Functions for profiling
#######################################################

profmem_xxx <- function(arithmetic, method, size, outfile){
	cmd <- paste0("profmem_", arithmetic, "_", method,
		"(size, outfile)")
	eval(parse(text=cmd))
}

# unfold
size_unfold <- list(
    "1E7" = c(170,200,300),
    "5E7" = c(170,200,1500),
    "1E8" = c(400,500,500),
    "5E8" = c(400,500,2500),
    "1E9" = c(1000,1000,1000)
)

profmem_unfold_rtensor <- function(size, outfile){
    data <- rand_tensor(size_unfold[[size]])
    out <- profmem(rTensor::unfold(data, row_idx=1:2, col_idx=3))
    save(out, file=outfile)
}

profmem_unfold_dense_delayedtensor <- function(size, outfile){
    setSparse(FALSE)
    data <- RandomNormArray(size_unfold[[size]])
    out <- profmem(DelayedTensor::unfold(data, row_idx=1:2, col_idx=3))
    save(out, file=outfile)
}

profmem_unfold_sparse_delayedtensor <- function(size, outfile){
    setSparse(TRUE)
    data <- RandomBinomArray(size_unfold[[size]], size=1, prob=0.1)
    out <- profmem(DelayedTensor::unfold(data, row_idx=1:2,
        col_idx=3))
    save(out, file=outfile)
}

# modesum
size_modesum <- size_unfold

profmem_modesum_rtensor <- function(size, outfile){
    data <- rand_tensor(size_modesum[[size]])
    out <- profmem(rTensor::modeSum(data, m=1, drop=FALSE))
    save(out, file=outfile)
}

profmem_modesum_dense_delayedtensor <- function(size, outfile){
    setSparse(FALSE)
    data <- RandomNormArray(size_modesum[[size]])
    out <- profmem(DelayedTensor::modeSum(data, m=1, drop=FALSE))
    save(out, file=outfile)
}

profmem_modesum_sparse_delayedtensor <- function(size, outfile){
    setSparse(TRUE)
    data <- RandomBinomArray(size_modesum[[size]],
        size=1, prob=0.1)
    out <- profmem(DelayedTensor::modeSum(data, m=1, drop=FALSE))
    save(out, file=outfile)
}

# innerprod
size_innerprod <- size_unfold

profmem_innerprod_rtensor <- function(size, outfile){
    data <- rand_tensor(size_innerprod[[size]])
    out <- profmem(rTensor::innerProd(data, data))
    save(out, file=outfile)
}

profmem_innerprod_dense_delayedtensor <- function(size, outfile){
    setSparse(FALSE)
    data <- RandomNormArray(size_innerprod[[size]])
    out <- profmem(DelayedTensor::innerProd(data, data))
    save(out, file=outfile)
}

profmem_innerprod_sparse_delayedtensor <- function(size, outfile){
    setSparse(TRUE)
    data <- RandomBinomArray(size_innerprod[[size]],
        size=1, prob=0.1)
    out <- profmem(DelayedTensor::innerProd(data, data))
    save(out, file=outfile)
}

# vec
size_vec <- size_unfold

profmem_vec_rtensor <- function(size, outfile){
    data <- rand_tensor(size_vec[[size]])
    out <- profmem(rTensor::vec(data))
    save(out, file=outfile)
}

profmem_vec_dense_delayedtensor <- function(size, outfile){
    setSparse(FALSE)
    data <- RandomNormArray(size_vec[[size]])
    out <- profmem(DelayedTensor::vec(data))
    save(out, file=outfile)
}

profmem_vec_sparse_delayedtensor <- function(size, outfile){
    setSparse(TRUE)
    data <- RandomBinomArray(size_vec[[size]],
        size=1, prob=0.1)
    out <- profmem(DelayedTensor::vec(data))
    save(out, file=outfile)
}

# kronecker
size_kronecker <- list(
    "1E7" = list(c(100,100), c(100,10)),
    "5E7" = list(c(100,100), c(100,50)),
    "1E8" = list(c(100,100), c(100,100)),
    "5E8" = list(c(100,100), c(100,500)),
    "1E9" = list(c(1000,100), c(100,100))
)

profmem_kronecker_rtensor <- function(size, outfile){
    data1 <- rand_tensor(size_kronecker[[size]][[1]])
    data2 <- rand_tensor(size_kronecker[[size]][[2]])
    out <- profmem(kronecker(data1@data, data2@data))
    save(out, file=outfile)
}

profmem_kronecker_dense_delayedtensor <- function(size, outfile){
    setSparse(FALSE)
    data1 <- RandomNormArray(size_kronecker[[size]][[1]])
    data2 <- RandomNormArray(size_kronecker[[size]][[2]])
    out <- profmem(DelayedTensor::kronecker(data1, data2))
    save(out, file=outfile)
}

profmem_kronecker_sparse_delayedtensor <- function(size, outfile){
    setSparse(TRUE)
    data1 <- RandomBinomArray(size_kronecker[[size]][[1]],
        size=1, prob=0.1)
    data2 <- RandomBinomArray(size_kronecker[[size]][[2]],
        size=1, prob=0.1)
    out <- profmem(DelayedTensor::kronecker(data1, data2))
    save(out, file=outfile)
}

# fold
profmem_fold_rtensor <- function(size, outfile){
    data <- rand_tensor(size_unfold[[size]])
    tmp <- rTensor::unfold(data, row_idx=1:2, col_idx=3)
    out <- profmem(rTensor::fold(tmp,
    	row_idx=1:2, col_idx=3, modes=dim(data)))
    save(out, file=outfile)
}

profmem_fold_dense_delayedtensor <- function(size, outfile){
    setSparse(FALSE)
    data <- RandomNormArray(size_unfold[[size]])
    tmp <- DelayedTensor::unfold(data, row_idx=1:2, col_idx=3)
    out <- profmem(DelayedTensor::fold(tmp,
    	row_idx=1:2, col_idx=3, modes=dim(data)))
    save(out, file=outfile)
}

profmem_fold_sparse_delayedtensor <- function(size, outfile){
    setSparse(TRUE)
    data <- RandomBinomArray(size_unfold[[size]], size=1, prob=0.1)
    tmp <- DelayedTensor::unfold(data, row_idx=1:2, col_idx=3)
    out <- profmem(DelayedTensor::fold(tmp,
    	row_idx=1:2, col_idx=3, modes=dim(data)))
    save(out, file=outfile)
}

# hosvd
size_hosvd <- list(
    "1E7" = c(1000,100,100),
    "5E7" = c(1000,100,500),
    "1E8" = c(1000,1000,100),
    "5E8" = c(1000,1000,500),
    "1E9" = c(1000,1000,1000)
)

profmem_hosvd_rtensor <- function(size, outfile){
    data <- rand_tensor(size_hosvd[[size]])
    out <- profmem(rTensor::hosvd(data, ranks=c(2,3,4)))
    save(out, file=outfile)
}

profmem_hosvd_dense_delayedtensor <- function(size, outfile){
    setSparse(FALSE)
    data <- RandomNormArray(size_hosvd[[size]])
    out <- profmem(DelayedTensor::hosvd(data, ranks=c(2,3,4)))
    save(out, file=outfile)
}

profmem_hosvd_sparse_delayedtensor <- function(size, outfile){
    setSparse(TRUE)
    data <- RandomBinomArray(size_hosvd[[size]],
        size=1, prob=0.1)
    out <- profmem(DelayedTensor::hosvd(data, ranks=c(2,3,4)))
    save(out, file=outfile)
}

# cp
size_cp <- size_hosvd

profmem_cp_rtensor <- function(size, outfile){
    data <- rand_tensor(size_cp[[size]])
    out <- profmem(rTensor::cp(data, num_components=2))
    save(out, file=outfile)
}

profmem_cp_dense_delayedtensor <- function(size, outfile){
    setSparse(FALSE)
    data <- RandomNormArray(size_cp[[size]])
    out <- profmem(DelayedTensor::cp(data, num_components=2))
    save(out, file=outfile)
}

profmem_cp_sparse_delayedtensor <- function(size, outfile){
    setSparse(TRUE)
    data <- RandomBinomArray(size_cp[[size]],
        size=1, prob=0.1)
    out <- profmem(DelayedTensor::cp(data, num_components=2))
    save(out, file=outfile)
}

# tucker
size_tucker <- size_hosvd

profmem_tucker_rtensor <- function(size, outfile){
    data <- rand_tensor(size_tucker[[size]])
    out <- profmem(rTensor::tucker(data, ranks=c(2,3,4)))
    save(out, file=outfile)
}

profmem_tucker_dense_delayedtensor <- function(size, outfile){
    setSparse(FALSE)
    data <- RandomNormArray(size_tucker[[size]])
    out <- profmem(DelayedTensor::tucker(data, ranks=c(2,3,4)))
    save(out, file=outfile)
}

profmem_tucker_sparse_delayedtensor <- function(size, outfile){
    setSparse(TRUE)
    data <- RandomBinomArray(size_tucker[[size]],
        size=1, prob=0.1)
    out <- profmem(DelayedTensor::tucker(data, ranks=c(2,3,4)))
    save(out, file=outfile)
}

# mpca
size_mpca <- size_hosvd

profmem_mpca_rtensor <- function(size, outfile){
    data <- rand_tensor(size_mpca[[size]])
    out <- profmem(rTensor::mpca(data, ranks=c(2,3)))
    save(out, file=outfile)
}

profmem_mpca_dense_delayedtensor <- function(size, outfile){
    setSparse(FALSE)
    data <- RandomNormArray(size_mpca[[size]])
    out <- profmem(DelayedTensor::mpca(data, ranks=c(2,3)))
    save(out, file=outfile)
}

profmem_mpca_sparse_delayedtensor <- function(size, outfile){
    setSparse(TRUE)
    data <- RandomBinomArray(size_mpca[[size]],
        size=1, prob=0.1)
    out <- profmem(DelayedTensor::mpca(data, ranks=c(2,3)))
    save(out, file=outfile)
}

# pvd
size_pvd <- size_hosvd

profmem_pvd_rtensor <- function(size, outfile){
    data <- rand_tensor(size_pvd[[size]])
    out <- profmem(rTensor::pvd(data,
        uranks=rep(2,dim(data)[3]),
        wranks=rep(3,dim(data)[3]), a=2, b=3))
    save(out, file=outfile)
}

profmem_pvd_dense_delayedtensor <- function(size, outfile){
    setSparse(FALSE)
    data <- RandomNormArray(size_pvd[[size]])
    out <- profmem(DelayedTensor::pvd(data,
        uranks=rep(2,dim(data)[3]),
        wranks=rep(3,dim(data)[3]), a=2, b=3))
    save(out, file=outfile)
}

profmem_pvd_sparse_delayedtensor <- function(size, outfile){
    setSparse(TRUE)
    data <- RandomBinomArray(size_pvd[[size]],
        size=1, prob=0.1)
    out <- profmem(DelayedTensor::pvd(data,
        uranks=rep(2,dim(data)[3]),
        wranks=rep(3,dim(data)[3]), a=2, b=3))
    save(out, file=outfile)
}

# einsum
size_einsum <- list(
    "1E7" = list(c(100,100), c(100,10)),
    "5E7" = list(c(100,100), c(100,50)),
    "1E8" = list(c(100,100), c(100,100)),
    "5E8" = list(c(100,100), c(100,500)),
    "1E9" = list(c(1000,100), c(100,100))
)

profmem_einsum_rtensor <- function(size, outfile){
    data1 <- rand_tensor(size_einsum[[size]][[1]])
    data2 <- rand_tensor(size_einsum[[size]][[2]])
    out <- profmem(einsum('ij,jk->ijk',
        data1@data, data2@data))
    save(out, file=outfile)
}

profmem_einsum_dense_delayedtensor <- function(size, outfile){
    setSparse(FALSE)
    data1 <- RandomNormArray(size_einsum[[size]][[1]])
    data2 <- RandomNormArray(size_einsum[[size]][[2]])
    out <- profmem(DelayedTensor::einsum('ij,jk->ijk',
        data1, data2))
    save(out, file=outfile)
}

profmem_einsum_sparse_delayedtensor <- function(size, outfile){
    setSparse(TRUE)
    data1 <- RandomBinomArray(size_einsum[[size]][[1]],
        size=1, prob=0.1)
    data2 <- RandomBinomArray(size_einsum[[size]][[2]],
        size=1, prob=0.1)
    out <- profmem(DelayedTensor::einsum('ij,jk->ijk',
        data1, data2))
    save(out, file=outfile)
}



#######################################################
# All Functions for Visualization (Calculation time)
#######################################################
# profmem
plot_profmem_time <- function(arithmetic, outfile){
	cmd <- paste0("plot_profmem_time_", arithmetic, "(outfile)")
	eval(parse(text=cmd))
}

aggregate_profmem_time_unfold <- function(){
    files <- list.files("profmem", pattern="^unfold")
    sec <- sapply(files, function(x){
        load(paste0("profmem/", x))
        out[[5]]})
    elements <- gsub("unfold_.*tensor_", "", gsub(".RData", "", files))
    method <- gsub("unfold_", "", gsub("_1E.*RData", "", files))
    method <- factor(method,
    	level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, sec, method)
    colnames(data) <- c("elements", "sec", "method")
    data
}

plot_profmem_time_unfold <- function(outfile){
	data <- aggregate_profmem_time_unfold()
	g <- ggplot(data, aes(x = elements, y = sec, fill = method))
	g <- g + geom_bar(stat = "identity", position = "dodge")
	g <- g + ggtitle("unfold()")
	g <- g + xlab("# Elements")
	g <- g + ylab("Calculation Time (sec)")
	g <- g + theme(legend.title = element_blank())
	g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
	g <- g + theme(axis.text.x = element_text(size=12))
	g <- g + theme(axis.text.y = element_text(size=12))
	g <- g + theme(axis.title.x = element_text(size=18))
	g <- g + theme(axis.title.y = element_text(size=18))
	ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_profmem_time_modesum <- function(){
    files <- list.files("profmem", pattern="^modesum")
    sec <- sapply(files, function(x){
        load(paste0("profmem/", x))
        out[[5]]})
    elements <- gsub("modesum_.*tensor_", "", gsub(".RData", "", files))
    method <- gsub("modesum_", "", gsub("_1E.*RData", "", files))
    method <- factor(method,
    	level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, sec, method)
    colnames(data) <- c("elements", "sec", "method")
    data
}

plot_profmem_time_modesum <- function(outfile){
	data <- aggregate_profmem_time_modesum()
	g <- ggplot(data, aes(x = elements, y = sec, fill = method))
	g <- g + geom_bar(stat = "identity", position = "dodge")
	g <- g + ggtitle("modeSum()")
	g <- g + xlab("# Elements")
	g <- g + ylab("Calculation Time (sec)")
	g <- g + theme(legend.title = element_blank())
	g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
	g <- g + theme(axis.text.x = element_text(size=12))
	g <- g + theme(axis.text.y = element_text(size=12))
	g <- g + theme(axis.title.x = element_text(size=18))
	g <- g + theme(axis.title.y = element_text(size=18))
	ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_profmem_time_innerprod <- function(){
    files <- list.files("profmem", pattern="^innerprod")
    sec <- sapply(files, function(x){
        load(paste0("profmem/", x))
        out[[5]]})
    elements <- gsub("innerprod_.*tensor_", "", gsub(".RData", "", files))
    method <- gsub("innerprod_", "", gsub("_1E.*RData", "", files))
    method <- factor(method,
    	level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, sec, method)
    colnames(data) <- c("elements", "sec", "method")
    data
}

plot_profmem_time_innerprod <- function(outfile){
	data <- aggregate_profmem_time_innerprod()
	g <- ggplot(data, aes(x = elements, y = sec, fill = method))
	g <- g + geom_bar(stat = "identity", position = "dodge")
	g <- g + ggtitle("innerProd()")
	g <- g + xlab("# Elements")
	g <- g + ylab("Calculation Time (sec)")
	g <- g + theme(legend.title = element_blank())
	g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
	g <- g + theme(axis.text.x = element_text(size=12))
	g <- g + theme(axis.text.y = element_text(size=12))
	g <- g + theme(axis.title.x = element_text(size=18))
	g <- g + theme(axis.title.y = element_text(size=18))
	ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_profmem_time_vec <- function(){
    files <- list.files("profmem", pattern="^vec")
    sec <- sapply(files, function(x){
        load(paste0("profmem/", x))
        out[[5]]})
    elements <- gsub("vec_.*tensor_", "", gsub(".RData", "", files))
    method <- gsub("vec_", "", gsub("_1E.*RData", "", files))
    method <- factor(method,
    	level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, sec, method)
    colnames(data) <- c("elements", "sec", "method")
    data
}

plot_profmem_time_vec <- function(outfile){
	data <- aggregate_profmem_time_vec()
	g <- ggplot(data, aes(x = elements, y = sec, fill = method))
	g <- g + geom_bar(stat = "identity", position = "dodge")
	g <- g + ggtitle("vec()")
	g <- g + xlab("# Elements")
	g <- g + ylab("Calculation Time (sec)")
	g <- g + theme(legend.title = element_blank())
	g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
	g <- g + theme(axis.text.x = element_text(size=12))
	g <- g + theme(axis.text.y = element_text(size=12))
	g <- g + theme(axis.title.x = element_text(size=18))
	g <- g + theme(axis.title.y = element_text(size=18))
	ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_profmem_time_kronecker <- function(){
    files <- list.files("profmem", pattern="^kronecker")
    sec <- sapply(files, function(x){
        load(paste0("profmem/", x))
        out[[5]]})
    elements <- gsub("kronecker_.*tensor_", "", gsub(".RData", "", files))
    method <- gsub("kronecker_", "", gsub("_1E.*RData", "", files))
    method <- factor(method,
    	level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, sec, method)
    colnames(data) <- c("elements", "sec", "method")
    data
}

plot_profmem_time_kronecker <- function(outfile){
	data <- aggregate_profmem_time_kronecker()
	g <- ggplot(data, aes(x = elements, y = sec, fill = method))
	g <- g + geom_bar(stat = "identity", position = "dodge")
	g <- g + ggtitle("kronecker()")
	g <- g + xlab("# Elements")
	g <- g + ylab("Calculation Time (sec)")
	g <- g + theme(legend.title = element_blank())
	g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
	g <- g + theme(axis.text.x = element_text(size=12))
	g <- g + theme(axis.text.y = element_text(size=12))
	g <- g + theme(axis.title.x = element_text(size=18))
	g <- g + theme(axis.title.y = element_text(size=18))
	ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_profmem_time_fold <- function(){
    files <- list.files("profmem", pattern="^fold")
    sec <- sapply(files, function(x){
        load(paste0("profmem/", x))
        out[[5]]})
    elements <- gsub("fold_.*tensor_", "", gsub(".RData", "", files))
    method <- gsub("fold_", "", gsub("_1E.*RData", "", files))
    method <- factor(method,
    	level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, sec, method)
    colnames(data) <- c("elements", "sec", "method")
    data
}

plot_profmem_time_fold <- function(outfile){
	data <- aggregate_profmem_time_fold()
	g <- ggplot(data, aes(x = elements, y = sec, fill = method))
	g <- g + geom_bar(stat = "identity", position = "dodge")
	g <- g + ggtitle("fold()")
	g <- g + xlab("# Elements")
	g <- g + ylab("Calculation Time (sec)")
	g <- g + theme(legend.title = element_blank())
	g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
	g <- g + theme(axis.text.x = element_text(size=12))
	g <- g + theme(axis.text.y = element_text(size=12))
	g <- g + theme(axis.title.x = element_text(size=18))
	g <- g + theme(axis.title.y = element_text(size=18))
	ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_profmem_time_hosvd <- function(){
    files <- list.files("profmem", pattern="^hosvd")
    sec <- sapply(files, function(x){
        load(paste0("profmem/", x))
        out[[5]]})
    elements <- gsub("hosvd_.*tensor_", "", gsub(".RData", "", files))
    method <- gsub("hosvd_", "", gsub("_1E.*RData", "", files))
    method <- factor(method,
    	level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, sec, method)
    colnames(data) <- c("elements", "sec", "method")
    data
}

plot_profmem_time_hosvd <- function(outfile){
	data <- aggregate_profmem_time_hosvd()
	g <- ggplot(data, aes(x = elements, y = sec, fill = method))
	g <- g + geom_bar(stat = "identity", position = "dodge")
	g <- g + ggtitle("hosvd()")
	g <- g + xlab("# Elements")
	g <- g + ylab("Calculation Time (sec)")
	g <- g + theme(legend.title = element_blank())
	g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
	g <- g + theme(axis.text.x = element_text(size=12))
	g <- g + theme(axis.text.y = element_text(size=12))
	g <- g + theme(axis.title.x = element_text(size=18))
	g <- g + theme(axis.title.y = element_text(size=18))
	ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_profmem_time_cp <- function(){
    files <- list.files("profmem", pattern="^cp")
    sec <- sapply(files, function(x){
        load(paste0("profmem/", x))
        out[[5]]})
    elements <- gsub("cp_.*tensor_", "", gsub(".RData", "", files))
    method <- gsub("cp_", "", gsub("_1E.*RData", "", files))
    method <- factor(method,
    	level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, sec, method)
    colnames(data) <- c("elements", "sec", "method")
    data
}

plot_profmem_time_cp <- function(outfile){
	data <- aggregate_profmem_time_cp()
	g <- ggplot(data, aes(x = elements, y = sec, fill = method))
	g <- g + geom_bar(stat = "identity", position = "dodge")
	g <- g + ggtitle("cp()")
	g <- g + xlab("# Elements")
	g <- g + ylab("Calculation Time (sec)")
	g <- g + theme(legend.title = element_blank())
	g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
	g <- g + theme(axis.text.x = element_text(size=12))
	g <- g + theme(axis.text.y = element_text(size=12))
	g <- g + theme(axis.title.x = element_text(size=18))
	g <- g + theme(axis.title.y = element_text(size=18))
	ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_profmem_time_tucker <- function(){
    files <- list.files("profmem", pattern="^tucker")
    sec <- sapply(files, function(x){
        load(paste0("profmem/", x))
        out[[5]]})
    elements <- gsub("tucker_.*tensor_", "", gsub(".RData", "", files))
    method <- gsub("tucker_", "", gsub("_1E.*RData", "", files))
    method <- factor(method,
    	level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, sec, method)
    colnames(data) <- c("elements", "sec", "method")
    data
}

plot_profmem_time_tucker <- function(outfile){
	data <- aggregate_profmem_time_tucker()
	g <- ggplot(data, aes(x = elements, y = sec, fill = method))
	g <- g + geom_bar(stat = "identity", position = "dodge")
	g <- g + ggtitle("tucker()")
	g <- g + xlab("# Elements")
	g <- g + ylab("Calculation Time (sec)")
	g <- g + theme(legend.title = element_blank())
	g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
	g <- g + theme(axis.text.x = element_text(size=12))
	g <- g + theme(axis.text.y = element_text(size=12))
	g <- g + theme(axis.title.x = element_text(size=18))
	g <- g + theme(axis.title.y = element_text(size=18))
	ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_profmem_time_mpca <- function(){
    files <- list.files("profmem", pattern="^mpca")
    sec <- sapply(files, function(x){
        load(paste0("profmem/", x))
        out[[5]]})
    elements <- gsub("mpca_.*tensor_", "", gsub(".RData", "", files))
    method <- gsub("mpca_", "", gsub("_1E.*RData", "", files))
    method <- factor(method,
    	level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, sec, method)
    colnames(data) <- c("elements", "sec", "method")
    data
}

plot_profmem_time_mpca <- function(outfile){
	data <- aggregate_profmem_time_mpca()
	g <- ggplot(data, aes(x = elements, y = sec, fill = method))
	g <- g + geom_bar(stat = "identity", position = "dodge")
	g <- g + ggtitle("mpca()")
	g <- g + xlab("# Elements")
	g <- g + ylab("Calculation Time (sec)")
	g <- g + theme(legend.title = element_blank())
	g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
	g <- g + theme(axis.text.x = element_text(size=12))
	g <- g + theme(axis.text.y = element_text(size=12))
	g <- g + theme(axis.title.x = element_text(size=18))
	g <- g + theme(axis.title.y = element_text(size=18))
	ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_profmem_time_pvd <- function(){
    files <- list.files("profmem", pattern="^pvd")
    sec <- sapply(files, function(x){
        load(paste0("profmem/", x))
        out[[5]]})
    elements <- gsub("pvd_.*tensor_", "", gsub(".RData", "", files))
    method <- gsub("pvd_", "", gsub("_1E.*RData", "", files))
    method <- factor(method,
    	level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, sec, method)
    colnames(data) <- c("elements", "sec", "method")
    data
}

plot_profmem_time_pvd <- function(outfile){
	data <- aggregate_profmem_time_pvd()
	g <- ggplot(data, aes(x = elements, y = sec, fill = method))
	g <- g + geom_bar(stat = "identity", position = "dodge")
	g <- g + ggtitle("pvd()")
	g <- g + xlab("# Elements")
	g <- g + ylab("Calculation Time (sec)")
	g <- g + theme(legend.title = element_blank())
	g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
	g <- g + theme(axis.text.x = element_text(size=12))
	g <- g + theme(axis.text.y = element_text(size=12))
	g <- g + theme(axis.title.x = element_text(size=18))
	g <- g + theme(axis.title.y = element_text(size=18))
	ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_profmem_time_einsum <- function(){
    files <- list.files("profmem", pattern="^einsum")
    sec <- sapply(files, function(x){
        load(paste0("profmem/", x))
        out[[5]]})
    elements <- gsub("einsum_.*tensor_", "", gsub(".RData", "", files))
    method <- gsub("einsum_", "", gsub("_1E.*RData", "", files))
    method <- factor(method,
    	level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, sec, method)
    colnames(data) <- c("elements", "sec", "method")
    data
}

plot_profmem_time_einsum <- function(outfile){
	data <- aggregate_profmem_time_einsum()
	g <- ggplot(data, aes(x = elements, y = sec, fill = method))
	g <- g + geom_bar(stat = "identity", position = "dodge")
	g <- g + ggtitle("einsum()")
	g <- g + xlab("# Elements")
	g <- g + ylab("Calculation Time (sec)")
	g <- g + theme(legend.title = element_blank())
	g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
	g <- g + theme(axis.text.x = element_text(size=12))
	g <- g + theme(axis.text.y = element_text(size=12))
	g <- g + theme(axis.title.x = element_text(size=18))
	g <- g + theme(axis.title.y = element_text(size=18))
	ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

# benchmark
plot_benchmark_time <- function(arithmetic, outfile){
    cmd <- paste0("plot_benchmark_time_", arithmetic, "(outfile)")
    eval(parse(text=cmd))
}

aggregate_benchmark_time_unfold <- function(){
    files <- list.files("benchmarks", pattern="^unfold")
    sec <- sapply(files, function(x){
        read.table(paste0("benchmarks/", x), stringsAsFactors=FALSE)[2,1]})
    sec <- as.numeric(sec)
    elements <- gsub("unfold_.*tensor_", "", gsub(".txt", "", files))
    method <- gsub("unfold_", "", gsub("_1E.*txt", "", files))
    method <- factor(method,
        level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, sec, method)
    colnames(data) <- c("elements", "sec", "method")
    data
}

plot_benchmark_time_unfold <- function(outfile){
    data <- aggregate_benchmark_time_unfold()
    g <- ggplot(data, aes(x = elements, y = sec, fill = method))
    g <- g + geom_bar(stat = "identity", position = "dodge")
    g <- g + ggtitle("unfold()")
    g <- g + xlab("# Elements")
    g <- g + ylab("Calculation Time (sec)")
    g <- g + theme(legend.title = element_blank())
    g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
    g <- g + theme(axis.text.x = element_text(size=12))
    g <- g + theme(axis.text.y = element_text(size=12))
    g <- g + theme(axis.title.x = element_text(size=18))
    g <- g + theme(axis.title.y = element_text(size=18))
    ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_benchmark_time_modesum <- function(){
    files <- list.files("benchmarks", pattern="^modesum")
    sec <- sapply(files, function(x){
        read.table(paste0("benchmarks/", x), stringsAsFactors=FALSE)[2,1]})
    sec <- as.numeric(sec)
    elements <- gsub("modesum_.*tensor_", "", gsub(".txt", "", files))
    method <- gsub("modesum_", "", gsub("_1E.*txt", "", files))
    method <- factor(method,
        level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, sec, method)
    colnames(data) <- c("elements", "sec", "method")
    data
}

plot_benchmark_time_modesum <- function(outfile){
    data <- aggregate_benchmark_time_modesum()
    g <- ggplot(data, aes(x = elements, y = sec, fill = method))
    g <- g + geom_bar(stat = "identity", position = "dodge")
    g <- g + ggtitle("modeSum()")
    g <- g + xlab("# Elements")
    g <- g + ylab("Calculation Time (sec)")
    g <- g + theme(legend.title = element_blank())
    g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
    g <- g + theme(axis.text.x = element_text(size=12))
    g <- g + theme(axis.text.y = element_text(size=12))
    g <- g + theme(axis.title.x = element_text(size=18))
    g <- g + theme(axis.title.y = element_text(size=18))
    ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_benchmark_time_innerprod <- function(){
    files <- list.files("benchmarks", pattern="^innerprod")
    sec <- sapply(files, function(x){
        read.table(paste0("benchmarks/", x), stringsAsFactors=FALSE)[2,1]})
    sec <- as.numeric(sec)
    elements <- gsub("innerprod_.*tensor_", "", gsub(".txt", "", files))
    method <- gsub("innerprod_", "", gsub("_1E.*txt", "", files))
    method <- factor(method,
        level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, sec, method)
    colnames(data) <- c("elements", "sec", "method")
    data
}

plot_benchmark_time_innerprod <- function(outfile){
    data <- aggregate_benchmark_time_innerprod()
    g <- ggplot(data, aes(x = elements, y = sec, fill = method))
    g <- g + geom_bar(stat = "identity", position = "dodge")
    g <- g + ggtitle("innerProd()")
    g <- g + xlab("# Elements")
    g <- g + ylab("Calculation Time (sec)")
    g <- g + theme(legend.title = element_blank())
    g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
    g <- g + theme(axis.text.x = element_text(size=12))
    g <- g + theme(axis.text.y = element_text(size=12))
    g <- g + theme(axis.title.x = element_text(size=18))
    g <- g + theme(axis.title.y = element_text(size=18))
    ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_benchmark_time_vec <- function(){
    files <- list.files("benchmarks", pattern="^vec")
    sec <- sapply(files, function(x){
        read.table(paste0("benchmarks/", x), stringsAsFactors=FALSE)[2,1]})
    sec <- as.numeric(sec)
    elements <- gsub("vec_.*tensor_", "", gsub(".txt", "", files))
    method <- gsub("vec_", "", gsub("_1E.*txt", "", files))
    method <- factor(method,
        level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, sec, method)
    colnames(data) <- c("elements", "sec", "method")
    data
}

plot_benchmark_time_vec <- function(outfile){
    data <- aggregate_benchmark_time_vec()
    g <- ggplot(data, aes(x = elements, y = sec, fill = method))
    g <- g + geom_bar(stat = "identity", position = "dodge")
    g <- g + ggtitle("vec()")
    g <- g + xlab("# Elements")
    g <- g + ylab("Calculation Time (sec)")
    g <- g + theme(legend.title = element_blank())
    g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
    g <- g + theme(axis.text.x = element_text(size=12))
    g <- g + theme(axis.text.y = element_text(size=12))
    g <- g + theme(axis.title.x = element_text(size=18))
    g <- g + theme(axis.title.y = element_text(size=18))
    ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_benchmark_time_kronecker <- function(){
    files <- list.files("benchmarks", pattern="^kronecker")
    sec <- sapply(files, function(x){
        read.table(paste0("benchmarks/", x), stringsAsFactors=FALSE)[2,1]})
    sec <- as.numeric(sec)
    elements <- gsub("kronecker_.*tensor_", "", gsub(".txt", "", files))
    method <- gsub("kronecker_", "", gsub("_1E.*txt", "", files))
    method <- factor(method,
        level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, sec, method)
    colnames(data) <- c("elements", "sec", "method")
    data
}

plot_benchmark_time_kronecker <- function(outfile){
    data <- aggregate_benchmark_time_kronecker()
    g <- ggplot(data, aes(x = elements, y = sec, fill = method))
    g <- g + geom_bar(stat = "identity", position = "dodge")
    g <- g + ggtitle("kronecker()")
    g <- g + xlab("# Elements")
    g <- g + ylab("Calculation Time (sec)")
    g <- g + theme(legend.title = element_blank())
    g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
    g <- g + theme(axis.text.x = element_text(size=12))
    g <- g + theme(axis.text.y = element_text(size=12))
    g <- g + theme(axis.title.x = element_text(size=18))
    g <- g + theme(axis.title.y = element_text(size=18))
    ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_benchmark_time_fold <- function(){
    files <- list.files("benchmarks", pattern="^fold")
    sec <- sapply(files, function(x){
        read.table(paste0("benchmarks/", x), stringsAsFactors=FALSE)[2,1]})
    sec <- as.numeric(sec)
    elements <- gsub("fold_.*tensor_", "", gsub(".txt", "", files))
    method <- gsub("fold_", "", gsub("_1E.*txt", "", files))
    method <- factor(method,
        level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, sec, method)
    colnames(data) <- c("elements", "sec", "method")
    data
}

plot_benchmark_time_fold <- function(outfile){
    data <- aggregate_benchmark_time_fold()
    g <- ggplot(data, aes(x = elements, y = sec, fill = method))
    g <- g + geom_bar(stat = "identity", position = "dodge")
    g <- g + ggtitle("fold()")
    g <- g + xlab("# Elements")
    g <- g + ylab("Calculation Time (sec)")
    g <- g + theme(legend.title = element_blank())
    g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
    g <- g + theme(axis.text.x = element_text(size=12))
    g <- g + theme(axis.text.y = element_text(size=12))
    g <- g + theme(axis.title.x = element_text(size=18))
    g <- g + theme(axis.title.y = element_text(size=18))
    ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_benchmark_time_hosvd <- function(){
    files <- list.files("benchmarks", pattern="^hosvd")
    sec <- sapply(files, function(x){
        read.table(paste0("benchmarks/", x), stringsAsFactors=FALSE)[2,1]})
    sec <- as.numeric(sec)
    elements <- gsub("hosvd_.*tensor_", "", gsub(".txt", "", files))
    method <- gsub("hosvd_", "", gsub("_1E.*txt", "", files))
    method <- factor(method,
        level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, sec, method)
    colnames(data) <- c("elements", "sec", "method")
    data
}

plot_benchmark_time_hosvd <- function(outfile){
    data <- aggregate_benchmark_time_hosvd()
    g <- ggplot(data, aes(x = elements, y = sec, fill = method))
    g <- g + geom_bar(stat = "identity", position = "dodge")
    g <- g + ggtitle("hosvd()")
    g <- g + xlab("# Elements")
    g <- g + ylab("Calculation Time (sec)")
    g <- g + theme(legend.title = element_blank())
    g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
    g <- g + theme(axis.text.x = element_text(size=12))
    g <- g + theme(axis.text.y = element_text(size=12))
    g <- g + theme(axis.title.x = element_text(size=18))
    g <- g + theme(axis.title.y = element_text(size=18))
    ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_benchmark_time_cp <- function(){
    files <- list.files("benchmarks", pattern="^cp")
    sec <- sapply(files, function(x){
        read.table(paste0("benchmarks/", x), stringsAsFactors=FALSE)[2,1]})
    sec <- as.numeric(sec)
    elements <- gsub("cp_.*tensor_", "", gsub(".txt", "", files))
    method <- gsub("cp_", "", gsub("_1E.*txt", "", files))
    method <- factor(method,
        level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, sec, method)
    colnames(data) <- c("elements", "sec", "method")
    data
}

plot_benchmark_time_cp <- function(outfile){
    data <- aggregate_benchmark_time_cp()
    g <- ggplot(data, aes(x = elements, y = sec, fill = method))
    g <- g + geom_bar(stat = "identity", position = "dodge")
    g <- g + ggtitle("cp()")
    g <- g + xlab("# Elements")
    g <- g + ylab("Calculation Time (sec)")
    g <- g + theme(legend.title = element_blank())
    g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
    g <- g + theme(axis.text.x = element_text(size=12))
    g <- g + theme(axis.text.y = element_text(size=12))
    g <- g + theme(axis.title.x = element_text(size=18))
    g <- g + theme(axis.title.y = element_text(size=18))
    ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_benchmark_time_tucker <- function(){
    files <- list.files("benchmarks", pattern="^tucker")
    sec <- sapply(files, function(x){
        read.table(paste0("benchmarks/", x), stringsAsFactors=FALSE)[2,1]})
    sec <- as.numeric(sec)
    elements <- gsub("tucker_.*tensor_", "", gsub(".txt", "", files))
    method <- gsub("tucker_", "", gsub("_1E.*txt", "", files))
    method <- factor(method,
        level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, sec, method)
    colnames(data) <- c("elements", "sec", "method")
    data
}

plot_benchmark_time_tucker <- function(outfile){
    data <- aggregate_benchmark_time_tucker()
    g <- ggplot(data, aes(x = elements, y = sec, fill = method))
    g <- g + geom_bar(stat = "identity", position = "dodge")
    g <- g + ggtitle("tucker()")
    g <- g + xlab("# Elements")
    g <- g + ylab("Calculation Time (sec)")
    g <- g + theme(legend.title = element_blank())
    g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
    g <- g + theme(axis.text.x = element_text(size=12))
    g <- g + theme(axis.text.y = element_text(size=12))
    g <- g + theme(axis.title.x = element_text(size=18))
    g <- g + theme(axis.title.y = element_text(size=18))
    ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_benchmark_time_mpca <- function(){
    files <- list.files("benchmarks", pattern="^mpca")
    sec <- sapply(files, function(x){
        read.table(paste0("benchmarks/", x), stringsAsFactors=FALSE)[2,1]})
    sec <- as.numeric(sec)
    elements <- gsub("mpca_.*tensor_", "", gsub(".txt", "", files))
    method <- gsub("mpca_", "", gsub("_1E.*txt", "", files))
    method <- factor(method,
        level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, sec, method)
    colnames(data) <- c("elements", "sec", "method")
    data
}

plot_benchmark_time_mpca <- function(outfile){
    data <- aggregate_benchmark_time_mpca()
    g <- ggplot(data, aes(x = elements, y = sec, fill = method))
    g <- g + geom_bar(stat = "identity", position = "dodge")
    g <- g + ggtitle("mpca()")
    g <- g + xlab("# Elements")
    g <- g + ylab("Calculation Time (sec)")
    g <- g + theme(legend.title = element_blank())
    g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
    g <- g + theme(axis.text.x = element_text(size=12))
    g <- g + theme(axis.text.y = element_text(size=12))
    g <- g + theme(axis.title.x = element_text(size=18))
    g <- g + theme(axis.title.y = element_text(size=18))
    ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_benchmark_time_pvd <- function(){
    files <- list.files("benchmarks", pattern="^pvd")
    sec <- sapply(files, function(x){
        read.table(paste0("benchmarks/", x), stringsAsFactors=FALSE)[2,1]})
    sec <- as.numeric(sec)
    elements <- gsub("pvd_.*tensor_", "", gsub(".txt", "", files))
    method <- gsub("pvd_", "", gsub("_1E.*txt", "", files))
    method <- factor(method,
        level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, sec, method)
    colnames(data) <- c("elements", "sec", "method")
    data
}

plot_benchmark_time_pvd <- function(outfile){
    data <- aggregate_benchmark_time_pvd()
    g <- ggplot(data, aes(x = elements, y = sec, fill = method))
    g <- g + geom_bar(stat = "identity", position = "dodge")
    g <- g + ggtitle("pvd()")
    g <- g + xlab("# Elements")
    g <- g + ylab("Calculation Time (sec)")
    g <- g + theme(legend.title = element_blank())
    g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
    g <- g + theme(axis.text.x = element_text(size=12))
    g <- g + theme(axis.text.y = element_text(size=12))
    g <- g + theme(axis.title.x = element_text(size=18))
    g <- g + theme(axis.title.y = element_text(size=18))
    ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_benchmark_time_einsum <- function(){
    files <- list.files("benchmarks", pattern="^einsum")
    sec <- sapply(files, function(x){
        read.table(paste0("benchmarks/", x), stringsAsFactors=FALSE)[2,1]})
    sec <- as.numeric(sec)
    elements <- gsub("einsum_.*tensor_", "", gsub(".txt", "", files))
    method <- gsub("einsum_", "", gsub("_1E.*txt", "", files))
    method <- factor(method,
        level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, sec, method)
    colnames(data) <- c("elements", "sec", "method")
    data
}

plot_benchmark_time_einsum <- function(outfile){
    data <- aggregate_benchmark_time_einsum()
    g <- ggplot(data, aes(x = elements, y = sec, fill = method))
    g <- g + geom_bar(stat = "identity", position = "dodge")
    g <- g + ggtitle("einsum()")
    g <- g + xlab("# Elements")
    g <- g + ylab("Calculation Time (sec)")
    g <- g + theme(legend.title = element_blank())
    g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
    g <- g + theme(axis.text.x = element_text(size=12))
    g <- g + theme(axis.text.y = element_text(size=12))
    g <- g + theme(axis.title.x = element_text(size=18))
    g <- g + theme(axis.title.y = element_text(size=18))
    ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

# gnutime
plot_gnutime_time <- function(arithmetic, outfile){
    cmd <- paste0("plot_gnutime_time_", arithmetic, "(outfile)")
    eval(parse(text=cmd))
}

aggregate_gnutime_time_unfold <- function(){
    files <- list.files("logs", pattern="^unfold")
    sec <- sapply(files, function(x){
        tmp <- read.delim(paste0("logs/", x), stringsAsFactors=FALSE)[,1]
        tmp <- tmp[grep("User time", tmp)]
        tmp <- gsub("User time \\(seconds\\): ", "", tmp)
        })
    sec <- as.numeric(sec)
    elements <- gsub("unfold_.*tensor_", "", gsub(".log", "", files))
    method <- gsub("unfold_", "", gsub("_1E.*log", "", files))
    method <- factor(method,
        level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, sec, method)
    colnames(data) <- c("elements", "sec", "method")
    data
}

plot_gnutime_time_unfold <- function(outfile){
    data <- aggregate_gnutime_time_unfold()
    g <- ggplot(data, aes(x = elements, y = sec, fill = method))
    g <- g + geom_bar(stat = "identity", position = "dodge")
    g <- g + ggtitle("unfold()")
    g <- g + xlab("# Elements")
    g <- g + ylab("Calculation Time (sec)")
    g <- g + theme(legend.title = element_blank())
    g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
    g <- g + theme(axis.text.x = element_text(size=12))
    g <- g + theme(axis.text.y = element_text(size=12))
    g <- g + theme(axis.title.x = element_text(size=18))
    g <- g + theme(axis.title.y = element_text(size=18))
    ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_gnutime_time_modesum <- function(){
    files <- list.files("logs", pattern="^modesum")
    sec <- sapply(files, function(x){
        tmp <- read.delim(paste0("logs/", x), stringsAsFactors=FALSE)[,1]
        tmp <- tmp[grep("User time", tmp)]
        tmp <- gsub("User time \\(seconds\\): ", "", tmp)
        })
    sec <- as.numeric(sec)
    elements <- gsub("modesum_.*tensor_", "", gsub(".log", "", files))
    method <- gsub("modesum_", "", gsub("_1E.*log", "", files))
    method <- factor(method,
        level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, sec, method)
    colnames(data) <- c("elements", "sec", "method")
    data
}

plot_gnutime_time_modesum <- function(outfile){
    data <- aggregate_gnutime_time_modesum()
    g <- ggplot(data, aes(x = elements, y = sec, fill = method))
    g <- g + geom_bar(stat = "identity", position = "dodge")
    g <- g + ggtitle("modeSum()")
    g <- g + xlab("# Elements")
    g <- g + ylab("Calculation Time (sec)")
    g <- g + theme(legend.title = element_blank())
    g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
    g <- g + theme(axis.text.x = element_text(size=12))
    g <- g + theme(axis.text.y = element_text(size=12))
    g <- g + theme(axis.title.x = element_text(size=18))
    g <- g + theme(axis.title.y = element_text(size=18))
    ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_gnutime_time_innerprod <- function(){
    files <- list.files("logs", pattern="^innerprod")
    sec <- sapply(files, function(x){
        tmp <- read.delim(paste0("logs/", x), stringsAsFactors=FALSE)[,1]
        tmp <- tmp[grep("User time", tmp)]
        tmp <- gsub("User time \\(seconds\\): ", "", tmp)
        })
    sec <- as.numeric(sec)
    elements <- gsub("innerprod_.*tensor_", "", gsub(".log", "", files))
    method <- gsub("innerprod_", "", gsub("_1E.*log", "", files))
    method <- factor(method,
        level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, sec, method)
    colnames(data) <- c("elements", "sec", "method")
    data
}

plot_gnutime_time_innerprod <- function(outfile){
    data <- aggregate_gnutime_time_innerprod()
    g <- ggplot(data, aes(x = elements, y = sec, fill = method))
    g <- g + geom_bar(stat = "identity", position = "dodge")
    g <- g + ggtitle("innerProd()")
    g <- g + xlab("# Elements")
    g <- g + ylab("Calculation Time (sec)")
    g <- g + theme(legend.title = element_blank())
    g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
    g <- g + theme(axis.text.x = element_text(size=12))
    g <- g + theme(axis.text.y = element_text(size=12))
    g <- g + theme(axis.title.x = element_text(size=18))
    g <- g + theme(axis.title.y = element_text(size=18))
    ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_gnutime_time_vec <- function(){
    files <- list.files("logs", pattern="^vec")
    sec <- sapply(files, function(x){
        tmp <- read.delim(paste0("logs/", x), stringsAsFactors=FALSE)[,1]
        tmp <- tmp[grep("User time", tmp)]
        tmp <- gsub("User time \\(seconds\\): ", "", tmp)
        })
    sec <- as.numeric(sec)
    elements <- gsub("vec_.*tensor_", "", gsub(".log", "", files))
    method <- gsub("vec_", "", gsub("_1E.*log", "", files))
    method <- factor(method,
        level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, sec, method)
    colnames(data) <- c("elements", "sec", "method")
    data
}

plot_gnutime_time_vec <- function(outfile){
    data <- aggregate_gnutime_time_vec()
    g <- ggplot(data, aes(x = elements, y = sec, fill = method))
    g <- g + geom_bar(stat = "identity", position = "dodge")
    g <- g + ggtitle("vec()")
    g <- g + xlab("# Elements")
    g <- g + ylab("Calculation Time (sec)")
    g <- g + theme(legend.title = element_blank())
    g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
    g <- g + theme(axis.text.x = element_text(size=12))
    g <- g + theme(axis.text.y = element_text(size=12))
    g <- g + theme(axis.title.x = element_text(size=18))
    g <- g + theme(axis.title.y = element_text(size=18))
    ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_gnutime_time_kronecker <- function(){
    files <- list.files("logs", pattern="^kronecker")
    sec <- sapply(files, function(x){
        tmp <- read.delim(paste0("logs/", x), stringsAsFactors=FALSE)[,1]
        tmp <- tmp[grep("User time", tmp)]
        tmp <- gsub("User time \\(seconds\\): ", "", tmp)
        })
    sec <- as.numeric(sec)
    elements <- gsub("kronecker_.*tensor_", "", gsub(".log", "", files))
    method <- gsub("kronecker_", "", gsub("_1E.*log", "", files))
    method <- factor(method,
        level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, sec, method)
    colnames(data) <- c("elements", "sec", "method")
    data
}

plot_gnutime_time_kronecker <- function(outfile){
    data <- aggregate_gnutime_time_kronecker()
    g <- ggplot(data, aes(x = elements, y = sec, fill = method))
    g <- g + geom_bar(stat = "identity", position = "dodge")
    g <- g + ggtitle("kronecker()")
    g <- g + xlab("# Elements")
    g <- g + ylab("Calculation Time (sec)")
    g <- g + theme(legend.title = element_blank())
    g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
    g <- g + theme(axis.text.x = element_text(size=12))
    g <- g + theme(axis.text.y = element_text(size=12))
    g <- g + theme(axis.title.x = element_text(size=18))
    g <- g + theme(axis.title.y = element_text(size=18))
    ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_gnutime_time_fold <- function(){
    files <- list.files("logs", pattern="^fold")
    sec <- sapply(files, function(x){
        tmp <- read.delim(paste0("logs/", x), stringsAsFactors=FALSE)[,1]
        tmp <- tmp[grep("User time", tmp)]
        tmp <- gsub("User time \\(seconds\\): ", "", tmp)
        })
    sec <- as.numeric(sec)
    elements <- gsub("fold_.*tensor_", "", gsub(".log", "", files))
    method <- gsub("fold_", "", gsub("_1E.*log", "", files))
    method <- factor(method,
        level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, sec, method)
    colnames(data) <- c("elements", "sec", "method")
    data
}

plot_gnutime_time_fold <- function(outfile){
    data <- aggregate_gnutime_time_fold()
    g <- ggplot(data, aes(x = elements, y = sec, fill = method))
    g <- g + geom_bar(stat = "identity", position = "dodge")
    g <- g + ggtitle("fold()")
    g <- g + xlab("# Elements")
    g <- g + ylab("Calculation Time (sec)")
    g <- g + theme(legend.title = element_blank())
    g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
    g <- g + theme(axis.text.x = element_text(size=12))
    g <- g + theme(axis.text.y = element_text(size=12))
    g <- g + theme(axis.title.x = element_text(size=18))
    g <- g + theme(axis.title.y = element_text(size=18))
    ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_gnutime_time_hosvd <- function(){
    files <- list.files("logs", pattern="^hosvd")
    sec <- sapply(files, function(x){
        tmp <- read.delim(paste0("logs/", x), stringsAsFactors=FALSE)[,1]
        tmp <- tmp[grep("User time", tmp)]
        tmp <- gsub("User time \\(seconds\\): ", "", tmp)
        })
    sec <- as.numeric(sec)
    elements <- gsub("hosvd_.*tensor_", "", gsub(".log", "", files))
    method <- gsub("hosvd_", "", gsub("_1E.*log", "", files))
    method <- factor(method,
        level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, sec, method)
    colnames(data) <- c("elements", "sec", "method")
    data
}

plot_gnutime_time_hosvd <- function(outfile){
    data <- aggregate_gnutime_time_hosvd()
    g <- ggplot(data, aes(x = elements, y = sec, fill = method))
    g <- g + geom_bar(stat = "identity", position = "dodge")
    g <- g + ggtitle("hosvd()")
    g <- g + xlab("# Elements")
    g <- g + ylab("Calculation Time (sec)")
    g <- g + theme(legend.title = element_blank())
    g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
    g <- g + theme(axis.text.x = element_text(size=12))
    g <- g + theme(axis.text.y = element_text(size=12))
    g <- g + theme(axis.title.x = element_text(size=18))
    g <- g + theme(axis.title.y = element_text(size=18))
    ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_gnutime_time_cp <- function(){
    files <- list.files("logs", pattern="^cp")
    sec <- sapply(files, function(x){
        tmp <- read.delim(paste0("logs/", x), stringsAsFactors=FALSE)[,1]
        tmp <- tmp[grep("User time", tmp)]
        tmp <- gsub("User time \\(seconds\\): ", "", tmp)
        })
    sec <- as.numeric(sec)
    elements <- gsub("cp_.*tensor_", "", gsub(".log", "", files))
    method <- gsub("cp_", "", gsub("_1E.*log", "", files))
    method <- factor(method,
        level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, sec, method)
    colnames(data) <- c("elements", "sec", "method")
    data
}

plot_gnutime_time_cp <- function(outfile){
    data <- aggregate_gnutime_time_cp()
    g <- ggplot(data, aes(x = elements, y = sec, fill = method))
    g <- g + geom_bar(stat = "identity", position = "dodge")
    g <- g + ggtitle("cp()")
    g <- g + xlab("# Elements")
    g <- g + ylab("Calculation Time (sec)")
    g <- g + theme(legend.title = element_blank())
    g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
    g <- g + theme(axis.text.x = element_text(size=12))
    g <- g + theme(axis.text.y = element_text(size=12))
    g <- g + theme(axis.title.x = element_text(size=18))
    g <- g + theme(axis.title.y = element_text(size=18))
    ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_gnutime_time_tucker <- function(){
    files <- list.files("logs", pattern="^tucker")
    sec <- sapply(files, function(x){
        tmp <- read.delim(paste0("logs/", x), stringsAsFactors=FALSE)[,1]
        tmp <- tmp[grep("User time", tmp)]
        tmp <- gsub("User time \\(seconds\\): ", "", tmp)
        })
    sec <- as.numeric(sec)
    elements <- gsub("tucker_.*tensor_", "", gsub(".log", "", files))
    method <- gsub("tucker_", "", gsub("_1E.*log", "", files))
    method <- factor(method,
        level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, sec, method)
    colnames(data) <- c("elements", "sec", "method")
    data
}

plot_gnutime_time_tucker <- function(outfile){
    data <- aggregate_gnutime_time_tucker()
    g <- ggplot(data, aes(x = elements, y = sec, fill = method))
    g <- g + geom_bar(stat = "identity", position = "dodge")
    g <- g + ggtitle("tucker()")
    g <- g + xlab("# Elements")
    g <- g + ylab("Calculation Time (sec)")
    g <- g + theme(legend.title = element_blank())
    g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
    g <- g + theme(axis.text.x = element_text(size=12))
    g <- g + theme(axis.text.y = element_text(size=12))
    g <- g + theme(axis.title.x = element_text(size=18))
    g <- g + theme(axis.title.y = element_text(size=18))
    ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_gnutime_time_mpca <- function(){
    files <- list.files("logs", pattern="^mpca")
    sec <- sapply(files, function(x){
        tmp <- read.delim(paste0("logs/", x), stringsAsFactors=FALSE)[,1]
        tmp <- tmp[grep("User time", tmp)]
        tmp <- gsub("User time \\(seconds\\): ", "", tmp)
        })
    sec <- as.numeric(sec)
    elements <- gsub("mpca_.*tensor_", "", gsub(".log", "", files))
    method <- gsub("mpca_", "", gsub("_1E.*log", "", files))
    method <- factor(method,
        level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, sec, method)
    colnames(data) <- c("elements", "sec", "method")
    data
}

plot_gnutime_time_mpca <- function(outfile){
    data <- aggregate_gnutime_time_mpca()
    g <- ggplot(data, aes(x = elements, y = sec, fill = method))
    g <- g + geom_bar(stat = "identity", position = "dodge")
    g <- g + ggtitle("mpca()")
    g <- g + xlab("# Elements")
    g <- g + ylab("Calculation Time (sec)")
    g <- g + theme(legend.title = element_blank())
    g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
    g <- g + theme(axis.text.x = element_text(size=12))
    g <- g + theme(axis.text.y = element_text(size=12))
    g <- g + theme(axis.title.x = element_text(size=18))
    g <- g + theme(axis.title.y = element_text(size=18))
    ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_gnutime_time_pvd <- function(){
    files <- list.files("logs", pattern="^pvd")
    sec <- sapply(files, function(x){
        tmp <- read.delim(paste0("logs/", x), stringsAsFactors=FALSE)[,1]
        tmp <- tmp[grep("User time", tmp)]
        tmp <- gsub("User time \\(seconds\\): ", "", tmp)
        })
    sec <- as.numeric(sec)
    elements <- gsub("pvd_.*tensor_", "", gsub(".log", "", files))
    method <- gsub("pvd_", "", gsub("_1E.*log", "", files))
    method <- factor(method,
        level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, sec, method)
    colnames(data) <- c("elements", "sec", "method")
    data
}

plot_gnutime_time_pvd <- function(outfile){
    data <- aggregate_gnutime_time_pvd()
    g <- ggplot(data, aes(x = elements, y = sec, fill = method))
    g <- g + geom_bar(stat = "identity", position = "dodge")
    g <- g + ggtitle("pvd()")
    g <- g + xlab("# Elements")
    g <- g + ylab("Calculation Time (sec)")
    g <- g + theme(legend.title = element_blank())
    g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
    g <- g + theme(axis.text.x = element_text(size=12))
    g <- g + theme(axis.text.y = element_text(size=12))
    g <- g + theme(axis.title.x = element_text(size=18))
    g <- g + theme(axis.title.y = element_text(size=18))
    ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_gnutime_time_einsum <- function(){
    files <- list.files("logs", pattern="^einsum")
    sec <- sapply(files, function(x){
        tmp <- read.delim(paste0("logs/", x), stringsAsFactors=FALSE)[,1]
        tmp <- tmp[grep("User time", tmp)]
        tmp <- gsub("User time \\(seconds\\): ", "", tmp)
        })
    sec <- as.numeric(sec)
    elements <- gsub("einsum_.*tensor_", "", gsub(".log", "", files))
    method <- gsub("einsum_", "", gsub("_1E.*log", "", files))
    method <- factor(method,
        level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, sec, method)
    colnames(data) <- c("elements", "sec", "method")
    data
}

plot_gnutime_time_einsum <- function(outfile){
    data <- aggregate_gnutime_time_einsum()
    g <- ggplot(data, aes(x = elements, y = sec, fill = method))
    g <- g + geom_bar(stat = "identity", position = "dodge")
    g <- g + ggtitle("einsum()")
    g <- g + xlab("# Elements")
    g <- g + ylab("Calculation Time (sec)")
    g <- g + theme(legend.title = element_blank())
    g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
    g <- g + theme(axis.text.x = element_text(size=12))
    g <- g + theme(axis.text.y = element_text(size=12))
    g <- g + theme(axis.title.x = element_text(size=18))
    g <- g + theme(axis.title.y = element_text(size=18))
    ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

#######################################################
# All Functions for Visualization (Memory usage)
#######################################################
# profmem
plot_profmem_memory <- function(arithmetic, outfile){
	cmd <- paste0("plot_profmem_memory_", arithmetic, "(outfile)")
	eval(parse(text=cmd))
}

aggregate_profmem_memory_unfold <- function(){
    files <- list.files("profmem", pattern="^unfold")
    gb <- sapply(files, function(x){
        load(paste0("profmem/", x))
        out[[1]] / 10^3})
    elements <- gsub("unfold_.*tensor_", "", gsub(".RData", "", files))
    method <- gsub("unfold_", "", gsub("_1E.*RData", "", files))
    method <- factor(method,
    	level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, gb, method)
    colnames(data) <- c("elements", "gb", "method")
    data
}

plot_profmem_memory_unfold <- function(outfile){
	data <- aggregate_profmem_memory_unfold()
	g <- ggplot(data, aes(x = elements, y = gb, fill = method))
	g <- g + geom_bar(stat = "identity", position = "dodge")
	g <- g + ggtitle("unfold()")
	g <- g + xlab("# Elements")
	g <- g + ylab("Memory usage (GB)")
	g <- g + theme(legend.title = element_blank())
	g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
	g <- g + theme(axis.text.x = element_text(size=12))
	g <- g + theme(axis.text.y = element_text(size=12))
	g <- g + theme(axis.title.x = element_text(size=18))
	g <- g + theme(axis.title.y = element_text(size=18))
	ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_profmem_memory_modesum <- function(){
    files <- list.files("profmem", pattern="^modesum")
    gb <- sapply(files, function(x){
        load(paste0("profmem/", x))
        out[[1]] / 10^3})
    elements <- gsub("modesum_.*tensor_", "", gsub(".RData", "", files))
    method <- gsub("modesum_", "", gsub("_1E.*RData", "", files))
    method <- factor(method,
    	level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, gb, method)
    colnames(data) <- c("elements", "gb", "method")
    data
}

plot_profmem_memory_modesum <- function(outfile){
	data <- aggregate_profmem_memory_modesum()
	g <- ggplot(data, aes(x = elements, y = gb, fill = method))
	g <- g + geom_bar(stat = "identity", position = "dodge")
	g <- g + ggtitle("modeSum()")
	g <- g + xlab("# Elements")
	g <- g + ylab("Memory usage (GB)")
	g <- g + theme(legend.title = element_blank())
	g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
	g <- g + theme(axis.text.x = element_text(size=12))
	g <- g + theme(axis.text.y = element_text(size=12))
	g <- g + theme(axis.title.x = element_text(size=18))
	g <- g + theme(axis.title.y = element_text(size=18))
	ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_profmem_memory_innerprod <- function(){
    files <- list.files("profmem", pattern="^innerprod")
    gb <- sapply(files, function(x){
        load(paste0("profmem/", x))
        out[[1]] / 10^3})
    elements <- gsub("innerprod_.*tensor_", "", gsub(".RData", "", files))
    method <- gsub("innerprod_", "", gsub("_1E.*RData", "", files))
    method <- factor(method,
    	level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, gb, method)
    colnames(data) <- c("elements", "gb", "method")
    data
}

plot_profmem_memory_innerprod <- function(outfile){
	data <- aggregate_profmem_memory_innerprod()
	g <- ggplot(data, aes(x = elements, y = gb, fill = method))
	g <- g + geom_bar(stat = "identity", position = "dodge")
	g <- g + ggtitle("innerProd()")
	g <- g + xlab("# Elements")
	g <- g + ylab("Memory usage (GB)")
	g <- g + theme(legend.title = element_blank())
	g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
	g <- g + theme(axis.text.x = element_text(size=12))
	g <- g + theme(axis.text.y = element_text(size=12))
	g <- g + theme(axis.title.x = element_text(size=18))
	g <- g + theme(axis.title.y = element_text(size=18))
	ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_profmem_memory_vec <- function(){
    files <- list.files("profmem", pattern="^vec")
    gb <- sapply(files, function(x){
        load(paste0("profmem/", x))
        out[[1]] / 10^3})
    elements <- gsub("vec_.*tensor_", "", gsub(".RData", "", files))
    method <- gsub("vec_", "", gsub("_1E.*RData", "", files))
    method <- factor(method,
    	level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, gb, method)
    colnames(data) <- c("elements", "gb", "method")
    data
}

plot_profmem_memory_vec <- function(outfile){
	data <- aggregate_profmem_memory_vec()
	g <- ggplot(data, aes(x = elements, y = gb, fill = method))
	g <- g + geom_bar(stat = "identity", position = "dodge")
	g <- g + ggtitle("vec()")
	g <- g + xlab("# Elements")
	g <- g + ylab("Memory usage (GB)")
	g <- g + theme(legend.title = element_blank())
	g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
	g <- g + theme(axis.text.x = element_text(size=12))
	g <- g + theme(axis.text.y = element_text(size=12))
	g <- g + theme(axis.title.x = element_text(size=18))
	g <- g + theme(axis.title.y = element_text(size=18))
	ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_profmem_memory_kronecker <- function(){
    files <- list.files("profmem", pattern="^kronecker")
    gb <- sapply(files, function(x){
        load(paste0("profmem/", x))
        out[[1]] / 10^3})
    elements <- gsub("kronecker_.*tensor_", "", gsub(".RData", "", files))
    method <- gsub("kronecker_", "", gsub("_1E.*RData", "", files))
    method <- factor(method,
    	level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, gb, method)
    colnames(data) <- c("elements", "gb", "method")
    data
}

plot_profmem_memory_kronecker <- function(outfile){
	data <- aggregate_profmem_memory_kronecker()
	g <- ggplot(data, aes(x = elements, y = gb, fill = method))
	g <- g + geom_bar(stat = "identity", position = "dodge")
	g <- g + ggtitle("kronecker()")
	g <- g + xlab("# Elements")
	g <- g + ylab("Memory usage (GB)")
	g <- g + theme(legend.title = element_blank())
	g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
	g <- g + theme(axis.text.x = element_text(size=12))
	g <- g + theme(axis.text.y = element_text(size=12))
	g <- g + theme(axis.title.x = element_text(size=18))
	g <- g + theme(axis.title.y = element_text(size=18))
	ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_profmem_memory_fold <- function(){
    files <- list.files("profmem", pattern="^fold")
    gb <- sapply(files, function(x){
        load(paste0("profmem/", x))
        out[[1]] / 10^3})
    elements <- gsub("fold_.*tensor_", "", gsub(".RData", "", files))
    method <- gsub("fold_", "", gsub("_1E.*RData", "", files))
    method <- factor(method,
    	level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, gb, method)
    colnames(data) <- c("elements", "gb", "method")
    data
}

plot_profmem_memory_fold <- function(outfile){
	data <- aggregate_profmem_memory_fold()
	g <- ggplot(data, aes(x = elements, y = gb, fill = method))
	g <- g + geom_bar(stat = "identity", position = "dodge")
	g <- g + ggtitle("fold()")
	g <- g + xlab("# Elements")
	g <- g + ylab("Memory usage (GB)")
	g <- g + theme(legend.title = element_blank())
	g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
	g <- g + theme(axis.text.x = element_text(size=12))
	g <- g + theme(axis.text.y = element_text(size=12))
	g <- g + theme(axis.title.x = element_text(size=18))
	g <- g + theme(axis.title.y = element_text(size=18))
	ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_profmem_memory_hosvd <- function(){
    files <- list.files("profmem", pattern="^hosvd")
    gb <- sapply(files, function(x){
        load(paste0("profmem/", x))
        out[[1]] / 10^3})
    elements <- gsub("hosvd_.*tensor_", "", gsub(".RData", "", files))
    method <- gsub("hosvd_", "", gsub("_1E.*RData", "", files))
    method <- factor(method,
    	level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, gb, method)
    colnames(data) <- c("elements", "gb", "method")
    data
}

plot_profmem_memory_hosvd <- function(outfile){
	data <- aggregate_profmem_memory_hosvd()
	g <- ggplot(data, aes(x = elements, y = gb, fill = method))
	g <- g + geom_bar(stat = "identity", position = "dodge")
	g <- g + ggtitle("hosvd()")
	g <- g + xlab("# Elements")
	g <- g + ylab("Memory usage (GB)")
	g <- g + theme(legend.title = element_blank())
	g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
	g <- g + theme(axis.text.x = element_text(size=12))
	g <- g + theme(axis.text.y = element_text(size=12))
	g <- g + theme(axis.title.x = element_text(size=18))
	g <- g + theme(axis.title.y = element_text(size=18))
	ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_profmem_memory_cp <- function(){
    files <- list.files("profmem", pattern="^cp")
    gb <- sapply(files, function(x){
        load(paste0("profmem/", x))
        out[[1]] / 10^3})
    elements <- gsub("cp_.*tensor_", "", gsub(".RData", "", files))
    method <- gsub("cp_", "", gsub("_1E.*RData", "", files))
    method <- factor(method,
    	level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, gb, method)
    colnames(data) <- c("elements", "gb", "method")
    data
}

plot_profmem_memory_cp <- function(outfile){
	data <- aggregate_profmem_memory_cp()
	g <- ggplot(data, aes(x = elements, y = gb, fill = method))
	g <- g + geom_bar(stat = "identity", position = "dodge")
	g <- g + ggtitle("cp()")
	g <- g + xlab("# Elements")
	g <- g + ylab("Memory usage (GB)")
	g <- g + theme(legend.title = element_blank())
	g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
	g <- g + theme(axis.text.x = element_text(size=12))
	g <- g + theme(axis.text.y = element_text(size=12))
	g <- g + theme(axis.title.x = element_text(size=18))
	g <- g + theme(axis.title.y = element_text(size=18))
	ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_profmem_memory_tucker <- function(){
    files <- list.files("profmem", pattern="^tucker")
    gb <- sapply(files, function(x){
        load(paste0("profmem/", x))
        out[[1]] / 10^3})
    elements <- gsub("tucker_.*tensor_", "", gsub(".RData", "", files))
    method <- gsub("tucker_", "", gsub("_1E.*RData", "", files))
    method <- factor(method,
    	level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, gb, method)
    colnames(data) <- c("elements", "gb", "method")
    data
}

plot_profmem_memory_tucker <- function(outfile){
	data <- aggregate_profmem_memory_tucker()
	g <- ggplot(data, aes(x = elements, y = gb, fill = method))
	g <- g + geom_bar(stat = "identity", position = "dodge")
	g <- g + ggtitle("tucker()")
	g <- g + xlab("# Elements")
	g <- g + ylab("Memory usage (GB)")
	g <- g + theme(legend.title = element_blank())
	g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
	g <- g + theme(axis.text.x = element_text(size=12))
	g <- g + theme(axis.text.y = element_text(size=12))
	g <- g + theme(axis.title.x = element_text(size=18))
	g <- g + theme(axis.title.y = element_text(size=18))
	ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_profmem_memory_mpca <- function(){
    files <- list.files("profmem", pattern="^mpca")
    gb <- sapply(files, function(x){
        load(paste0("profmem/", x))
        out[[1]] / 10^3})
    elements <- gsub("mpca_.*tensor_", "", gsub(".RData", "", files))
    method <- gsub("mpca_", "", gsub("_1E.*RData", "", files))
    method <- factor(method,
    	level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, gb, method)
    colnames(data) <- c("elements", "gb", "method")
    data
}

plot_profmem_memory_mpca <- function(outfile){
	data <- aggregate_profmem_memory_mpca()
	g <- ggplot(data, aes(x = elements, y = gb, fill = method))
	g <- g + geom_bar(stat = "identity", position = "dodge")
	g <- g + ggtitle("mpca()")
	g <- g + xlab("# Elements")
	g <- g + ylab("Memory usage (GB)")
	g <- g + theme(legend.title = element_blank())
	g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
	g <- g + theme(axis.text.x = element_text(size=12))
	g <- g + theme(axis.text.y = element_text(size=12))
	g <- g + theme(axis.title.x = element_text(size=18))
	g <- g + theme(axis.title.y = element_text(size=18))
	ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_profmem_memory_pvd <- function(){
    files <- list.files("profmem", pattern="^pvd")
    gb <- sapply(files, function(x){
        load(paste0("profmem/", x))
        out[[1]] / 10^3})
    elements <- gsub("pvd_.*tensor_", "", gsub(".RData", "", files))
    method <- gsub("pvd_", "", gsub("_1E.*RData", "", files))
    method <- factor(method,
    	level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, gb, method)
    colnames(data) <- c("elements", "gb", "method")
    data
}

plot_profmem_memory_pvd <- function(outfile){
	data <- aggregate_profmem_memory_pvd()
	g <- ggplot(data, aes(x = elements, y = gb, fill = method))
	g <- g + geom_bar(stat = "identity", position = "dodge")
	g <- g + ggtitle("pvd()")
	g <- g + xlab("# Elements")
	g <- g + ylab("Memory usage (GB)")
	g <- g + theme(legend.title = element_blank())
	g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
	g <- g + theme(axis.text.x = element_text(size=12))
	g <- g + theme(axis.text.y = element_text(size=12))
	g <- g + theme(axis.title.x = element_text(size=18))
	g <- g + theme(axis.title.y = element_text(size=18))
	ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_profmem_memory_einsum <- function(){
    files <- list.files("profmem", pattern="^einsum")
    gb <- sapply(files, function(x){
        load(paste0("profmem/", x))
        out[[1]] / 10^3})
    elements <- gsub("einsum_.*tensor_", "", gsub(".RData", "", files))
    method <- gsub("einsum_", "", gsub("_1E.*RData", "", files))
    method <- factor(method,
    	level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, gb, method)
    colnames(data) <- c("elements", "gb", "method")
    data
}

plot_profmem_memory_einsum <- function(outfile){
	data <- aggregate_profmem_memory_einsum()
	g <- ggplot(data, aes(x = elements, y = gb, fill = method))
	g <- g + geom_bar(stat = "identity", position = "dodge")
	g <- g + ggtitle("einsum()")
	g <- g + xlab("# Elements")
	g <- g + ylab("Memory usage (GB)")
	g <- g + theme(legend.title = element_blank())
	g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
	g <- g + theme(axis.text.x = element_text(size=12))
	g <- g + theme(axis.text.y = element_text(size=12))
	g <- g + theme(axis.title.x = element_text(size=18))
	g <- g + theme(axis.title.y = element_text(size=18))
	ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

# benchmark
plot_benchmark_memory <- function(arithmetic, outfile){
    cmd <- paste0("plot_benchmark_memory_", arithmetic, "(outfile)")
    eval(parse(text=cmd))
}

aggregate_benchmark_memory_unfold <- function(){
    files <- list.files("benchmarks", pattern="^unfold")
    gb <- sapply(files, function(x){
        read.table(paste0("benchmarks/", x),
            stringsAsFactors=FALSE)[2,3]})
    gb <- as.numeric(gb) / 10^3
    elements <- gsub("unfold_.*tensor_", "", gsub(".txt", "", files))
    method <- gsub("unfold_", "", gsub("_1E.*txt", "", files))
    method <- factor(method,
        level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, gb, method)
    colnames(data) <- c("elements", "gb", "method")
    data
}

plot_benchmark_memory_unfold <- function(outfile){
    data <- aggregate_benchmark_memory_unfold()
    g <- ggplot(data, aes(x = elements, y = gb, fill = method))
    g <- g + geom_bar(stat = "identity", position = "dodge")
    g <- g + ggtitle("unfold()")
    g <- g + xlab("# Elements")
    g <- g + ylab("Memory usage (GB)")
    g <- g + theme(legend.title = element_blank())
    g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
    g <- g + theme(axis.text.x = element_text(size=12))
    g <- g + theme(axis.text.y = element_text(size=12))
    g <- g + theme(axis.title.x = element_text(size=18))
    g <- g + theme(axis.title.y = element_text(size=18))
    ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_benchmark_memory_modesum <- function(){
    files <- list.files("benchmarks", pattern="^modesum")
    gb <- sapply(files, function(x){
        read.table(paste0("benchmarks/", x),
            stringsAsFactors=FALSE)[2,3]})
    gb <- as.numeric(gb) / 10^3
    elements <- gsub("modesum_.*tensor_", "", gsub(".txt", "", files))
    method <- gsub("modesum_", "", gsub("_1E.*txt", "", files))
    method <- factor(method,
        level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, gb, method)
    colnames(data) <- c("elements", "gb", "method")
    data
}

plot_benchmark_memory_modesum <- function(outfile){
    data <- aggregate_benchmark_memory_modesum()
    g <- ggplot(data, aes(x = elements, y = gb, fill = method))
    g <- g + geom_bar(stat = "identity", position = "dodge")
    g <- g + ggtitle("modeSum()")
    g <- g + xlab("# Elements")
    g <- g + ylab("Memory usage (GB)")
    g <- g + theme(legend.title = element_blank())
    g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
    g <- g + theme(axis.text.x = element_text(size=12))
    g <- g + theme(axis.text.y = element_text(size=12))
    g <- g + theme(axis.title.x = element_text(size=18))
    g <- g + theme(axis.title.y = element_text(size=18))
    ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_benchmark_memory_innerprod <- function(){
    files <- list.files("benchmarks", pattern="^innerprod")
    gb <- sapply(files, function(x){
        read.table(paste0("benchmarks/", x),
            stringsAsFactors=FALSE)[2,3]})
    gb <- as.numeric(gb) / 10^3
    elements <- gsub("innerprod_.*tensor_", "", gsub(".txt", "", files))
    method <- gsub("innerprod_", "", gsub("_1E.*txt", "", files))
    method <- factor(method,
        level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, gb, method)
    colnames(data) <- c("elements", "gb", "method")
    data
}

plot_benchmark_memory_innerprod <- function(outfile){
    data <- aggregate_benchmark_memory_innerprod()
    g <- ggplot(data, aes(x = elements, y = gb, fill = method))
    g <- g + geom_bar(stat = "identity", position = "dodge")
    g <- g + ggtitle("innerProd()")
    g <- g + xlab("# Elements")
    g <- g + ylab("Memory usage (GB)")
    g <- g + theme(legend.title = element_blank())
    g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
    g <- g + theme(axis.text.x = element_text(size=12))
    g <- g + theme(axis.text.y = element_text(size=12))
    g <- g + theme(axis.title.x = element_text(size=18))
    g <- g + theme(axis.title.y = element_text(size=18))
    ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_benchmark_memory_vec <- function(){
    files <- list.files("benchmarks", pattern="^vec")
    gb <- sapply(files, function(x){
        read.table(paste0("benchmarks/", x),
            stringsAsFactors=FALSE)[2,3]})
    gb <- as.numeric(gb) / 10^3
    elements <- gsub("vec_.*tensor_", "", gsub(".txt", "", files))
    method <- gsub("vec_", "", gsub("_1E.*txt", "", files))
    method <- factor(method,
        level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, gb, method)
    colnames(data) <- c("elements", "gb", "method")
    data
}

plot_benchmark_memory_vec <- function(outfile){
    data <- aggregate_benchmark_memory_vec()
    g <- ggplot(data, aes(x = elements, y = gb, fill = method))
    g <- g + geom_bar(stat = "identity", position = "dodge")
    g <- g + ggtitle("vec()")
    g <- g + xlab("# Elements")
    g <- g + ylab("Memory usage (GB)")
    g <- g + theme(legend.title = element_blank())
    g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
    g <- g + theme(axis.text.x = element_text(size=12))
    g <- g + theme(axis.text.y = element_text(size=12))
    g <- g + theme(axis.title.x = element_text(size=18))
    g <- g + theme(axis.title.y = element_text(size=18))
    ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_benchmark_memory_kronecker <- function(){
    files <- list.files("benchmarks", pattern="^kronecker")
    gb <- sapply(files, function(x){
        read.table(paste0("benchmarks/", x),
            stringsAsFactors=FALSE)[2,3]})
    gb <- as.numeric(gb) / 10^3
    elements <- gsub("kronecker_.*tensor_", "", gsub(".txt", "", files))
    method <- gsub("kronecker_", "", gsub("_1E.*txt", "", files))
    method <- factor(method,
        level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, gb, method)
    colnames(data) <- c("elements", "gb", "method")
    data
}

plot_benchmark_memory_kronecker <- function(outfile){
    data <- aggregate_benchmark_memory_kronecker()
    g <- ggplot(data, aes(x = elements, y = gb, fill = method))
    g <- g + geom_bar(stat = "identity", position = "dodge")
    g <- g + ggtitle("kronecker()")
    g <- g + xlab("# Elements")
    g <- g + ylab("Memory usage (GB)")
    g <- g + theme(legend.title = element_blank())
    g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
    g <- g + theme(axis.text.x = element_text(size=12))
    g <- g + theme(axis.text.y = element_text(size=12))
    g <- g + theme(axis.title.x = element_text(size=18))
    g <- g + theme(axis.title.y = element_text(size=18))
    ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_benchmark_memory_fold <- function(){
    files <- list.files("benchmarks", pattern="^fold")
    gb <- sapply(files, function(x){
        read.table(paste0("benchmarks/", x),
            stringsAsFactors=FALSE)[2,3]})
    gb <- as.numeric(gb) / 10^3
    elements <- gsub("fold_.*tensor_", "", gsub(".txt", "", files))
    method <- gsub("fold_", "", gsub("_1E.*txt", "", files))
    method <- factor(method,
        level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, gb, method)
    colnames(data) <- c("elements", "gb", "method")
    data
}

plot_benchmark_memory_fold <- function(outfile){
    data <- aggregate_benchmark_memory_fold()
    g <- ggplot(data, aes(x = elements, y = gb, fill = method))
    g <- g + geom_bar(stat = "identity", position = "dodge")
    g <- g + ggtitle("fold()")
    g <- g + xlab("# Elements")
    g <- g + ylab("Memory usage (GB)")
    g <- g + theme(legend.title = element_blank())
    g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
    g <- g + theme(axis.text.x = element_text(size=12))
    g <- g + theme(axis.text.y = element_text(size=12))
    g <- g + theme(axis.title.x = element_text(size=18))
    g <- g + theme(axis.title.y = element_text(size=18))
    ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_benchmark_memory_hosvd <- function(){
    files <- list.files("benchmarks", pattern="^hosvd")
    gb <- sapply(files, function(x){
        read.table(paste0("benchmarks/", x),
            stringsAsFactors=FALSE)[2,3]})
    gb <- as.numeric(gb) / 10^3
    elements <- gsub("hosvd_.*tensor_", "", gsub(".txt", "", files))
    method <- gsub("hosvd_", "", gsub("_1E.*txt", "", files))
    method <- factor(method,
        level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, gb, method)
    colnames(data) <- c("elements", "gb", "method")
    data
}

plot_benchmark_memory_hosvd <- function(outfile){
    data <- aggregate_benchmark_memory_hosvd()
    g <- ggplot(data, aes(x = elements, y = gb, fill = method))
    g <- g + geom_bar(stat = "identity", position = "dodge")
    g <- g + ggtitle("hosvd()")
    g <- g + xlab("# Elements")
    g <- g + ylab("Memory usage (GB)")
    g <- g + theme(legend.title = element_blank())
    g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
    g <- g + theme(axis.text.x = element_text(size=12))
    g <- g + theme(axis.text.y = element_text(size=12))
    g <- g + theme(axis.title.x = element_text(size=18))
    g <- g + theme(axis.title.y = element_text(size=18))
    ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_benchmark_memory_cp <- function(){
    files <- list.files("benchmarks", pattern="^cp")
    gb <- sapply(files, function(x){
        read.table(paste0("benchmarks/", x),
            stringsAsFactors=FALSE)[2,3]})
    gb <- as.numeric(gb) / 10^3
    elements <- gsub("cp_.*tensor_", "", gsub(".txt", "", files))
    method <- gsub("cp_", "", gsub("_1E.*txt", "", files))
    method <- factor(method,
        level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, gb, method)
    colnames(data) <- c("elements", "gb", "method")
    data
}

plot_benchmark_memory_cp <- function(outfile){
    data <- aggregate_benchmark_memory_cp()
    g <- ggplot(data, aes(x = elements, y = gb, fill = method))
    g <- g + geom_bar(stat = "identity", position = "dodge")
    g <- g + ggtitle("cp()")
    g <- g + xlab("# Elements")
    g <- g + ylab("Memory usage (GB)")
    g <- g + theme(legend.title = element_blank())
    g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
    g <- g + theme(axis.text.x = element_text(size=12))
    g <- g + theme(axis.text.y = element_text(size=12))
    g <- g + theme(axis.title.x = element_text(size=18))
    g <- g + theme(axis.title.y = element_text(size=18))
    ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_benchmark_memory_tucker <- function(){
    files <- list.files("benchmarks", pattern="^tucker")
    gb <- sapply(files, function(x){
        read.table(paste0("benchmarks/", x),
            stringsAsFactors=FALSE)[2,3]})
    gb <- as.numeric(gb) / 10^3
    elements <- gsub("tucker_.*tensor_", "", gsub(".txt", "", files))
    method <- gsub("tucker_", "", gsub("_1E.*txt", "", files))
    method <- factor(method,
        level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, gb, method)
    colnames(data) <- c("elements", "gb", "method")
    data
}

plot_benchmark_memory_tucker <- function(outfile){
    data <- aggregate_benchmark_memory_tucker()
    g <- ggplot(data, aes(x = elements, y = gb, fill = method))
    g <- g + geom_bar(stat = "identity", position = "dodge")
    g <- g + ggtitle("tucker()")
    g <- g + xlab("# Elements")
    g <- g + ylab("Memory usage (GB)")
    g <- g + theme(legend.title = element_blank())
    g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
    g <- g + theme(axis.text.x = element_text(size=12))
    g <- g + theme(axis.text.y = element_text(size=12))
    g <- g + theme(axis.title.x = element_text(size=18))
    g <- g + theme(axis.title.y = element_text(size=18))
    ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_benchmark_memory_mpca <- function(){
    files <- list.files("benchmarks", pattern="^mpca")
    gb <- sapply(files, function(x){
        read.table(paste0("benchmarks/", x),
            stringsAsFactors=FALSE)[2,3]})
    gb <- as.numeric(gb) / 10^3
    elements <- gsub("mpca_.*tensor_", "", gsub(".txt", "", files))
    method <- gsub("mpca_", "", gsub("_1E.*txt", "", files))
    method <- factor(method,
        level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, gb, method)
    colnames(data) <- c("elements", "gb", "method")
    data
}

plot_benchmark_memory_mpca <- function(outfile){
    data <- aggregate_benchmark_memory_mpca()
    g <- ggplot(data, aes(x = elements, y = gb, fill = method))
    g <- g + geom_bar(stat = "identity", position = "dodge")
    g <- g + ggtitle("mpca()")
    g <- g + xlab("# Elements")
    g <- g + ylab("Memory usage (GB)")
    g <- g + theme(legend.title = element_blank())
    g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
    g <- g + theme(axis.text.x = element_text(size=12))
    g <- g + theme(axis.text.y = element_text(size=12))
    g <- g + theme(axis.title.x = element_text(size=18))
    g <- g + theme(axis.title.y = element_text(size=18))
    ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_benchmark_memory_pvd <- function(){
    files <- list.files("benchmarks", pattern="^pvd")
    gb <- sapply(files, function(x){
        read.table(paste0("benchmarks/", x),
            stringsAsFactors=FALSE)[2,3]})
    gb <- as.numeric(gb) / 10^3
    elements <- gsub("pvd_.*tensor_", "", gsub(".txt", "", files))
    method <- gsub("pvd_", "", gsub("_1E.*txt", "", files))
    method <- factor(method,
        level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, gb, method)
    colnames(data) <- c("elements", "gb", "method")
    data
}

plot_benchmark_memory_pvd <- function(outfile){
    data <- aggregate_benchmark_memory_pvd()
    g <- ggplot(data, aes(x = elements, y = gb, fill = method))
    g <- g + geom_bar(stat = "identity", position = "dodge")
    g <- g + ggtitle("pvd()")
    g <- g + xlab("# Elements")
    g <- g + ylab("Memory usage (GB)")
    g <- g + theme(legend.title = element_blank())
    g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
    g <- g + theme(axis.text.x = element_text(size=12))
    g <- g + theme(axis.text.y = element_text(size=12))
    g <- g + theme(axis.title.x = element_text(size=18))
    g <- g + theme(axis.title.y = element_text(size=18))
    ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_benchmark_memory_einsum <- function(){
    files <- list.files("benchmarks", pattern="^einsum")
    gb <- sapply(files, function(x){
        read.table(paste0("benchmarks/", x),
            stringsAsFactors=FALSE)[2,3]})
    gb <- as.numeric(gb) / 10^3
    elements <- gsub("einsum_.*tensor_", "", gsub(".txt", "", files))
    method <- gsub("einsum_", "", gsub("_1E.*txt", "", files))
    method <- factor(method,
        level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, gb, method)
    colnames(data) <- c("elements", "gb", "method")
    data
}

plot_benchmark_memory_einsum <- function(outfile){
    data <- aggregate_benchmark_memory_einsum()
    g <- ggplot(data, aes(x = elements, y = gb, fill = method))
    g <- g + geom_bar(stat = "identity", position = "dodge")
    g <- g + ggtitle("einsum()")
    g <- g + xlab("# Elements")
    g <- g + ylab("Memory usage (GB)")
    g <- g + theme(legend.title = element_blank())
    g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
    g <- g + theme(axis.text.x = element_text(size=12))
    g <- g + theme(axis.text.y = element_text(size=12))
    g <- g + theme(axis.title.x = element_text(size=18))
    g <- g + theme(axis.title.y = element_text(size=18))
    ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

# gnutime
plot_gnutime_memory <- function(arithmetic, outfile){
    cmd <- paste0("plot_gnutime_memory_", arithmetic, "(outfile)")
    eval(parse(text=cmd))
}

aggregate_gnutime_memory_unfold <- function(){
    files <- list.files("logs", pattern="^unfold")
    gb <- sapply(files, function(x){
        tmp <- read.delim(paste0("logs/", x), stringsAsFactors=FALSE)[,1]
        tmp <- tmp[grep("Maximum resident", tmp)]
        tmp <- gsub("Maximum resident set size \\(kbytes\\): ", "", tmp)
        })
    gb <- as.numeric(gb) / 10^6
    elements <- gsub("unfold_.*tensor_", "", gsub(".log", "", files))
    method <- gsub("unfold_", "", gsub("_1E.*log", "", files))
    method <- factor(method,
        level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, gb, method)
    colnames(data) <- c("elements", "gb", "method")
    data
}

plot_gnutime_memory_unfold <- function(outfile){
    data <- aggregate_gnutime_memory_unfold()
    g <- ggplot(data, aes(x = elements, y = gb, fill = method))
    g <- g + geom_bar(stat = "identity", position = "dodge")
    g <- g + ggtitle("unfold()")
    g <- g + xlab("# Elements")
    g <- g + ylab("Memory usage (GB)")
    g <- g + theme(legend.title = element_blank())
    g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
    g <- g + theme(axis.text.x = element_text(size=12))
    g <- g + theme(axis.text.y = element_text(size=12))
    g <- g + theme(axis.title.x = element_text(size=18))
    g <- g + theme(axis.title.y = element_text(size=18))
    ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_gnutime_memory_modesum <- function(){
    files <- list.files("logs", pattern="^modesum")
    gb <- sapply(files, function(x){
        tmp <- read.delim(paste0("logs/", x), stringsAsFactors=FALSE)[,1]
        tmp <- tmp[grep("Maximum resident", tmp)]
        tmp <- gsub("Maximum resident set size \\(kbytes\\): ", "", tmp)
        })
    gb <- as.numeric(gb) / 10^6
    elements <- gsub("modesum_.*tensor_", "", gsub(".log", "", files))
    method <- gsub("modesum_", "", gsub("_1E.*log", "", files))
    method <- factor(method,
        level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, gb, method)
    colnames(data) <- c("elements", "gb", "method")
    data
}

plot_gnutime_memory_modesum <- function(outfile){
    data <- aggregate_gnutime_memory_modesum()
    g <- ggplot(data, aes(x = elements, y = gb, fill = method))
    g <- g + geom_bar(stat = "identity", position = "dodge")
    g <- g + ggtitle("modeSum()")
    g <- g + xlab("# Elements")
    g <- g + ylab("Memory usage (GB)")
    g <- g + theme(legend.title = element_blank())
    g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
    g <- g + theme(axis.text.x = element_text(size=12))
    g <- g + theme(axis.text.y = element_text(size=12))
    g <- g + theme(axis.title.x = element_text(size=18))
    g <- g + theme(axis.title.y = element_text(size=18))
    ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_gnutime_memory_innerprod <- function(){
    files <- list.files("logs", pattern="^innerprod")
    gb <- sapply(files, function(x){
        tmp <- read.delim(paste0("logs/", x), stringsAsFactors=FALSE)[,1]
        tmp <- tmp[grep("Maximum resident", tmp)]
        tmp <- gsub("Maximum resident set size \\(kbytes\\): ", "", tmp)
        })
    gb <- as.numeric(gb) / 10^6
    elements <- gsub("innerprod_.*tensor_", "", gsub(".log", "", files))
    method <- gsub("innerprod_", "", gsub("_1E.*log", "", files))
    method <- factor(method,
        level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, gb, method)
    colnames(data) <- c("elements", "gb", "method")
    data
}

plot_gnutime_memory_innerprod <- function(outfile){
    data <- aggregate_gnutime_memory_innerprod()
    g <- ggplot(data, aes(x = elements, y = gb, fill = method))
    g <- g + geom_bar(stat = "identity", position = "dodge")
    g <- g + ggtitle("innerProd()")
    g <- g + xlab("# Elements")
    g <- g + ylab("Memory usage (GB)")
    g <- g + theme(legend.title = element_blank())
    g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
    g <- g + theme(axis.text.x = element_text(size=12))
    g <- g + theme(axis.text.y = element_text(size=12))
    g <- g + theme(axis.title.x = element_text(size=18))
    g <- g + theme(axis.title.y = element_text(size=18))
    ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_gnutime_memory_vec <- function(){
    files <- list.files("logs", pattern="^vec")
    gb <- sapply(files, function(x){
        tmp <- read.delim(paste0("logs/", x), stringsAsFactors=FALSE)[,1]
        tmp <- tmp[grep("Maximum resident", tmp)]
        tmp <- gsub("Maximum resident set size \\(kbytes\\): ", "", tmp)
        })
    gb <- as.numeric(gb) / 10^6
    elements <- gsub("vec_.*tensor_", "", gsub(".log", "", files))
    method <- gsub("vec_", "", gsub("_1E.*log", "", files))
    method <- factor(method,
        level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, gb, method)
    colnames(data) <- c("elements", "gb", "method")
    data
}

plot_gnutime_memory_vec <- function(outfile){
    data <- aggregate_gnutime_memory_vec()
    g <- ggplot(data, aes(x = elements, y = gb, fill = method))
    g <- g + geom_bar(stat = "identity", position = "dodge")
    g <- g + ggtitle("vec()")
    g <- g + xlab("# Elements")
    g <- g + ylab("Memory usage (GB)")
    g <- g + theme(legend.title = element_blank())
    g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
    g <- g + theme(axis.text.x = element_text(size=12))
    g <- g + theme(axis.text.y = element_text(size=12))
    g <- g + theme(axis.title.x = element_text(size=18))
    g <- g + theme(axis.title.y = element_text(size=18))
    ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_gnutime_memory_kronecker <- function(){
    files <- list.files("logs", pattern="^kronecker")
    gb <- sapply(files, function(x){
        tmp <- read.delim(paste0("logs/", x), stringsAsFactors=FALSE)[,1]
        tmp <- tmp[grep("Maximum resident", tmp)]
        tmp <- gsub("Maximum resident set size \\(kbytes\\): ", "", tmp)
        })
    gb <- as.numeric(gb) / 10^6
    elements <- gsub("kronecker_.*tensor_", "", gsub(".log", "", files))
    method <- gsub("kronecker_", "", gsub("_1E.*log", "", files))
    method <- factor(method,
        level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, gb, method)
    colnames(data) <- c("elements", "gb", "method")
    data
}

plot_gnutime_memory_kronecker <- function(outfile){
    data <- aggregate_gnutime_memory_kronecker()
    g <- ggplot(data, aes(x = elements, y = gb, fill = method))
    g <- g + geom_bar(stat = "identity", position = "dodge")
    g <- g + ggtitle("kronecker()")
    g <- g + xlab("# Elements")
    g <- g + ylab("Memory usage (GB)")
    g <- g + theme(legend.title = element_blank())
    g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
    g <- g + theme(axis.text.x = element_text(size=12))
    g <- g + theme(axis.text.y = element_text(size=12))
    g <- g + theme(axis.title.x = element_text(size=18))
    g <- g + theme(axis.title.y = element_text(size=18))
    ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_gnutime_memory_fold <- function(){
    files <- list.files("logs", pattern="^fold")
    gb <- sapply(files, function(x){
        tmp <- read.delim(paste0("logs/", x), stringsAsFactors=FALSE)[,1]
        tmp <- tmp[grep("Maximum resident", tmp)]
        tmp <- gsub("Maximum resident set size \\(kbytes\\): ", "", tmp)
        })
    gb <- as.numeric(gb) / 10^6
    elements <- gsub("fold_.*tensor_", "", gsub(".log", "", files))
    method <- gsub("fold_", "", gsub("_1E.*log", "", files))
    method <- factor(method,
        level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, gb, method)
    colnames(data) <- c("elements", "gb", "method")
    data
}

plot_gnutime_memory_fold <- function(outfile){
    data <- aggregate_gnutime_memory_fold()
    g <- ggplot(data, aes(x = elements, y = gb, fill = method))
    g <- g + geom_bar(stat = "identity", position = "dodge")
    g <- g + ggtitle("fold()")
    g <- g + xlab("# Elements")
    g <- g + ylab("Memory usage (GB)")
    g <- g + theme(legend.title = element_blank())
    g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
    g <- g + theme(axis.text.x = element_text(size=12))
    g <- g + theme(axis.text.y = element_text(size=12))
    g <- g + theme(axis.title.x = element_text(size=18))
    g <- g + theme(axis.title.y = element_text(size=18))
    ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_gnutime_memory_hosvd <- function(){
    files <- list.files("logs", pattern="^hosvd")
    gb <- sapply(files, function(x){
        tmp <- read.delim(paste0("logs/", x), stringsAsFactors=FALSE)[,1]
        tmp <- tmp[grep("Maximum resident", tmp)]
        tmp <- gsub("Maximum resident set size \\(kbytes\\): ", "", tmp)
        })
    gb <- as.numeric(gb) / 10^6
    elements <- gsub("hosvd_.*tensor_", "", gsub(".log", "", files))
    method <- gsub("hosvd_", "", gsub("_1E.*log", "", files))
    method <- factor(method,
        level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, gb, method)
    colnames(data) <- c("elements", "gb", "method")
    data
}

plot_gnutime_memory_hosvd <- function(outfile){
    data <- aggregate_gnutime_memory_hosvd()
    g <- ggplot(data, aes(x = elements, y = gb, fill = method))
    g <- g + geom_bar(stat = "identity", position = "dodge")
    g <- g + ggtitle("hosvd()")
    g <- g + xlab("# Elements")
    g <- g + ylab("Memory usage (GB)")
    g <- g + theme(legend.title = element_blank())
    g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
    g <- g + theme(axis.text.x = element_text(size=12))
    g <- g + theme(axis.text.y = element_text(size=12))
    g <- g + theme(axis.title.x = element_text(size=18))
    g <- g + theme(axis.title.y = element_text(size=18))
    ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_gnutime_memory_cp <- function(){
    files <- list.files("logs", pattern="^cp")
    gb <- sapply(files, function(x){
        tmp <- read.delim(paste0("logs/", x), stringsAsFactors=FALSE)[,1]
        tmp <- tmp[grep("Maximum resident", tmp)]
        tmp <- gsub("Maximum resident set size \\(kbytes\\): ", "", tmp)
        })
    gb <- as.numeric(gb) / 10^6
    elements <- gsub("cp_.*tensor_", "", gsub(".log", "", files))
    method <- gsub("cp_", "", gsub("_1E.*log", "", files))
    method <- factor(method,
        level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, gb, method)
    colnames(data) <- c("elements", "gb", "method")
    data
}

plot_gnutime_memory_cp <- function(outfile){
    data <- aggregate_gnutime_memory_cp()
    g <- ggplot(data, aes(x = elements, y = gb, fill = method))
    g <- g + geom_bar(stat = "identity", position = "dodge")
    g <- g + ggtitle("cp()")
    g <- g + xlab("# Elements")
    g <- g + ylab("Memory usage (GB)")
    g <- g + theme(legend.title = element_blank())
    g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
    g <- g + theme(axis.text.x = element_text(size=12))
    g <- g + theme(axis.text.y = element_text(size=12))
    g <- g + theme(axis.title.x = element_text(size=18))
    g <- g + theme(axis.title.y = element_text(size=18))
    ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_gnutime_memory_tucker <- function(){
    files <- list.files("logs", pattern="^tucker")
    gb <- sapply(files, function(x){
        tmp <- read.delim(paste0("logs/", x), stringsAsFactors=FALSE)[,1]
        tmp <- tmp[grep("Maximum resident", tmp)]
        tmp <- gsub("Maximum resident set size \\(kbytes\\): ", "", tmp)
        })
    gb <- as.numeric(gb) / 10^6
    elements <- gsub("tucker_.*tensor_", "", gsub(".log", "", files))
    method <- gsub("tucker_", "", gsub("_1E.*log", "", files))
    method <- factor(method,
        level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, gb, method)
    colnames(data) <- c("elements", "gb", "method")
    data
}

plot_gnutime_memory_tucker <- function(outfile){
    data <- aggregate_gnutime_memory_tucker()
    g <- ggplot(data, aes(x = elements, y = gb, fill = method))
    g <- g + geom_bar(stat = "identity", position = "dodge")
    g <- g + ggtitle("tucker()")
    g <- g + xlab("# Elements")
    g <- g + ylab("Memory usage (GB)")
    g <- g + theme(legend.title = element_blank())
    g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
    g <- g + theme(axis.text.x = element_text(size=12))
    g <- g + theme(axis.text.y = element_text(size=12))
    g <- g + theme(axis.title.x = element_text(size=18))
    g <- g + theme(axis.title.y = element_text(size=18))
    ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_gnutime_memory_mpca <- function(){
    files <- list.files("logs", pattern="^mpca")
    gb <- sapply(files, function(x){
        tmp <- read.delim(paste0("logs/", x), stringsAsFactors=FALSE)[,1]
        tmp <- tmp[grep("Maximum resident", tmp)]
        tmp <- gsub("Maximum resident set size \\(kbytes\\): ", "", tmp)
        })
    gb <- as.numeric(gb) / 10^6
    elements <- gsub("mpca_.*tensor_", "", gsub(".log", "", files))
    method <- gsub("mpca_", "", gsub("_1E.*log", "", files))
    method <- factor(method,
        level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, gb, method)
    colnames(data) <- c("elements", "gb", "method")
    data
}

plot_gnutime_memory_mpca <- function(outfile){
    data <- aggregate_gnutime_memory_mpca()
    g <- ggplot(data, aes(x = elements, y = gb, fill = method))
    g <- g + geom_bar(stat = "identity", position = "dodge")
    g <- g + ggtitle("mpca()")
    g <- g + xlab("# Elements")
    g <- g + ylab("Memory usage (GB)")
    g <- g + theme(legend.title = element_blank())
    g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
    g <- g + theme(axis.text.x = element_text(size=12))
    g <- g + theme(axis.text.y = element_text(size=12))
    g <- g + theme(axis.title.x = element_text(size=18))
    g <- g + theme(axis.title.y = element_text(size=18))
    ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_gnutime_memory_pvd <- function(){
    files <- list.files("logs", pattern="^pvd")
    gb <- sapply(files, function(x){
        tmp <- read.delim(paste0("logs/", x), stringsAsFactors=FALSE)[,1]
        tmp <- tmp[grep("Maximum resident", tmp)]
        tmp <- gsub("Maximum resident set size \\(kbytes\\): ", "", tmp)
        })
    gb <- as.numeric(gb) / 10^6
    elements <- gsub("pvd_.*tensor_", "", gsub(".log", "", files))
    method <- gsub("pvd_", "", gsub("_1E.*log", "", files))
    method <- factor(method,
        level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, gb, method)
    colnames(data) <- c("elements", "gb", "method")
    data
}

plot_gnutime_memory_pvd <- function(outfile){
    data <- aggregate_gnutime_memory_pvd()
    g <- ggplot(data, aes(x = elements, y = gb, fill = method))
    g <- g + geom_bar(stat = "identity", position = "dodge")
    g <- g + ggtitle("pvd()")
    g <- g + xlab("# Elements")
    g <- g + ylab("Memory usage (GB)")
    g <- g + theme(legend.title = element_blank())
    g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
    g <- g + theme(axis.text.x = element_text(size=12))
    g <- g + theme(axis.text.y = element_text(size=12))
    g <- g + theme(axis.title.x = element_text(size=18))
    g <- g + theme(axis.title.y = element_text(size=18))
    ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}

aggregate_gnutime_memory_einsum <- function(){
    files <- list.files("logs", pattern="^einsum")
    gb <- sapply(files, function(x){
        tmp <- read.delim(paste0("logs/", x), stringsAsFactors=FALSE)[,1]
        tmp <- tmp[grep("Maximum resident", tmp)]
        tmp <- gsub("Maximum resident set size \\(kbytes\\): ", "", tmp)
        })
    gb <- as.numeric(gb) / 10^6
    elements <- gsub("einsum_.*tensor_", "", gsub(".log", "", files))
    method <- gsub("einsum_", "", gsub("_1E.*log", "", files))
    method <- factor(method,
        level=c("rtensor", "dense_delayedtensor", "sparse_delayedtensor"))
    data <- data.frame(elements, gb, method)
    colnames(data) <- c("elements", "gb", "method")
    data
}

plot_gnutime_memory_einsum <- function(outfile){
    data <- aggregate_gnutime_memory_einsum()
    g <- ggplot(data, aes(x = elements, y = gb, fill = method))
    g <- g + geom_bar(stat = "identity", position = "dodge")
    g <- g + ggtitle("einsum()")
    g <- g + xlab("# Elements")
    g <- g + ylab("Memory usage (GB)")
    g <- g + theme(legend.title = element_blank())
    g <- g + theme(plot.title = element_text(size=22, hjust = 0.5))
    g <- g + theme(axis.text.x = element_text(size=12))
    g <- g + theme(axis.text.y = element_text(size=12))
    g <- g + theme(axis.title.x = element_text(size=18))
    g <- g + theme(axis.title.y = element_text(size=18))
    ggsave(file=outfile, plot=g, dpi=200, width=7.0, height=4.0)
}