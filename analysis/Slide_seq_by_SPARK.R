counts = read.csv("spatial_gene_profile_slide_seq_2w_raw.csv",row.names = 1)
#counts = read.csv("spatial_gene_profile_slide_seq_3d_raw.csv",row.names = 1)
x = counts[1,3541]
y = counts[1,3542]
counts = counts[,1:3540]
colnames(counts)=c(1:3540)

rows <- sample(nrow(counts))
counts <- counts[rows,]

bin1 = rep(c(1:x),each=y)
bin2 = rep(c(1:y),times=x)
samples = data.frame(cbind(bin1,bin2))
colnames(samples)=c("x","y")
rownames(samples)=colnames(counts)


library(SPARK)
spark <- CreateSPARKObject(counts = counts, location = samples[,1:2], 
                           percentage = 0, min_total_counts = 0)
#spark <- CreateSPARKObject(counts = counts, location = info[, 1:2], 
#                           percentage = 0.1, min_total_counts = 10)

spark@lib_size <- apply(spark@counts, 2, sum)
spark <- spark.vc(spark, covariates = NULL, lib_size = spark@lib_size, 
                  num_core = 8, verbose = T, fit.maxiter = 500)

spark <- spark.test(spark, check_positive = T, verbose = T)

##Cluster SE genes using Hierarchical clustering (optional)
source("SPARK-Analysis-master/SPARK-Analysis-master/funcs/funcs.R")

LMReg <- function(ct, T) {
  return(lm(ct ~ T)$residuals)
}

##
counts <- spark@counts
info <- spark@location
vst_count <- var_stabilize(counts) # R function in funcs.R
sig_vst_count <- vst_count[which(spark@res_mtest$adjusted_pvalue < 0.05),]
sig_vst_res <- t(apply(sig_vst_count, 1, LMReg, T = log(spark@lib_size)))

library(amap)
hc <- hcluster(sig_vst_res, method = "euc", link = "ward", nbproc = 1, 
               doubleprecision = TRUE)
numC <- 10
memb <- cutree(hc, k = numC)
