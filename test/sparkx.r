library(Seurat)
library(SPARK)
setwd('D:\\AI\\mouse postbrain')
rds<-readRDS('mouse_postbrain.rds')
counts<-rds@assays$RNA@counts
location<-as.matrix(data.frame(x=rds@meta.data$x,y=rds@meta.data$y))
mt_idx<- grep("mt-",rownames(counts))
if(length(mt_idx)!=0){
  counts    <- counts[-mt_idx,]
}

sparkX <- sparkx(counts,location,numCores=1,option="mixture")
write.table(sparkX$res_mtest[sparkX$res_mtest<0.01,2,drop=F],'mouse_postbrain_sparkx_svg.txt',quote=F,sep='\t')


setwd('D:\\AI\\breast')
rds<-readRDS('breast_cancer.rds')
counts<-rds@assays$RNA@counts
location<-as.matrix(data.frame(x=rds@meta.data$x,y=rds@meta.data$y))
mt_idx<- grep("MT-",rownames(counts))
if(length(mt_idx)!=0){
  counts    <- counts[-mt_idx,]
}

sparkX <- sparkx(counts,location,numCores=1,option="mixture")
write.table(sparkX$res_mtest[sparkX$res_mtest<0.01,2,drop=F],'breast_sparkx_svg.txt',quote=F,sep='\t')
