#install.packages("GeneNet")

#require(GeneNet)
library(bnlearn)
        
#data(ecoli, package="GeneNet")

ecoli_net = bn.net(ecoli70)
