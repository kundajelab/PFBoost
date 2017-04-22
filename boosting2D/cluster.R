### Code to cluster regulators
### 6/15/15


library(futile.matrix)
r = read.matrix('/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/regulatorExpression_full.txt')

### SCG3
reg = read.table('/srv/gsfs0/projects/kundaje/users/pgreens/projects/boosting/data/hematopoeisis_data/regulatorExpression_pairwise_full.txt')
d = dist(reg)
c = hclust(d, method='average')

### Plot everything (TOO BIG)
# plot(c, labels=TRUE)

# Figure out with SPI1 is close to
# distance matrix
m =as.matrix(d)
# correlation matrix
mat=reg[names(which(apply(reg, 1, sd)>0.1)),]
m = cor(t(mat))

m[order(m[,'SPI1']),'SPI1']
paste(names(m[order(m[,'TAD2AB']),'TAD2AB'][1:50]), collapse=" ")
paste(names(m[order(m[,'SPI1']),'SPI1'][1:50]), collapse=" ")
m['TADA2B','SPI1']

png('/srv/gsfs0/projects/kundaje/users/pgreens/projects/boosting/plots/hema_full_regulator_heatmap.png',width = 16, height = 16, units = "in",res=300)
heatmap.2(m, trace="none", dendrogram="row")
dev.off()