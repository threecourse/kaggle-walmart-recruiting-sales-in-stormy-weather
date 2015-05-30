# read files
df <- read.table("data/train.csv", sep=',', header=T)
store_item_nbrs <- read.table("model/store_item_nbrs.csv", sep=',', header=T)

# calculate log1p
df$log1p <- log1p(df$units)

# calculate days from 2012-01-01
origin <- as.integer(floor(julian(as.POSIXlt('2012-01-01'))))
df$date2j <- as.integer(floor(julian((as.POSIXlt(df$date))))) - origin

# exclude 2013-12-25
date_excl <- as.integer(floor(julian(as.POSIXlt('2013-12-25')))) - origin
df <- df[df$date2j != date_excl, ]

# for each item_nbr/store_nbrs, fitting by ppr function
df_fitted <- data.frame(date2j=c(), sno=c(), ino=c())

rng <- 1:nrow(store_item_nbrs)

for (i in rng) {
  ino <- store_item_nbrs[i, "item_nbr"]
  sno <- store_item_nbrs[i, "store_nbr"]
  df0 <- subset(df, store_nbr == sno & item_nbr == ino)
  df0.ppr <- ppr(log1p ~ date2j, data = df0, nterms=3, max.terms=5)
  
  df1 <- data.frame(date2j=0:1034, store_nbr=sno, item_nbr=ino)
  df1$ppr_fitted <- predict(df0.ppr, df1)
  
  #plot(df0$date2j, df0$log1p, main=paste(c("result", ino, sno)))
  #lines(newdf$date2j, newdf$gampred, col="red")
  #lines(newdf$date2j, newdf$pprpred, col="blue")
  
  df_fitted <- rbind(df_fitted, df1)
}

write.table(df_fitted, "model/baseline.csv", quote=F, col.names=T, append=F, sep=",", row.names=F)

cat("curve fitting finished")

q("no")