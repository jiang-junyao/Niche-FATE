# Called by nichefate.signal_significance(): official CytoSignal (R package cytosignal) on one section for a list of ligand-receptor
# pairs, then its own per-cell significance test (inferSignif with its defaults: spatial FDR, p < 0.05 on the smoothed LR score).
# Rscript cytosignal_signif.R <dir>  -> <dir>/signif_<slot>.tsv (interaction id, significant cells comma-separated),
#                                       <dir>/kept_<slot>.txt (cells scored after QC), <dir>/ids.tsv
suppressMessages({library(cytosignal); library(Matrix)})
d <- commandArgs(TRUE)[1]
X <- readMM(file.path(d, "counts.mtx")); X <- as(X, "CsparseMatrix")
rownames(X) <- readLines(file.path(d, "genes.txt")); colnames(X) <- readLines(file.path(d, "cells.txt"))
loc <- as.matrix(read.csv(file.path(d, "loc.csv"), header = FALSE)); rownames(loc) <- colnames(X); colnames(loc) <- c("x", "y")
lr <- read.csv(file.path(d, "lr.csv"), stringsAsFactors = FALSE)
par <- as.numeric(readLines(file.path(d, "params.txt")))
split_ <- function(s) strsplit(s, "_", fixed = TRUE)[[1]]
ids <- sprintf("CPI-CC%06d", seq_len(nrow(lr)))
mxL <- max(sapply(lr$ligand, function(s) length(split_(s)))); mxR <- max(sapply(lr$receptor, function(s) length(split_(s))))
pad <- function(v, n) c(v, rep(NA, n - length(v)))
ligs <- data.frame(interaction_id = ids, do.call(rbind, lapply(lr$ligand, function(s) pad(split_(s), mxL))), stringsAsFactors = FALSE)
recs <- data.frame(interaction_id = ids, do.call(rbind, lapply(lr$receptor, function(s) pad(split_(s), mxR))), stringsAsFactors = FALSE)
colnames(ligs) <- c("interaction_id", paste0("L", seq_len(mxL))); colnames(recs) <- c("interaction_id", paste0("R", seq_len(mxR)))
itype <- data.frame(interaction_id = ids, type = lr$type, stringsAsFactors = FALSE)
g2u <- data.frame(gene_name = toupper(rownames(X)), uniprot = paste0("U", toupper(rownames(X))), stringsAsFactors = FALSE)
ligs[, -1] <- lapply(ligs[, -1, drop = FALSE], toupper); recs[, -1] <- lapply(recs[, -1, drop = FALSE], toupper)
db <- formatLRDB(itype, ligs, recs, g2u)
cs <- createCytoSignal(raw.data = X, cells.loc = loc)
cs <- addIntrDB(cs, g2u, db$db.diff, db$db.cont, db$intr.index)
cs <- removeLowQuality(cs, counts.thresh = par[3])
cs <- changeUniprot(cs)
cs <- inferEpsParams(cs, scale.factor = par[1])
cs <- findNN(cs)
cs <- imputeLR(cs)
cs <- inferIntrScore(cs, intr.type = "diff", perm.size = par[2])
slots <- names(cs@lrscore); slots <- slots[slots != "default"]
cs <- inferSignif(cs, p.thresh = 0.05, reads.thresh = 0, sig.thresh = 0)    # default: the smoothed LR-score slot(s)
cat("slots:", slots, "\n")
for (s in slots) {
  if (!isS4(cs@lrscore[[s]]) || length(cs@lrscore[[s]]@res.list) == 0) next
  f <- gsub("[^A-Za-z0-9]+", "_", s); rl <- cs@lrscore[[s]]@res.list$result
  writeLines(rownames(cs@lrscore[[s]]@score), file.path(d, paste0("kept_", f, ".txt")))
  out <- vapply(names(rl), function(n) paste(rl[[n]], collapse = ","), "")
  write.table(data.frame(intr = names(rl), cells = out), file.path(d, paste0("signif_", f, ".tsv")), sep = "\t", quote = FALSE, row.names = FALSE)
}
writeLines(paste(ids, lr$ligand, lr$receptor, sep = "\t"), file.path(d, "ids.tsv"))
cat("done", slots, "\n")
