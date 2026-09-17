suppressMessages({library(readxl); library(openxlsx); library(data.table)})
d <- "processed data/trainingfinal/"
pat <- c(LH="^long hour", INFLEX="^inflexible", VAR="^schedule and hour variability", INSUF="^insufficient", LOWPRED="^low predict", UNDES="^undesirable", NOPTO="^lack of paid", WLC="^broad life", NOWFH="^lack of work-from", GENC="^general complaints",
  REG="^regular hours", FLEX="^flexibility or autonomy", DES="^desirable", CONS="^hour consistency", PRED="^hour predictability", PTO="^paid time", ADDHRS="^additional hours", WLB="^work-life balance", WFH="^work-from-home", OVERC="^overall compliment",
  UNDER="^under-staffing", OVER="^over-staffing", JCAUT="^lack of job autonomy", UNEVEN="^uneven or unfair", CYCLIC="^inconsistent or cyclical", MANAGE="^manageable", JPAUT="^job autonomy and")
SC <- names(pat)[1:20]; JC <- names(pat)[21:27]
CH <- rbind(
  data.table(file="r2", id=81779699, dim="UNDER", val=0, reason="E7 bare workload stays uncoded"),
  data.table(file="dd", id=12489, dim="FLEX", val=0, reason="E13 plenty of free time = manageable workload"),
  data.table(file="dd", id=12489, dim="MANAGE", val=1, reason="E13 plenty of free time = manageable workload"),
  data.table(file="dd", id=86371, dim="INFLEX", val=1, reason="E24 hours hard to work around classes: WLC + inflexible"),
  data.table(file="dd", id=58589, dim="WLC", val=1, reason="E24 schedule around personal life: WLC + inflexible"),
  data.table(file="dd", id=58172, dim="WLC", val=1, reason="E24 kids schedule, hard to take off: WLC + inflexible"))
files <- list(r2=c("candidates_round2_final.xlsx","to_code"), dd=c("dataset_detailed_dimensions.xlsx","detailed"))
LOG <- list(); ddtc <- list()
for (key in names(files)) {
  f <- files[[key]][1]; sh <- files[[key]][2]
  h <- read_excel(paste0(d,f), sheet=sh, col_names=FALSE, n_max=2); x <- read_excel(paste0(d,f), sheet=sh, col_names=FALSE, skip=2)
  h1 <- as.character(unlist(h[1,])); h2 <- as.character(unlist(h[2,]))
  colidx <- sapply(pat, function(p) which(grepl(p, h2))); tc <- sapply(c("schedule_related","job_control_related","not_related"), function(n) which(h1==n))
  wb <- loadWorkbook(paste0(d,f)); cc <- CH[file==key]
  for (i in seq_len(nrow(cc))) { r <- which(x[[1]]==cc$id[i]); stopifnot(length(r)==1); j <- colidx[[cc$dim[i]]]
    old <- x[[j]][r]; oldv <- ifelse(is.na(old),0,old); x[r,j] <- if (cc$val[i]==1) 1 else NA
    writeData(wb, sh, if (cc$val[i]==1) 1 else NA_real_, startRow=r+2, startCol=j)
    LOG[[length(LOG)+1]] <- data.table(file=f, id=cc$id[i], dim=h2[j], old=oldv, new=cc$val[i], changed=as.integer(oldv!=cc$val[i]), reason=cc$reason[i]) }
  if (key=="dd") for (id in unique(cc$id)) { r <- which(x[[1]]==id)
    s <- as.integer(any(x[r, colidx[SC]]==1, na.rm=TRUE)); j <- as.integer(any(x[r, colidx[JC]]==1, na.rm=TRUE)); n <- as.integer(s==0&j==0)
    old <- c(x[[tc[1]]][r], x[[tc[2]]][r], x[[tc[3]]][r]); new <- c(s,j,n)
    for (k in 1:3) if (old[k]!=new[k]) { writeData(wb, sh, new[k], startRow=r+2, startCol=tc[k]); LOG[[length(LOG)+1]] <- data.table(file="dataset_detailed_dimensions.xlsx (and dataset_three_categories.xlsx)", id=id, dim=names(tc)[k], old=old[k], new=new[k], changed=1L, reason="three-category flag recomputed from dimensions"); ddtc[[length(ddtc)+1]] <- data.table(id=id, k=k, v=new[k]) } }
  saveWorkbook(wb, paste0(d,f), overwrite=TRUE)
}
if (length(ddtc)) { wb3 <- loadWorkbook(paste0(d,"dataset_three_categories.xlsx")); x3 <- read.xlsx(paste0(d,"dataset_three_categories.xlsx"))
  for (t in rbindlist(ddtc)[, .(id,k,v)][, .SD]) NULL
  T3 <- rbindlist(ddtc); for (i in seq_len(nrow(T3))) { r3 <- which(x3$id==T3$id[i]); stopifnot(length(r3)==1); writeData(wb3, 1, T3$v[i], startRow=r3+1, startCol=which(names(x3)==c("schedule_related","job_control_related","not_related")[T3$k[i]])) }
  saveWorkbook(wb3, paste0(d,"dataset_three_categories.xlsx"), overwrite=TRUE) }
L <- rbindlist(LOG); print(L[, .(file=substr(file,1,20), id, dim, old, new, changed)])
log <- fread(paste0(d,"coding_changes_2026-09-04.csv")); fwrite(rbind(log, L), paste0(d,"coding_changes_2026-09-04.csv")); cat("log rows:", nrow(log)+nrow(L), "\n")
# verify
for (key in names(files)) { f <- files[[key]][1]; sh <- files[[key]][2]; h <- read_excel(paste0(d,f), sheet=sh, col_names=FALSE, n_max=2); x <- read_excel(paste0(d,f), sheet=sh, col_names=FALSE, skip=2); h2 <- as.character(unlist(h[2,]))
  for (id in unique(CH[file==key, id])) { r <- which(x[[1]]==id); on <- h2[which(!is.na(h2) & unlist(x[r, which(!is.na(h2))])==1)]; cat(key, id, ":", paste(sapply(on, function(n) names(pat)[sapply(pat, function(p) grepl(p,n))]), collapse=";"), if (key=="dd") paste("| tc:", paste(unlist(x[r,6:8]), collapse=",")), "\n") }
  cat(key, "rows:", nrow(x), " merges:", length(loadWorkbook(paste0(d,f))$worksheets[[1]]$mergeCells), "\n") }
