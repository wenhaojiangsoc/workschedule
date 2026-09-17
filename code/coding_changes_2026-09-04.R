suppressMessages({library(readxl); library(openxlsx); library(data.table)})
d <- "processed data/trainingfinal/"
S <- Sys.getenv("S")
NEW_UNEVEN <- "uneven or unfair allocation of work"
NEW_CYCLIC <- "inconsistent or cyclical workload over time"
pat <- c(LH="^long hour", INFLEX="^inflexible", VAR="^schedule and hour variability", INSUF="^insufficient", LOWPRED="^low predict", UNDES="^undesirable", NOPTO="^lack of paid", WLC="^broad life", NOWFH="^lack of work-from", GENC="^general complaints",
  REG="^regular hours", FLEX="^flexibility or autonomy", DES="^desirable", CONS="^hour consistency", PRED="^hour predictability", PTO="^paid time", ADDHRS="^additional hours", WLB="^work-life balance", WFH="^work-from-home", OVERC="^overall compliment",
  UNDER="^under-staffing", OVER="^over-staffing", JCAUT="^lack of job autonomy", UNEVEN="^uneven or unfair", CYCLIC="^inconsistent or cyclical", MANAGE="^manageable", JPAUT="^job autonomy and")
SC <- names(pat)[1:20]; JC <- names(pat)[21:27]

## ---------- change list: file, id, dim, val, cat ----------
ch <- function(file, ids, dim, val, cat) data.table(file=file, id=as.numeric(ids), dim=dim, val=val, cat=cat)
L <- list(
 ch("r2", c(8216730,83151467,63541995,74800677,37074454,82125031,72257694,58866471,76007625,49703089,64970001,32907645,7218508,49520596,4322307,14457241,58926027,81756637,68225030,82727874,31217746,82751595,21782563,10084080), "JCAUT", 1, "A micromanagement"),
 ch("r2", 83151467, "UNDER", 1, "A back-to-back calls, no help"),
 ch("r2", 72257694, "REG", 0, "D wrong polarity"), ch("r2", 72257694, "LH", 1, "D over worked"),
 ch("r2", c(57760272,37575192), "JPAUT", 1, "A2 no micromanaging / little supervision"),
 ch("r2", c(49224222,83946651,6374953,70141314,51091932,34728669,9905777,83960313,56005397,70783964,82602932,78499742,14424529,8952204), "MANAGE", 1, "B busy praise"),
 ch("r2", c(78859505,82282900,21999212,5791638,71711612,76007625,83661953,9379466,74568100,78943540,69747388), "UNEVEN", 1, "C uneven allocation"),
 ch("r2", c(78694844,49769711,83322334,12269076), "CYCLIC", 1, "C2 workload varies over time"),
 ch("r2", 12269076, "UNDER", 1, "C2 crunch periods"), ch("r2", 12269076, "OVER", 1, "C2 downtime between projects"),
 ch("r2", c(3372830,11448475,15364084,19919348,25921932,35777998,37951023,54916491,63392014,64092046,64509950,71392667,74072045,83078423), "CYCLIC", 1, "C3 cyclical too much/nothing"),
 ch("r2", 2687732, "MANAGE", 1, "D less work"),
 ch("r2", c(4653052,13818800), "FLEX", 1, "D flexible with school/availability"), ch("r2", c(4653052,13818800), "JPAUT", 0, "D flexible with school/availability is scheduling"),
 ch("r2", 14021766, "VAR", 0, "D not supported"), ch("r2", 14021766, "LOWPRED", 0, "D not supported"),
 ch("r2", c(16011541,63622739,65949829), "UNDER", 0, "D pro coded as con"), ch("r2", c(16011541,63622739,65949829), "MANAGE", 1, "D manageable/low workload"),
 ch("r2", 23283696, "LH", 1, "D mandatory ot"), ch("r2", 23283696, "INFLEX", 1, "D mandatory ot"),
 ch("r2", 48935547, "UNDER", 1, "D too much work"),
 ch("r2", 83193687, "WLB", 0, "D wrong polarity"), ch("r2", 83193687, "WLC", 1, "D home time non existent"),
 ch("r2", 83601759, "UNDER", 1, "D/E29 heavy workload in pros box"),
 ch("r2", c(7046708,22367966,47179449,66645488), "UNDER", 1, "E1 fast paced"),
 ch("r2", c(2929396,9417972,12442351,13388285,14745029,16607339,17761562,20112935,20830860,28808465,48856551,51175266,52102784,56373719,62918503,76032937,81685721), "MANAGE", 1, "E10 laid back"),
 ch("r2", c(12064503,24073287,36968904,69768926,72541574,83445865,83896783,82315333,83497105), "MANAGE", 1, "E12 low stress / no pressure"),
 ch("r2", 61645910, "UNDES", 0, "E15 no off day = inflexible only"),
 ch("r2", 17125007, "PRED", 1, "E16 explicit 8-5"),
 ch("r2", c(23517643,28679744), "UNDER", 1, "E2 hectic"),
 ch("r2", 8022307, "WLB", 0, "E21 time away from work"), ch("r2", 8022307, "FLEX", 1, "E21 time away from work"),
 ch("r2", 83015990, "INFLEX", 1, "E26 called in"), ch("r2", 83015990, "LOWPRED", 1, "E26 called in"),
 ch("r2", 74317129, "UNDER", 1, "E27 burnout"),
 ch("r2", c(12405467,30483966,47671618), "INSUF", 0, "E3 not enough work = over-staffing"), ch("r2", c(12405467,30483966,47671618), "OVER", 1, "E3 not enough work = over-staffing"),
 ch("r2", c(2283747,19407233,33739531,49432554,64824392,71569499), "UNDER", 1, "E4 quotas/deadlines"),
 ch("r2", 13301880, "UNDER", 1, "E5 workload for pay"),
 ch("r2", 83742870, "JCAUT", 0, "E8 strict on down time"), ch("r2", 83742870, "INFLEX", 1, "E8 strict on down time"),
 ch("r2", 68112201, "OVER", 1, "E9 bored"),
 ## ---- dd ----
 ch("dd", c(53291,92580,66590,56954,84932,77838), "JCAUT", 1, "A micromanagement"),
 ch("dd", c(28520,21682,40308,2791,350), "JPAUT", 1, "A2 no micromanaging / independence"),
 ch("dd", c(20757,17182,37118,18035), "MANAGE", 1, "B busy praise"),
 ch("dd", 96961, "UNEVEN", 1, "C hours go to seniority"),
 ch("dd", c(80352,64601), "CYCLIC", 1, "C2 not consistently busy / unstable work flow"),
 ch("dd", c(12299,17157,40762,47053), "FLEX", 0, "D bare flexibility"), ch("dd", c(12299,17157,40762,47053), "JPAUT", 1, "D bare flexibility"),
 ch("dd", 13995, "JPAUT", 0, "D employee owned is not autonomy"),
 ch("dd", c(15015,28313,42582), "LH", 1, "D/E29 long hours in pros box"),
 ch("dd", 67835, "FLEX", 1, "D/E29 flexible in cons box"),
 ch("dd", 24532, "MANAGE", 0, "D works with school"), ch("dd", 24532, "FLEX", 1, "D works with school"),
 ch("dd", 28977, "PTO", 0, "D paid learning week is not PTO"),
 ch("dd", 30709, "MANAGE", 0, "D borderline hectic is not praise"),
 ch("dd", 31988, "PTO", 1, "D more leaves"),
 ch("dd", 34066, "DES", 0, "D weekend store is not desirable"),
 ch("dd", 36732, "MANAGE", 1, "D work is manageable"),
 ch("dd", 39012, "ADDHRS", 0, "D commission not hours"),
 ch("dd", 49809, "MANAGE", 1, "D staffing adequate"),
 ch("dd", 50273, "UNDER", 1, "D staffing too lean"),
 ch("dd", 74433, "UNDES", 0, "D holiday pay is compensation"), ch("dd", 74433, "NOPTO", 0, "D holiday pay is compensation"),
 ch("dd", 80803, "NOWFH", 1, "D required days in office"),
 ch("dd", 81632, "INSUF", 0, "D limited vacation dates"), ch("dd", 81632, "INFLEX", 1, "D limited vacation dates"),
 ch("dd", c(85521,87346,97163), "INFLEX", 1, "D forced/have to"),
 ch("dd", 87821, "UNDER", 0, "D pay complaint"),
 ch("dd", c(23214,26936,31910), "MANAGE", 1, "E10 laid back / relaxed"),
 ch("dd", c(9243,11233), "MANAGE", 1, "E12 low stress / no pressure"),
 ch("dd", c(76735,88098), "UNDER", 1, "E14 rush / busy holidays"),
 ch("dd", 40606, "REG", 1, "E16 set hours no nights/weekends"),
 ch("dd", 60967, "VAR", 1, "E17 seasonal hour cuts"),
 ch("dd", 52065, "LH", 0, "E19 exhausting hours = general"), ch("dd", 52065, "GENC", 1, "E19 exhausting hours = general"),
 ch("dd", 84000, "LH", 0, "E20 no overtime = insufficient"), ch("dd", 84000, "INSUF", 1, "E20 no overtime = insufficient"),
 ch("dd", 26584, "INFLEX", 1, "E29 mandatory weekend shifts in pros box"), ch("dd", 26584, "UNDES", 1, "E29 mandatory weekend shifts in pros box"),
 ch("dd", 51165, "INSUF", 1, "E3 not enough hours to do the job"), ch("dd", 84249, "UNDER", 1, "E3 not enough hours to do the job"),
 ch("dd", 63197, "LH", 0, "E5 too much work = under-staffing"), ch("dd", 63197, "UNDER", 1, "E5 too much work = under-staffing")
)
CH <- rbindlist(L)
texts <- data.table(id=c(57760272,77514082,74072045,25921932), text=c(
 "-no micromanaging -great culture -people there are amazing",
 "- lots of transferable skills learnt and developed which are admired in other job roles - always busy, rarely boring - reasonable pay",
 "- overworked, understaffed or overstaffed with nothing to do - mentally and physically taxing work",
 "- no career advancement - no yearly reviews/raises - occasionally stressful and understaffed - either too much to do or nothing to do - not much communication during closing process"))

## ---------- rebuild a workbook with two inserted columns ----------
rebuild <- function(fname, sheet, key) {
  h <- read_excel(paste0(d,fname), sheet=sheet, col_names=FALSE, n_max=2)
  x <- read_excel(paste0(d,fname), sheet=sheet, col_names=FALSE, skip=2, col_types="text")
  x <- as.data.frame(x, stringsAsFactors=FALSE)
  h1 <- as.character(unlist(h[1,])); h2 <- as.character(unlist(h[2,]))
  wbold <- loadWorkbook(paste0(d,fname)); oldw <- unlist(wbold$colWidths[[1]]); oldwidx <- as.integer(names(oldw))
  jc_idx <- which(grepl(pat["JCAUT"], h2))
  stopifnot(length(jc_idx)==1)
  ins <- function(v, at, vals) c(v[1:at], vals, v[(at+1):length(v)])
  h1 <- ins(h1, jc_idx, c(NA,NA)); h2 <- ins(h2, jc_idx, c(NEW_UNEVEN, NEW_CYCLIC))
  x <- cbind(x[,1:jc_idx], NA, NA, x[,(jc_idx+1):ncol(x)]); names(x) <- paste0("c",seq_len(ncol(x)))
  widths <- rep(8.43, ncol(x)); for (k in seq_along(oldw)) { j <- oldwidx[k]; jj <- if (j > jc_idx) j+2 else j; widths[jj] <- as.numeric(oldw[k]) }
  widths[jc_idx+1:2] <- widths[jc_idx]
  dimcols <- which(!is.na(h2)); ab <- sapply(dimcols, function(j){ k <- names(pat)[sapply(pat, function(p) grepl(p, h2[j]))]; stopifnot(length(k)==1); k })
  colidx <- setNames(dimcols, ab)
  for (j in dimcols) x[[j]] <- suppressWarnings(as.numeric(x[[j]]))
  idcol <- which(h1=="id"); x[[idcol]] <- as.numeric(x[[idcol]])
  tc <- sapply(c("schedule_related","job_control_related","not_related"), function(n) which(h1==n)); for (j in tc) x[[j]] <- suppressWarnings(as.numeric(x[[j]]))
  list(h1=h1, h2=h2, x=x, colidx=colidx, idcol=idcol, tc=tc, widths=widths, textcol=which(h1=="review_text"), sheet=sheet, fname=fname)
}
apply_ch <- function(W, key) {
  cc <- CH[file==key]; log <- list()
  for (i in seq_len(nrow(cc))) {
    r <- which(W$x[[W$idcol]]==cc$id[i]); stopifnot(length(r)==1)
    j <- W$colidx[[cc$dim[i]]]; old <- W$x[r,j]; oldv <- ifelse(is.na(old),0,old)
    newv <- cc$val[i]; W$x[r,j] <- if (newv==1) 1 else NA
    log[[i]] <- data.table(file=W$fname, id=cc$id[i], dim=W$h2[j], old=oldv, new=newv, changed=as.integer(oldv!=newv), reason=cc$cat[i])
  }
  W$log <- rbindlist(log); W
}
three_cat <- function(W, ids) {
  sc <- W$colidx[SC]; jc <- W$colidx[JC]
  for (id in ids) { r <- which(W$x[[W$idcol]]==id)
    s <- as.integer(any(W$x[r,sc]==1, na.rm=TRUE)); j <- as.integer(any(W$x[r,jc]==1, na.rm=TRUE))
    W$x[r, W$tc[1]] <- s; W$x[r, W$tc[2]] <- j; W$x[r, W$tc[3]] <- as.integer(s==0 & j==0) }
  W
}
save_wb <- function(W, out) {
  wb <- createWorkbook(); addWorksheet(wb, W$sheet)
  writeData(wb, 1, t(as.data.frame(W$h1)), startRow=1, colNames=FALSE)
  writeData(wb, 1, t(as.data.frame(W$h2)), startRow=2, colNames=FALSE)
  writeData(wb, 1, W$x, startRow=3, colNames=FALSE, keepNA=FALSE)
  n <- length(W$h1); j <- 1
  while (j <= n) {
    if (!is.na(W$h1[j]) && is.na(W$h2[j])) { mergeCells(wb, 1, cols=j, rows=1:2); j <- j+1 }
    else if (!is.na(W$h1[j])) { k <- j; while (k+1 <= n && is.na(W$h1[k+1]) && !is.na(W$h2[k+1])) k <- k+1; if (k>j) mergeCells(wb, 1, cols=j:k, rows=1); j <- k+1 }
    else j <- j+1
  }
  hs <- createStyle(textDecoration="bold", wrapText=TRUE, halign="center", valign="center", border="Bottom")
  addStyle(wb, 1, hs, rows=1:2, cols=1:n, gridExpand=TRUE)
  setColWidths(wb, 1, cols=1:n, widths=W$widths); setRowHeights(wb, 1, rows=2, heights=90)
  freezePane(wb, 1, firstActiveRow=3, firstActiveCol=W$textcol+1)
  saveWorkbook(wb, out, overwrite=TRUE)
}

R2 <- rebuild("candidates_round2_final.xlsx","to_code","r2"); R2 <- apply_ch(R2,"r2")
for (i in seq_len(nrow(texts))) { r <- which(R2$x[[R2$idcol]]==texts$id[i]); cat("restore text", texts$id[i], "was:", R2$x[r,R2$textcol], "\n"); R2$x[r,R2$textcol] <- texts$text[i] }
r2ids <- unique(CH[file=="r2", id]); filled <- r2ids[sapply(r2ids, function(id){ r <- which(R2$x[[R2$idcol]]==id); any(!is.na(unlist(R2$x[r,R2$tc]))) })]
cat("r2 changed rows with three-cat filled:", length(filled), "\n"); R2 <- three_cat(R2, filled)
DD <- rebuild("dataset_detailed_dimensions.xlsx","detailed","dd"); DD <- apply_ch(DD,"dd")
ddids <- unique(c(CH[file=="dd", id], 53291)); DD <- three_cat(DD, ddids)
save_wb(R2, paste0(d,"candidates_round2_final.xlsx")); save_wb(DD, paste0(d,"dataset_detailed_dimensions.xlsx"))

## three-categories mirror
wb3 <- loadWorkbook(paste0(d,"dataset_three_categories.xlsx")); x3 <- read.xlsx(paste0(d,"dataset_three_categories.xlsx"), sheet=1)
n3 <- 0
for (id in ddids) { r <- which(DD$x[[DD$idcol]]==id); r3 <- which(x3$id==id); stopifnot(length(r3)==1)
  vals <- unlist(DD$x[r, DD$tc]); old <- unlist(x3[r3, c("schedule_related","job_control_related","not_related")])
  if (any(is.na(old)) || any(old!=vals)) { n3 <- n3+1; for (k in 1:3) writeData(wb3, 1, vals[k], startRow=r3+1, startCol=which(names(x3)==c("schedule_related","job_control_related","not_related")[k])) } }
cat("three_categories rows updated:", n3, "\n"); saveWorkbook(wb3, paste0(d,"dataset_three_categories.xlsx"), overwrite=TRUE)

LOG <- rbind(R2$log, DD$log); fwrite(LOG, paste0(d,"coding_changes_2026-09-04.csv"))
cat("log rows:", nrow(LOG), " actually changed:", sum(LOG$changed), " already as requested:", sum(LOG$changed==0), "\n")
print(LOG[changed==0, .(file, id, dim, old, new)])
