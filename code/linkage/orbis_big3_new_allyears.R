library(data.table)

## -------------------------------------------------------------------
## CONFIG
## -------------------------------------------------------------------

# Pattern for the historical ownership link files
link_pattern <- "^Links_[0-9]{4}\\.txt$"

# Output files
out_main        <- "big3_shares_allyears.csv"
out_diag_blk    <- "big3_blackrock_diag_allyears.csv"
out_diag_vgd    <- "big3_vanguard_diag_allyears.csv"
out_diag_ss     <- "big3_statestreet_diag_allyears.csv"

# Big-3 GUO IDs (in this historical extract)
guo_blackrock <- "US320174431"   # BlackRock Finance Inc (GUO in 2016 data)
guo_vanguard  <- "US149144472L"  # Vanguard
guo_ss        <- "US042456637"   # State Street

# Exclude this custodian from State Street group
custodian_exclude <- "US041867445"

# GUO threshold flag: "50" or "25"
guo_threshold <- "50"            # you can switch to "25" if needed

## -------------------------------------------------------------------
## Helper for progress messages
## -------------------------------------------------------------------

msg <- function(...) cat(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "-", ..., "\n")

## -------------------------------------------------------------------
## Percentage parser (handles codes like MO, WO, etc.)
## -------------------------------------------------------------------

parse_pct <- function(x) {
  if (is.numeric(x)) return(x)
  x <- trimws(as.character(x))
  x[x == "" | x == "n.a."] <- NA_character_
  
  # Special codes per BvD mapping
  special_map <- c(
    "MO" = "50.01",
    "WO" = "98.00",
    "JO" = "50.00",
    "BR" = "100.00",
    "FC" = "100.00",
    "T"  = "100.00"
  )
  is_special <- x %in% names(special_map)
  x[is_special] <- special_map[x[is_special]]
  
  # >X, <X, +/-X patterns
  gt_idx <- grepl("^>[0-9.]+$", x)
  lt_idx <- grepl("^<[0-9.]+$", x)
  eq_idx <- grepl("^\\+/-[0-9.]+$", x)
  
  if (any(gt_idx)) {
    base <- as.numeric(sub("^>", "", x[gt_idx]))
    x[gt_idx] <- sprintf("%.2f", base + 0.01)
  }
  if (any(lt_idx)) {
    base <- as.numeric(sub("^<", "", x[lt_idx]))
    x[lt_idx] <- sprintf("%.2f", base - 0.01)
  }
  if (any(eq_idx)) {
    base <- as.numeric(sub("^\\+/-", "", x[eq_idx]))
    x[eq_idx] <- sprintf("%.2f", base)
  }
  
  suppressWarnings(as.numeric(x))
}

## -------------------------------------------------------------------
## Helper: build Big-3 group membership from GUO column
## -------------------------------------------------------------------

build_big3_groups <- function(dt, guo_col) {
  subs_guo <- unique(dt[, .(`Subsidiary BvD ID`, guo = get(guo_col))])
  
  blackrock_ids <- unique(subs_guo[guo == guo_blackrock, `Subsidiary BvD ID`])
  vanguard_ids  <- unique(subs_guo[guo == guo_vanguard,  `Subsidiary BvD ID`])
  ss_ids        <- unique(subs_guo[guo == guo_ss,        `Subsidiary BvD ID`])
  
  # Ensure GUO IDs themselves are in the group
  blackrock_ids <- unique(c(blackrock_ids, guo_blackrock))
  vanguard_ids  <- unique(c(vanguard_ids,  guo_vanguard))
  ss_ids        <- unique(c(ss_ids,        guo_ss))
  
  # Exclude specific custodian from State Street
  ss_ids <- setdiff(ss_ids, custodian_exclude)
  
  list(
    blackrock_ids = blackrock_ids,
    vanguard_ids  = vanguard_ids,
    ss_ids        = ss_ids
  )
}

## -------------------------------------------------------------------
## Helper: process one file and return main + diag tables
## -------------------------------------------------------------------

process_links_file <- function(file) {
  year <- as.integer(sub("^Links_([0-9]{4})\\.txt$", "\\1", file))
  msg("Processing file:", file, "for year", year)
  
  needed_cols <- c(
    "Subsidiary BvD ID",
    "Shareholder BvD ID",
    "Direct %",
    "Direct % (only figures)",
    "Total %",
    "Total % (only figures)",
    "Information date",
    "Type of relation",
    "Active/archived",
    "GUO 50",
    "GUO 25"
  )
  
  dt <- fread(
    file,
    sep = "\t",
    header = TRUE,
    check.names = FALSE,
    select = needed_cols,
    na.strings = c("n.a.", "NA", "")
  )
  
  # Keep original line numbers (offset +1 for header)
  dt[, file_row := .I + 1L]
  dt[, source_file := file]
  
  msg("  Rows read:", nrow(dt))
  
  # Build group membership using chosen GUO threshold
  guo_col <- if (guo_threshold == "50") "GUO 50" else "GUO 25"
  groups  <- build_big3_groups(dt, guo_col)
  blackrock_ids <- groups$blackrock_ids
  vanguard_ids  <- groups$vanguard_ids
  ss_ids        <- groups$ss_ids
  
  msg("  Group sizes - BlackRock:", length(blackrock_ids),
      "Vanguard:", length(vanguard_ids),
      "State Street:", length(ss_ids))
  
  # Parse percentages
  msg("  Parsing percentage columns...")
  dt[, `Direct %`                := parse_pct(`Direct %`)]
  dt[, `Direct % (only figures)` := parse_pct(`Direct % (only figures)`)]
  dt[, `Total %`                 := parse_pct(`Total %`)]
  dt[, `Total % (only figures)`  := parse_pct(`Total % (only figures)`)]
  
  # Use "only figures" when available; otherwise fall back
  # direct_pct: for subsidiary sums
  dt[, direct_pct := fifelse(!is.na(`Direct % (only figures)`),
                             `Direct % (only figures)`,
                             `Direct %`)]
  # total_pct: for GUO Total % when shareholder is GUO entity
  dt[, total_pct  := fifelse(!is.na(`Total % (only figures)`),
                             `Total % (only figures)`,
                             `Total %`)]
  
  # Normalize Information date as integer (YYYYMMDD)
  dt[, Information_int := as.integer(`Information date`)]
  
  # Branch mapping for target companies
  dt[, company_id := sub("-B[0-9]+$", "", `Subsidiary BvD ID`)]
  
  # Filter to SHH + active rows
  msg("  Filtering to SHH + active rows...")
  shh <- dt[`Type of relation` == "SHH" & `Active/archived` == "active"]
  msg("  SHH+active rows:", nrow(shh))
  
  # We only care about rows where shareholder is in any Big-3 group
  any_big3_ids <- unique(c(blackrock_ids, vanguard_ids, ss_ids))
  shh <- shh[`Shareholder BvD ID` %in% any_big3_ids]
  msg("  SHH rows with Big-3 shareholders:", nrow(shh))
  
  if (nrow(shh) == 0L) {
    msg("  No Big-3 holdings in this file.")
    return(list(
      main = data.table(),
      diag_blk = data.table(),
      diag_vgd = data.table(),
      diag_ss  = data.table()
    ))
  }
  
  # For each (company, shareholder) keep the *last* row in the year
  setorder(shh, company_id, `Shareholder BvD ID`, Information_int)
  shh_latest <- shh[, .SD[.N], by = .(company_id, `Shareholder BvD ID`)]
  shh_latest[, year := year]
  
  # Helper: compute per-group ownership + diagnostics
  compute_group <- function(group_ids, guo_id, label) {
    dt_g <- shh_latest[`Shareholder BvD ID` %in% group_ids]
    if (nrow(dt_g) == 0L) {
      return(list(main = data.table(), diag = data.table()))
    }
    
    # GUO rows: shareholder == guo_id with non-missing total_pct
    guo_rows <- dt_g[`Shareholder BvD ID` == guo_id & !is.na(total_pct)]
    guo_main <- data.table()
    diag_guo <- data.table()
    if (nrow(guo_rows) > 0L) {
      # If more than one GUO row per company (shouldn't), keep highest total_pct
      guo_main <- guo_rows[, .(
        pct = max(total_pct, na.rm = TRUE)
      ), by = .(company_id, year)]
      
      diag_guo <- guo_rows[
        , .(
          company_id,
          year,
          big3_label   = label,
          method       = "GUO_total",
          big3_pct     = total_pct,
          shareholder  = `Shareholder BvD ID`,
          direct_pct,
          total_pct,
          Information_date = `Information date`,
          Information_int,
          source_file,
          file_row
        )
      ]
    }
    
    # Companies already covered by GUO_total
    covered_companies <- if (nrow(guo_main) > 0L) guo_main$company_id else character()
    
    # Sum of direct_pct across all group subsidiaries (including GUO if not covered)
    subs_rows <- dt_g[!is.na(direct_pct) & !(company_id %in% covered_companies)]
    sum_main  <- data.table()
    diag_subs <- data.table()
    
    if (nrow(subs_rows) > 0L) {
      sum_main <- subs_rows[
        , .(pct = sum(direct_pct, na.rm = TRUE)),
        by = .(company_id, year)
      ]
      
      if (nrow(sum_main) > 0L) {
        # Attach final pct to each component row for diagnostics
        sum_main[, tmp_key := paste(company_id, year)]
        subs_rows[, tmp_key := paste(company_id, year)]
        
        joined <- sum_main[subs_rows, on = "tmp_key", nomatch = 0L]
        
        diag_subs <- joined[
          , .(
            company_id,
            year,
            big3_label   = label,
            method       = "sum_direct_component",
            big3_pct     = pct,   # final pct for that company/year
            shareholder  = `Shareholder BvD ID`,
            direct_pct,
            total_pct,
            Information_date = `Information date`,
            Information_int,
            source_file,
            file_row
          )
        ]
      }
    }
    
    # Combine GUO_main and sum_main into group main
    group_main <- rbindlist(list(guo_main, sum_main), use.names = TRUE, fill = TRUE)
    if (nrow(group_main) > 0L) {
      # If duplicates (shouldn't), keep max pct
      group_main <- group_main[, .(pct = max(pct, na.rm = TRUE)), by = .(company_id, year)]
    }
    
    group_diag <- rbindlist(list(diag_guo, diag_subs), use.names = TRUE, fill = TRUE)
    
    list(main = group_main, diag = group_diag)
  }
  
  msg("  Computing Big-3 ownership and diagnostics for year", year, "...")
  
  res_blk <- compute_group(blackrock_ids, guo_blackrock, "BlackRock")
  res_vgd <- compute_group(vanguard_ids,  guo_vanguard,  "Vanguard")
  res_ss  <- compute_group(ss_ids,        guo_ss,        "StateStreet")
  
  # Rename pct columns for main
  blk_main <- res_blk$main
  vgd_main <- res_vgd$main
  ss_main  <- res_ss$main
  
  if (nrow(blk_main)) setnames(blk_main, "pct", "blackrock_pct")
  if (nrow(vgd_main)) setnames(vgd_main, "pct", "vanguard_pct")
  if (nrow(ss_main))  setnames(ss_main,  "pct", "statestreet_pct")
  
  # Merge to one per-file main table (company_id, year, 3 pct columns)
  main_list <- list(blk_main, vgd_main, ss_main)
  main_list <- main_list[sapply(main_list, nrow) > 0L]
  
  if (length(main_list) == 0L) {
    main_dt <- data.table()
  } else {
    main_dt <- Reduce(
      function(x, y) merge(x, y, by = c("company_id", "year"), all = TRUE),
      main_list
    )
  }
  
  if (nrow(main_dt) > 0L) {
    # Ensure all three pct columns exist
    if (!"blackrock_pct"   %in% names(main_dt)) main_dt[, blackrock_pct   := 0]
    if (!"vanguard_pct"    %in% names(main_dt)) main_dt[, vanguard_pct    := 0]
    if (!"statestreet_pct" %in% names(main_dt)) main_dt[, statestreet_pct := 0]
    
    # Replace NA with 0 (no holding)
    main_dt[is.na(blackrock_pct),   blackrock_pct   := 0]
    main_dt[is.na(vanguard_pct),    vanguard_pct    := 0]
    main_dt[is.na(statestreet_pct), statestreet_pct := 0]
  }
  
  list(
    main     = main_dt,
    diag_blk = res_blk$diag,
    diag_vgd = res_vgd$diag,
    diag_ss  = res_ss$diag
  )
}

## -------------------------------------------------------------------
## MAIN LOOP OVER ALL LINKS_*.TXT FILES
## -------------------------------------------------------------------

files <- list.files(pattern = link_pattern)
files <- sort(files)

if (length(files) == 0L) {
  stop("No Links_XXXX.txt files found in working directory.")
}

msg("Found", length(files), "files:", paste(files, collapse = ", "))

main_all     <- vector("list", length(files))
diag_blk_all <- vector("list", length(files))
diag_vgd_all <- vector("list", length(files))
diag_ss_all  <- vector("list", length(files))

for (i in seq_along(files)) {
  res <- process_links_file(files[i])
  main_all[[i]]     <- res$main
  diag_blk_all[[i]] <- res$diag_blk
  diag_vgd_all[[i]] <- res$diag_vgd
  diag_ss_all[[i]]  <- res$diag_ss
}

msg("Combining results across all years...")

main_all_dt <- rbindlist(main_all, use.names = TRUE, fill = TRUE)

if (nrow(main_all_dt)) {
  # If any duplicates company-year across files (shouldn't), keep last
  setorder(main_all_dt, company_id, year)
  main_all_dt <- main_all_dt[, .SD[.N], by = .(company_id, year)]
  
  # Ensure zeros instead of NA
  if (!"blackrock_pct"   %in% names(main_all_dt)) main_all_dt[, blackrock_pct   := 0]
  if (!"vanguard_pct"    %in% names(main_all_dt)) main_all_dt[, vanguard_pct    := 0]
  if (!"statestreet_pct" %in% names(main_all_dt)) main_all_dt[, statestreet_pct := 0]
  
  main_all_dt[is.na(blackrock_pct),   blackrock_pct   := 0]
  main_all_dt[is.na(vanguard_pct),    vanguard_pct    := 0]
  main_all_dt[is.na(statestreet_pct), statestreet_pct := 0]
}

diag_blk_dt <- rbindlist(diag_blk_all, use.names = TRUE, fill = TRUE)
diag_vgd_dt <- rbindlist(diag_vgd_all, use.names = TRUE, fill = TRUE)
diag_ss_dt  <- rbindlist(diag_ss_all,  use.names = TRUE, fill = TRUE)

msg("Writing main output to:", out_main)
fwrite(main_all_dt[order(company_id, year)], out_main)

msg("Writing diagnostics:")
msg("  BlackRock ->", out_diag_blk)
fwrite(diag_blk_dt[order(company_id, year, file_row)], out_diag_blk)

msg("  Vanguard  ->", out_diag_vgd)
fwrite(diag_vgd_dt[order(company_id, year, file_row)], out_diag_vgd)

msg("  State St  ->", out_diag_ss)
fwrite(diag_ss_dt[order(company_id, year, file_row)], out_diag_ss)

msg("Done.")

