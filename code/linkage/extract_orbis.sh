#!/bin/zsh
set -u
D="/Users/wj93/Library/CloudStorage/Box-Box/adhoc_orbis_historical/Text Data (as of Jun. 2024)/Descriptive-Data-Jun24-text"
S="/private/tmp/claude-501/-Users-wj93-Library-CloudStorage-Box-Box-Finance-and-Work-Flexibility/7bbc71e4-1316-4d6d-9e0c-6a82c046313d/scratchpad"
IDS="$S/orbis_ids.txt"
mkdir -p "$S/orbis"
run() {
  name="$1"; arch="$2"
  out="$S/orbis/${name}.tsv"
  [[ -s "$out" ]] && { echo "skip $name"; return; }
  echo "=== $name start $(date +%T)"
  7zz e -so "$arch" 2>/dev/null | awk -F'\t' -v OFS='\t' '
     NR==FNR { sub(/\r$/,"",$0); ids[$0]=1; next }
     { sub(/\r$/,"",$0); sub(/^\xef\xbb\xbf/,"",$1) }
     FNR==1 { print; next }
     ($1 in ids) { print }
  ' "$IDS" - > "$out"
  echo "=== $name done $(date +%T) rows=$(wc -l < "$out")"
}
run identifiers      "$D/Other-Identifiers/Identifiers.part01.rar"
run overviews        "$D/Overviews-StockInfo-TradeDescriptions-StatusHistory/Overviews.rar"
run addl_info        "$D/Additional-Company-Info/Additional_company_info.part1.rar"
run legal_info       "$D/Legal-Other-Advisors/Legal_info.part01.rar"
run contact_info     "$D/Contact-Info/Contact_info.part001.rar"
run bvd_name         "$D/BvD-Identifiers/BvD_ID_and_Name.part01.rar"
run trade_desc       "$D/Overviews-StockInfo-TradeDescriptions-StatusHistory/Trade_description.part001.rar"
run industry_class   "$D/Other-Identifiers/Industry-Classifications/Industry_classifications.part001.rar"
echo "ALL DONE $(date +%T)"
