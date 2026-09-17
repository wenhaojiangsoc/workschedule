#!/bin/zsh
name="$1"; arch="$2"
S="/private/tmp/claude-501/-Users-wj93-Library-CloudStorage-Box-Box-Finance-and-Work-Flexibility/7bbc71e4-1316-4d6d-9e0c-6a82c046313d/scratchpad"
out="$S/orbis/${name}.tsv"
echo "start $(date +%T) $name"
unar -o - "$arch" 2>/dev/null | awk -F'\t' -v OFS='\t' '
   NR==FNR { sub(/\r$/,"",$0); ids[$0]=1; next }
   { sub(/\r$/,"",$0) }
   FNR==1 { print; next }
   ($1 in ids) { print }
' "$S/orbis_ids.txt" - > "$out"
echo "done $(date +%T) $name rows=$(wc -l < "$out")"
