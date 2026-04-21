#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


FILES = {
    "GSE92742": {
        "gene_info": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_gene_info.txt.gz",
        "cell_info": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_cell_info.txt.gz",
        "sig_info": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_sig_info.txt.gz",
        "level5": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx.gz",
    },
    "GSE70138": {
        "gene_info": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE70nnn/GSE70138/suppl/GSE70138_Broad_LINCS_gene_info_2017-03-06.txt.gz",
        "cell_info": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE70nnn/GSE70138/suppl/GSE70138_Broad_LINCS_cell_info_2017-04-28.txt.gz",
        "sig_info": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE70nnn/GSE70138/suppl/GSE70138_Broad_LINCS_sig_info_2017-03-06.txt.gz",
        "level5": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE70nnn/GSE70138/suppl/GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328_2017-03-06.gctx.gz",
    },
}


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, target.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def maybe_decompress(path: Path) -> Path:
    if path.suffix != ".gz":
        return path
    output = path.with_suffix("")
    with gzip.open(path, "rb") as src, output.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    return output


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="Download explicit public LINCS L1000 GEO files.")
    parser.add_argument(
        "--accession",
        action="append",
        choices=sorted(FILES.keys()),
        default=["GSE92742", "GSE70138"],
        help="GEO accession(s) to download.",
    )
    parser.add_argument(
        "--include-level5",
        action="store_true",
        help="Download the very large Level-5 GCTX files in addition to metadata tables.",
    )
    parser.add_argument(
        "--decompress",
        action="store_true",
        help="Decompress .gz files after download.",
    )
    parser.add_argument(
        "--output-root",
        default="data/drug perturbation/LINCS_public",
        help="Destination directory.",
    )
    args = parser.parse_args()

    output_root = (ROOT / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root).resolve()
    for accession in args.accession:
        for key, url in FILES[accession].items():
            if key == "level5" and not args.include_level5:
                continue
            target = output_root / accession / Path(url).name
            print(f"downloading={url} target={_display_path(target)}")
            download(url, target)
            if args.decompress:
                decompressed = maybe_decompress(target)
                print(f"decompressed={_display_path(decompressed)}")


if __name__ == "__main__":
    main()
