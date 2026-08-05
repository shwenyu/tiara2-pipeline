#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ncbi_pipeline.py  —  Tiara2 训练数据获取/QC/下载/split 统一管线（无 SQL 版）

设计要点（对应“总体结论”）：
  * 不用数据库：canonical 总表用 TSV，所有中间/分类表都从总表派生（字段统一）。
  * 数据文件按文件夹名分类：raw/<class>/<split>/<accession>/...
  * 一个 CLI，多个幂等、可断点重跑的 stage。
  * 先全量 metadata → metadata QC → 分层选择 → 并行下载 → MD5/gzip/FASTA 校验 → backup。
  * 不在 inventory 阶段删候选；accepted / quarantine / rejected 全保留。
  * split 按 ANI group → species_taxid，避免物种/同义组泄露。
  * JGI isolate、Virus 与 MAG 采用显式 dataset_role；测试库绝不混入训练 raw/。
  * 下载先取 md5checksums.txt → 定位真实文件名 + expected MD5。
  * 真断点续传（.part）+ resume 重校 MD5。
  * organelle 流式下载 bulk FASTA+GBFF，从 GBFF 解析 taxid/organism/organelle/length。

依赖：仅 Python 3.8+ 标准库 + 外部命令 wget/curl 或 aria2c（可选）。
用法：
    python3 ncbi_pipeline.py <stage> [--config config.json]
    stage ∈ {fetch-metadata, build-index, apply-qc, select,
             plan-downloads, download, verify, export, split,
             fetch-organelle, run, status}
"""

import argparse
import concurrent.futures as cf
import csv
import gzip
import hashlib
import itertools
import json
import math
import os
import queue
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

csv.field_size_limit(10 * 1024 * 1024)

# --------------------------------------------------------------------------- #
# 0. 配置
# --------------------------------------------------------------------------- #

DEFAULT_CONFIG = {
    "storage": {
        "primary_root": "/data/shouhanyu/Tiara2",
        "backup_root": "/backup/shouhanyu/Tiara2",
        "enable_backup": False,
    },
    "asm_sources": ["refseq", "genbank"],
    "external_sources": {
        "jgi": {
            "enabled": False,
            "workbooks": [],
            "force_reimport": False,
            # 向后兼容旧的单表 TSV/CSV adapter。
            "metadata_path": "",
            "delimiter": "auto",
            "column_map": {},
        },
    },
    "reports_dir": "/data/ncbi_reports_all",
    "groups": [
        {"group": "bacteria", "klass": "bacteria", "kind": "asm", "label": "prok", "supergroup": "Bacteria"},
        {"group": "archaea", "klass": "archaea", "kind": "asm", "label": "prok", "supergroup": "Archaea"},
        {"group": "fungi", "klass": "nuclear_euk", "kind": "asm", "label": "euk", "supergroup": "Opisthokonta-Fungi"},
        {"group": "protozoa", "klass": "nuclear_euk", "kind": "asm", "label": "euk", "supergroup": "Protist(SAR/Excavata/Amoebozoa)"},
        {"group": "plant", "klass": "nuclear_euk", "kind": "asm", "label": "euk", "supergroup": "Archaeplastida"},
        {"group": "invertebrate", "klass": "nuclear_euk", "kind": "asm", "label": "euk", "supergroup": "Opisthokonta-Metazoa"},
        {"group": "vertebrate_other", "klass": "nuclear_euk", "kind": "asm", "label": "euk", "supergroup": "Opisthokonta-Metazoa"},
        {"group": "vertebrate_mammalian", "klass": "nuclear_euk", "kind": "asm", "label": "euk", "supergroup": "Host-Mammalia"},
        {"group": "mitochondrion", "klass": "mitochondria", "kind": "organelle", "label": "euk", "supergroup": "Organelle-Mito"},
        {"group": "plastid", "klass": "plastid", "kind": "organelle", "label": "euk", "supergroup": "Organelle-Plastid"},
        {"group": "viral", "klass": "virus", "kind": "asm", "dataset_role": "test_only"},
    ],
    "filters": {
        "require_latest": True,
        "require_full": True,
        "exclude_mag": False,
        "mag_policy": "test_only",
        "fetch_before": "",
        "fetch_after": "",
        "drop_excluded_from_refseq_anomalies": True,
    },
    "select": {
        "one_isolate_per_species": True,
        "one_mag_per_species": True,
        "feature_metadata_path": "",
        "unknown_species_policy": "unique_entity",
        "weights": {
            "log10_n50": 12.0,
            "refseq": 8.0,
            "third_generation": 6.0,
            "reference_category": 4.0,
            "type_material": 2.0,
            "assembly_level": 2.0,
            "qc_score": 1.0
        },
        "per_class_cap": {}
    },
    "split": {"test_pct": 20, "seed": 42},
    "organelle": {"max_per_species": 5, "min_len": 5000,
                   "fetch_with_metadata": True},
    "virus": {
        "enabled": True,
        "include_assembly_summary": True,
        "include_genome_report_segments": True,
        "genome_report_url": "https://ftp.ncbi.nlm.nih.gov/genomes/GENOME_REPORTS/viruses.txt",
        "min_len": 1000,
        # 高质量、规模受控的独立病毒 benchmark；不进入训练集。
        "selection_mode": "high_quality_test",
        "allowed_genome_types": ["dsDNA", "ssDNA"],
        "require_taxid": True,
        "prefer_refseq": True,
        "one_genome_per_species": True,
        "max_genomes": 20000,
        "max_n_fraction": 0.05,
        "max_other_fraction": 0.01,
        "keep_all_segments_of_selected_genome": True,
    },
    # NCBI E-utilities：API key 只从环境变量读取，禁止写入配置/日志。
    # virus_sequence 会批量走 EFetch；assembly 大文件仍走 HTTPS。
    "ncbi_api": {
        "enabled": True,
        "api_key_env": "NCBI_API_KEY",
        "efetch_url": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        "tool": "tiara2_pipeline",
        "email_env": "NCBI_EMAIL",
        "max_requests_per_second": 8.0,
        "batch_size": 100,
        "workers": 4,
        "connect_timeout": 20,
        "read_timeout": 180,
        "max_retries": 3,
        "retry_delay": 3,
        "fallback_to_sviewer": True,
    },
    "qc": {
        "size_low_ratio": 0.5,
        "size_high_ratio": 2.0,
        "contamination_frac_quarantine": 0.05,
    },
    "download": {
        "concurrency": 16,
        "connections_per_file": 1,
        # 首轮快速扫过；failed 会在下次 download 自动 resume/retry。
        "max_retries": 1,
        "retry_delay": 2,
        "timeout": 300,
        "summary_max_age_days": 7,
        "status_flush_every": 500,
        # size-aware 分流调度
        "size_aware": True,
        "enable_ftp_fallback": False,
        "small_max_mb": 20,
        "medium_max_mb": 200,
        "small_concurrency": 20,
        "medium_concurrency": 8,
        "large_concurrency": 6,
        "small_timeout": 90,
        "medium_timeout": 180,
        "large_timeout": 300,
        "small_connections_per_file": 1,
        "medium_connections_per_file": 1,
        "large_connections_per_file": 1,
    },
}

FTP = "https://ftp.ncbi.nlm.nih.gov"

# canonical 总表列（所有分类表/下游都从这里派生）
MASTER_COLUMNS = [
    "entity_id", "entity_type", "klass", "group_name", "source",
    "dataset_role", "evaluation_collection",
    "source_native_id", "metadata_origin", "source_record_url",
    "is_mag", "mag_evidence", "species_taxid_method",
    # JGI GOLD relational provenance（Analysis Project -> Project -> Organism）
    "gold_analysis_project_id", "gold_project_ids", "gold_organism_id",
    "img_taxon_id", "sequencing_strategy", "project_status", "sequencing_status",
    "jgi_data_utilization_status", "gold_phylum", "organism_cultured", "type_strain",
    "assembly_accession", "paired_accession", "paired_comparison",
    "taxid", "species_taxid", "organism_name", "infraspecific_name",
    "bioproject", "biosample", "wgs_master",
    "refseq_category", "version_status", "assembly_level", "genome_rep", "assembly_type",
    "seq_rel_date", "asm_name", "ftp_path",
    "genome_size", "genome_size_ungapped", "gc_percent",
    "scaffold_count", "contig_count", "scaffold_n50", "contig_n50",
    "n50_value", "n50_metric", "sequencing_technology", "is_third_generation",
    "relation_to_type_material", "excluded_from_refseq",
    # 非 NCBI 来源统一下载接口
    "download_url", "expected_md5", "source_filename", "source_file_size",
    # organelle 专用
    "sequence_accession", "organelle_type", "sequence_length",
    "refseq_source_file", "definition",
    # virus inventory 专用
    "virus_group", "virus_subgroup", "segment_name", "host",
    "virus_genome_id", "virus_reported_genome_length", "is_refseq_viral",
]

QC_COLUMNS = [
    "entity_id", "qc_status", "score", "flags", "split_group_id",
]


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print("[{}] {}".format(ts, msg), flush=True)


def load_config(path):
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            user = json.load(fh)
        _deep_update(cfg, user)
        # 向后兼容旧配置：只有 asm_source 时按单来源运行。
        if "asm_source" in user and "asm_sources" not in user:
            cfg["asm_sources"] = [user["asm_source"]]
    _asm_sources(cfg)  # 尽早校验，避免跑到一半才失败
    return cfg


def _asm_sources(cfg):
    """返回已校验、去重且保持顺序的 assembly 来源列表。"""
    raw = cfg.get("asm_sources")
    if raw is None:
        raw = [cfg.get("asm_source", "refseq")]
    if isinstance(raw, str):
        raw = [raw]
    allowed = {"refseq", "genbank"}
    out = []
    for value in raw:
        src = str(value).strip().lower()
        if src not in allowed:
            raise ValueError("asm_sources 只支持 refseq/genbank，收到: {}".format(value))
        if src not in out:
            out.append(src)
    if not out:
        raise ValueError("asm_sources 不能为空")
    return out


def _deep_update(base, extra):
    for k, v in extra.items():
        if k == "comment":
            continue
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


class Paths:
    """统一管理存储结构（对齐下游 03_04_prepare_dataset）。

    <root>/
      metadata/assembly_summary/   原始 summary 缓存（只读源）
      index/                       canonical TSV 索引（唯一真源）
      catalog/                     派生视图：classes/qc/accessions/manifests
      store/assembly|organelle/    内容寻址的实际下载文件（稳定）
      raw/<group>/<split>/         ★ 下游输入（symlink → store）
      select/groups.tsv,taxid.tsv  ★ 下游输入
      logs/  tmp/
    """

    def __init__(self, cfg):
        self.root = cfg["storage"]["primary_root"]
        self.backup_root = cfg["storage"]["backup_root"]
        self.index = os.path.join(self.root, "index")
        self.master = self.index                             # canonical 总表目录
        self.meta = os.path.join(self.root, "metadata", "assembly_summary")
        self.reports_meta = os.path.join(self.root, "metadata", "genome_reports")
        self.catalog = os.path.join(self.root, "catalog")
        self.classes = os.path.join(self.catalog, "classes")
        self.qc = os.path.join(self.catalog, "qc")
        self.accessions = os.path.join(self.catalog, "accessions")
        self.manifests = os.path.join(self.catalog, "manifests")
        self.store = os.path.join(self.root, "store")
        self.raw = os.path.join(self.root, "raw")            # 下游 group/split 树
        self.evaluation = os.path.join(self.root, "evaluation")
        self.select = os.path.join(self.root, "select")
        self.logs = os.path.join(self.root, "logs")
        self.tmp = os.path.join(self.root, "tmp")
        for d in (self.index, self.meta, self.reports_meta, self.classes, self.qc,
                  self.accessions, self.manifests,
                  os.path.join(self.store, "assembly"),
                  os.path.join(self.store, "organelle"),
                  os.path.join(self.store, "sequence"),
                  self.raw, self.evaluation, self.select, self.logs, self.tmp):
            os.makedirs(d, exist_ok=True)

    @property
    def master_tsv(self):
        return os.path.join(self.index, "all_entities.tsv")

    @property
    def qc_tsv(self):
        return os.path.join(self.index, "qc_flags.tsv")

    @property
    def status_tsv(self):
        return os.path.join(self.index, "download_status.tsv")


# --------------------------------------------------------------------------- #
# 通用工具
# --------------------------------------------------------------------------- #

def which(prog):
    return shutil.which(prog) is not None


def read_tsv(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def write_tsv(path, rows, columns):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, delimiter="\t",
                           extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    os.replace(tmp, path)
    log("  wrote {} rows -> {}".format(len(rows), path))


def md5_of(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def gzip_ok(path):
    try:
        with gzip.open(path, "rb") as fh:
            while fh.read(1 << 20):
                pass
        return True
    except Exception:
        return False


def fasta_sane(path, min_records=1):
    """至少一个 header + 一段非空序列。"""
    try:
        headers = 0
        seq_chars = 0
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith(">"):
                    headers += 1
                    if headers >= min_records and seq_chars > 0:
                        return True
                else:
                    seq_chars += len(line.strip())
        return headers >= min_records and seq_chars > 0
    except Exception:
        return False


def fasta_composition_sane(path, max_n_fraction=0.05, max_other_fraction=0.01):
    """Single-pass FASTA syntax/composition QC for the curated virus benchmark."""
    try:
        opener = gzip.open if path.endswith(".gz") else open
        headers = acgt = n_count = other = 0
        with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith(">"):
                    headers += 1
                    continue
                seq = line.strip().upper()
                for ch in seq:
                    if ch in "ACGT":
                        acgt += 1
                    elif ch == "N":
                        n_count += 1
                    elif ch.isalpha():
                        # Other IUPAC ambiguity codes are retained but bounded.
                        other += 1
        total = acgt + n_count + other
        return (headers > 0 and total > 0
                and n_count / total <= float(max_n_fraction)
                and other / total <= float(max_other_fraction))
    except Exception:
        return False


def http_download(url, dest, timeout, retries, delay, resume=True,
                  connections_per_file=1):
    """下载到 dest（支持 .part 断点续传）。成功返回 True。

    timeout 只作为「网络静默超时」交给 wget/aria2c/curl 自身；不再对整个进程
    设置 subprocess timeout，避免正常传输中的大文件被强杀。
    """
    part = dest + ".part"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    conns = max(1, int(connections_per_file or 1))
    for attempt in range(1, retries + 1):
        try:
            if which("aria2c"):
                cmd = ["aria2c", "-c",
                       "-x", str(conns), "-s", str(conns),
                       "--file-allocation=none", "--max-tries=1",
                       "--connect-timeout=30",
                       "--timeout={}".format(timeout),
                       "--retry-wait={}".format(delay),
                       "--auto-file-renaming=false",
                       "--console-log-level=warn",
                       "-d", os.path.dirname(part), "-o",
                       os.path.basename(part), url]
            elif which("wget"):
                cmd = ["wget", "-q", "--tries=1", "-T", str(timeout)]
                cmd += ["-c"] if resume else ["--no-continue"]
                cmd += ["-O", part, url]
            elif which("curl"):
                # --speed-time/--speed-limit: 若传输速率长时间过低才失败，
                # 不限制大文件的总下载时长。
                cmd = ["curl", "-sS", "-L",
                       "--connect-timeout", "30",
                       "--speed-time", str(timeout),
                       "--speed-limit", "1024"]
                if resume and os.path.exists(part):
                    cmd += ["-C", "-"]
                cmd += ["-o", part, url]
            else:
                raise RuntimeError("未找到 aria2c/wget/curl")
            # 关键：不再设置 subprocess timeout；由工具自身的网络超时负责。
            rc = subprocess.run(cmd).returncode
            if rc == 0 and os.path.exists(part) and os.path.getsize(part) > 0:
                os.replace(part, dest)
                return True
            log("    download failed ({}/{}) rc={} url={}".format(
                attempt, retries, rc, url))
        except Exception as exc:  # noqa
            log("    download err ({}/{}): {}".format(attempt, retries, exc))
        if attempt < retries:
            time.sleep(delay)
    return False


def fetch_text(url, dest, timeout=120, retries=3, delay=5):
    """下载小文本文件（summary / md5checksums）。"""
    return http_download(url, dest, timeout, retries, delay, resume=False)


# --------------------------------------------------------------------------- #
# size-aware 下载调度：根据预估文件大小分流（small / medium / large）
# --------------------------------------------------------------------------- #

_MB = 1024 * 1024


def _safe_int(value, default=0):
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def download_size_class(rec, cfg=None):
    """根据 rec 的 size_hint / file_type 判定属于 small / medium / large。

    - virus 单序列（viral_segment_fna）永远归为 small。
    - size_hint <= 0（未知大小）保守归为 medium，避免占用 large 通道。
    - 阈值可通过 config.download.small_max_mb / medium_max_mb 调整。
    注：genome_size / source_file_size 多为未压缩大小，仅作为量级提示。
    """
    if rec.get("file_type") == "viral_segment_fna":
        return "small"
    dl = (cfg or {}).get("download", {}) if cfg else {}
    small_max = _safe_int(dl.get("small_max_mb"), 20) * _MB
    medium_max = _safe_int(dl.get("medium_max_mb"), 200) * _MB
    size = _safe_int(rec.get("size_hint"), 0)
    if size <= 0:
        return "medium"
    if size <= small_max:
        return "small"
    if size <= medium_max:
        return "medium"
    return "large"


def download_profile(rec, cfg):
    """返回当前 rec 的下载参数（size_class / timeout / connections_per_file）。"""
    dl = cfg.get("download", {})
    size_class = download_size_class(rec, cfg)
    defaults = {
        "small": (dl.get("small_timeout", 120), dl.get("small_connections_per_file", 1)),
        "medium": (dl.get("medium_timeout", 300), dl.get("medium_connections_per_file", 1)),
        "large": (dl.get("large_timeout", 600), dl.get("large_connections_per_file", 1)),
    }
    timeout, conns = defaults[size_class]
    return {
        "size_class": size_class,
        "timeout": _safe_int(timeout, dl.get("timeout", 300)),
        "connections_per_file": max(1, _safe_int(conns, 1)),
    }


# --------------------------------------------------------------------------- #
# 1. fetch-metadata : 下载各 group 的 assembly_summary
# --------------------------------------------------------------------------- #

def _summary_sane(path, max_lines=500):
    """快速验证 summary 不是空文件、HTML 错误页或只有表头的残缺文件。"""
    if not os.path.exists(path) or os.path.getsize(path) < 100:
        return False
    header_seen = False
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh):
                if i >= max_lines:
                    break
                if line.startswith("#") and "assembly_accession" in line:
                    header_seen = True
                    continue
                if header_seen and not line.startswith("#"):
                    acc = line.split("\t", 1)[0]
                    if re.match(r"^GC[AF]_\d+\.\d+$", acc):
                        return True
    except OSError:
        return False
    return False

def stage_fetch_metadata(cfg, paths):
    log("== fetch-metadata ==")
    sources = _asm_sources(cfg)
    log("  assembly sources: {}".format(", ".join(sources)))
    max_age = cfg["download"]["summary_max_age_days"] * 86400
    for src in sources:
        for g in cfg["groups"]:
            if g["kind"] != "asm":
                continue
            if (g.get("klass") == "virus" and
                    not cfg.get("virus", {}).get("include_assembly_summary", True)):
                continue
            if (g.get("klass") == "virus" and
                    not cfg.get("virus", {}).get("include_assembly_summary", True)):
                continue
            group = g["group"]
            url = "{}/genomes/{}/{}/assembly_summary.txt".format(FTP, src, group)
            dest = os.path.join(paths.meta, "assembly_summary.{}.{}.txt".format(src, group))
            if (os.path.exists(dest) and _summary_sane(dest)
                    and (time.time() - os.path.getmtime(dest)) < max_age):
                log("  cached: {}".format(os.path.basename(dest)))
                continue
            if os.path.exists(dest) and not _summary_sane(dest):
                log("  !! invalid cached summary, removing: {}".format(dest))
                os.remove(dest)
            log("  fetch {} ...".format(url))
            if not fetch_text(url, dest):
                log("  !! 下载失败: {}".format(url))
            elif not _summary_sane(dest):
                log("  !! summary 无有效数据行（可能是错误页/残缺下载）: {}".format(url))
                os.remove(dest)
    virus_cfg = cfg.get("virus", {})
    if virus_cfg.get("enabled") and virus_cfg.get("include_genome_report_segments"):
        url = virus_cfg["genome_report_url"]
        dest = os.path.join(paths.reports_meta, "viruses.txt")
        fresh = (os.path.exists(dest) and os.path.getsize(dest) > 100
                 and (time.time() - os.path.getmtime(dest)) < max_age)
        if not fresh:
            log("  fetch virus Genome Report ...")
            if not fetch_text(url, dest):
                log("  !! 病毒 Genome Report 下载失败: {}".format(url))
    # JGI 使用用户从 Data Portal/API 导出的本地 TSV/CSV；认证下载结构明确后再接 API。
    if cfg.get("external_sources", {}).get("jgi", {}).get("enabled"):
        stage_ingest_jgi_metadata(cfg, paths)
    # organelle metadata 与 assembly metadata 在同一阶段完成，避免第二次单独启动。
    if cfg.get("organelle", {}).get("fetch_with_metadata", True):
        stage_fetch_organelle(cfg, paths)
    log("fetch-metadata done")


JGI_ALIASES = {
    "native_id": ["taxon_oid", "img_taxon_oid", "jgi_id", "project_id", "id"],
    "assembly_accession": ["assembly_accession", "ncbi_assembly_accession", "genbank_accession", "accession"],
    "taxid": ["taxid", "ncbi_taxid", "taxonomy_id"],
    "species_taxid": ["species_taxid", "ncbi_species_taxid"],
    "organism_name": ["organism_name", "organism", "name", "taxon_name"],
    "group_name": ["group_name", "group", "division"],
    "domain": ["domain", "kingdom", "superkingdom"],
    "genome_type": ["genome_type", "dataset_type", "type"],
    "is_mag": ["is_mag", "mag", "metagenome_assembled", "metagenome_assembled_genome"],
    "genome_size": ["genome_size", "genome_length", "size_bp"],
    "gc_percent": ["gc_percent", "gc", "gc_content"],
    "assembly_level": ["assembly_level", "status", "completion"],
    "bioproject": ["bioproject", "bioproject_accession"],
    "biosample": ["biosample", "biosample_accession"],
    "download_url": ["download_url", "file_url", "fasta_url", "url"],
    "filename": ["filename", "file_name", "fasta_filename"],
    "md5": ["md5", "checksum_md5", "file_md5"],
    "file_size": ["file_size", "size_bytes"],
    "record_url": ["record_url", "portal_url", "dataset_url"],
}


def _read_delimited(path, delimiter="auto"):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace", newline="") as fh:
        sample = fh.read(65536)
        fh.seek(0)
        if delimiter == "auto":
            try:
                delim = csv.Sniffer().sniff(sample, delimiters="\t,;").delimiter
            except csv.Error:
                delim = "\t"
        else:
            delim = "\t" if delimiter in ("tab", "\\t") else delimiter
        yield from csv.DictReader(fh, delimiter=delim)


def _jgi_pick(row, key, column_map):
    lower = {str(k).strip().lower(): v for k, v in row.items()}
    explicit = column_map.get(key)
    names = ([explicit] if explicit else []) + JGI_ALIASES[key]
    for name in names:
        if name and str(name).lower() in lower:
            value = lower[str(name).lower()]
            if value not in (None, "", "na", "NA"):
                return str(value).strip()
    return ""


def _jgi_group_and_class(group, domain, genome_type):
    text = " ".join((group, domain, genome_type)).lower()
    if "mitochond" in text:
        return "mitochondrion", "mitochondria"
    if "plastid" in text or "chloroplast" in text:
        return "plastid", "plastid"
    if "bacteria" in text:
        return "bacteria", "bacteria"
    if "archaea" in text:
        return "archaea", "archaea"
    if "fung" in text:
        return "fungi", "nuclear_euk"
    if "plant" in text or "viridiplant" in text:
        return "plant", "nuclear_euk"
    if "euk" in text:
        return "jgi_eukaryota", "nuclear_euk"
    return "", ""


def _truthy(value):
    return str(value or "").strip().lower() in ("1", "true", "yes", "y", "mag")


def _mag_evidence(m):
    explicit = m.get("is_mag", "")
    if _truthy(explicit):
        return "explicit_is_mag"
    fields = [m.get("excluded_from_refseq", ""), m.get("mag_evidence", ""),
              m.get("genome_type", ""), m.get("assembly_type", ""),
              m.get("definition", "")]
    text = " ".join(str(x or "") for x in fields).lower()
    terms = (
        "derived from metagenome", "metagenome-derived", "metagenome derived",
        "metagenome-assembled", "metagenome assembled genome",
    )
    for term in terms:
        if term in text:
            return term
    # 单独的 MAG 标签，避免把普通单词片段误判。
    if re.search(r"(^|[^a-z])mag([^a-z]|$)", text):
        return "MAG"
    return ""


def stage_ingest_jgi_metadata(cfg, paths):
    log("== ingest-jgi-metadata ==")
    jc = cfg.get("external_sources", {}).get("jgi", {})

    # GOLD 全量 XLSX 是关系型导出，不能用单表 aliases 直接解释。
    workbooks = [os.path.abspath(os.path.expanduser(x)) for x in jc.get("workbooks", [])]
    if workbooks:
        missing = [x for x in workbooks if not os.path.exists(x)]
        if missing:
            log("  !! JGI GOLD workbook 不存在: {}".format(" | ".join(missing)))
            return
        output = os.path.join(paths.index, "jgi_entities.tsv")
        newest_input = max(os.path.getmtime(x) for x in workbooks)
        cache_ok = (os.path.exists(output) and os.path.getsize(output) > 0
                    and os.path.getmtime(output) >= newest_input
                    and not jc.get("force_reimport", False))
        if cache_ok:
            log("  cached GOLD import: {}".format(output))
            return
        from jgi_import import import_gold
        stats = import_gold(workbooks, paths.index, master_columns=MASTER_COLUMNS)
        log("  GOLD imported: accepted={} JGI-only={} MAG_candidates={}".format(
            stats.get("accepted_for_merge", 0), stats.get("accepted_jgi_only", 0),
            stats.get("mag_rows", 0)))
        return

    # 向后兼容旧的扁平 TSV/CSV JGI metadata。
    path = os.path.expanduser(jc.get("metadata_path", ""))
    if not path or not os.path.exists(path):
        log("  !! JGI enabled 但 metadata_path 不存在: {}".format(path or "<empty>"))
        return
    column_map = jc.get("column_map", {})
    rows, rejected = [], []
    for rn, raw in enumerate(_read_delimited(path, jc.get("delimiter", "auto")), 2):
        native = _jgi_pick(raw, "native_id", column_map)
        acc = _jgi_pick(raw, "assembly_accession", column_map).upper()
        if acc and not re.match(r"^GC[AF]_\d+\.\d+$", acc):
            acc = ""
        if not native and not acc:
            rejected.append({"source_row": rn, "reason": "MISSING_NATIVE_AND_ASSEMBLY_ID"})
            continue
        group, klass = _jgi_group_and_class(
            _jgi_pick(raw, "group_name", column_map),
            _jgi_pick(raw, "domain", column_map),
            _jgi_pick(raw, "genome_type", column_map))
        if not group:
            rejected.append({"source_row": rn, "reason": "UNMAPPED_DOMAIN_OR_GROUP"})
            continue
        eid = acc or "JGI:" + native
        download_url = _jgi_pick(raw, "download_url", column_map)
        filename = _jgi_pick(raw, "filename", column_map)
        if not filename and download_url:
            filename = os.path.basename(urlparse(download_url).path)
        genome_type = _jgi_pick(raw, "genome_type", column_map)
        explicit_mag = _jgi_pick(raw, "is_mag", column_map)
        jgi_row = {
            "entity_id": eid, "entity_type": "assembly", "klass": klass,
            "group_name": group, "source": "jgi", "source_native_id": native,
            "dataset_role": "train_candidate", "evaluation_collection": "",
            "metadata_origin": os.path.abspath(path),
            "source_record_url": _jgi_pick(raw, "record_url", column_map),
            "is_mag": int(_truthy(explicit_mag)),
            "mag_evidence": genome_type,
            "assembly_accession": acc,
            "taxid": _jgi_pick(raw, "taxid", column_map),
            "species_taxid": _jgi_pick(raw, "species_taxid", column_map),
            "organism_name": _jgi_pick(raw, "organism_name", column_map),
            "bioproject": _jgi_pick(raw, "bioproject", column_map),
            "biosample": _jgi_pick(raw, "biosample", column_map),
            "assembly_level": _jgi_pick(raw, "assembly_level", column_map),
            "genome_rep": "Full", "version_status": "latest",
            "genome_size": _jgi_pick(raw, "genome_size", column_map),
            "gc_percent": _jgi_pick(raw, "gc_percent", column_map),
            "download_url": download_url,
            "expected_md5": _jgi_pick(raw, "md5", column_map).lower(),
            "source_filename": filename,
            "source_file_size": _jgi_pick(raw, "file_size", column_map),
        }
        evidence = _mag_evidence(jgi_row)
        jgi_row["is_mag"] = int(bool(evidence))
        jgi_row["mag_evidence"] = evidence
        rows.append(jgi_row)
    write_tsv(os.path.join(paths.index, "jgi_entities.tsv"), rows, MASTER_COLUMNS)
    write_tsv(os.path.join(paths.index, "jgi_ingest_rejected.tsv"), rejected,
              ["source_row", "reason"])
    stats = [{
        "metadata_path": os.path.abspath(path), "normalized_records": len(rows),
        "rejected_records": len(rejected),
        "with_ncbi_assembly_accession": sum(bool(r.get("assembly_accession")) for r in rows),
        "source_native_only": sum(not bool(r.get("assembly_accession")) for r in rows),
        "with_download_url": sum(bool(r.get("download_url")) for r in rows),
    }]
    write_tsv(os.path.join(paths.index, "jgi_ingest_stats.tsv"), stats,
              ["metadata_path", "normalized_records", "rejected_records",
               "with_ncbi_assembly_accession", "source_native_only", "with_download_url"])
    log("  JGI normalized={} rejected={} download_urls={}".format(
        len(rows), len(rejected), stats[0]["with_download_url"]))


def _iter_summary(path, group, klass, source):
    """按表头字段名解析 assembly_summary（不用固定列号）。"""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        header = None
        for line in fh:
            if line.startswith("##"):
                continue
            if line.startswith("#"):
                # 最后一行以 # 开头的就是真正表头
                header = line.lstrip("#").rstrip("\n").split("\t")
                header = [h.strip() for h in header]
                continue
            if header is None:
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header):
                parts += [""] * (len(header) - len(parts))
            row = dict(zip(header, parts))
            yield _summary_to_master(row, group, klass, source)


def _g(row, *names):
    for n in names:
        if n in row and row[n] not in ("", "na"):
            return row[n]
    return ""


def _summary_to_master(row, group, klass, source):
    acc = _g(row, "assembly_accession")
    excluded = _g(row, "excluded_from_refseq")
    mag_evidence = "derived from metagenome" \
        if "derived from metagenome" in excluded.lower() else ""
    return {
        "entity_id": acc,
        "entity_type": "assembly",
        "klass": klass,
        "group_name": group,
        "source": source,
        "dataset_role": "test_only" if klass == "virus" else "train_candidate",
        "evaluation_collection": "virus_assembly" if klass == "virus" else "",
        "is_mag": int(bool(mag_evidence)),
        "mag_evidence": mag_evidence,
        "assembly_accession": acc,
        "paired_accession": _g(row, "gbrs_paired_asm"),
        "paired_comparison": _g(row, "paired_asm_comp"),
        "taxid": _g(row, "taxid"),
        "species_taxid": _g(row, "species_taxid"),
        "organism_name": _g(row, "organism_name"),
        "infraspecific_name": _g(row, "infraspecific_name"),
        "bioproject": _g(row, "bioproject"),
        "biosample": _g(row, "biosample"),
        "wgs_master": _g(row, "wgs_master"),
        "refseq_category": _g(row, "refseq_category"),
        "version_status": _g(row, "version_status"),
        "assembly_level": _g(row, "assembly_level"),
        "genome_rep": _g(row, "genome_rep"),
        "assembly_type": _g(row, "assembly_type"),
        "seq_rel_date": _g(row, "seq_rel_date"),
        "asm_name": _g(row, "asm_name"),
        "ftp_path": _g(row, "ftp_path"),
        "genome_size": _g(row, "genome_size"),
        "genome_size_ungapped": _g(row, "genome_size_ungapped"),
        "gc_percent": _g(row, "gc_percent"),
        "scaffold_count": _g(row, "scaffold_count"),
        "contig_count": _g(row, "contig_count"),
        "scaffold_n50": _g(row, "scaffold_n50", "scaffold N50"),
        "contig_n50": _g(row, "contig_n50", "contig N50"),
        "sequencing_technology": _g(row, "sequencing_technology", "sequencing_tech"),
        "relation_to_type_material": _g(row, "relation_to_type_material"),
        "excluded_from_refseq": excluded,
        "sequence_accession": "",
        "organelle_type": "",
        "sequence_length": "",
        "refseq_source_file": "",
        "definition": "",
    }


_VIRUS_ACC_RE = re.compile(r"\b(?:[A-Z]{1,4}_)?[A-Z]{0,2}\d{5,}(?:\.\d+)?\b")


def _iter_virus_report(path, min_len=0):
    """将 Genome Reports/viruses.txt 展开为一条 segment accession 一个实体。"""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8", errors="replace", newline="") as fh:
        first = ""
        for line in fh:
            candidate = line.lstrip("#").rstrip("\r\n")
            if "\t" in candidate and "TaxID" in candidate and (
                    "Segmemts" in candidate or "Segments" in candidate):
                first = candidate
                break
        if not first:
            return
        reader = csv.DictReader(fh, fieldnames=first.split("\t"), delimiter="\t")
        seen = set()
        for line_no, row in enumerate(reader, 2):
            lower = {str(k).strip().lower().replace(" ", "_"): (v or "").strip()
                     for k, v in row.items()}
            segments = (lower.get("segmemts") or lower.get("segments") or
                        lower.get("replicons") or "")
            if not segments or segments == "-":
                continue
            taxid = lower.get("taxid", "")
            organism = lower.get("organism/name", "") or lower.get("organism_name", "")
            size_kb = _to_float(lower.get("size_(kb)") or lower.get("size_kb"))
            if min_len and size_kb and size_kb * 1000 < min_len:
                continue
            # 一行代表一个病毒基因组；多节段 accession 必须作为同一选择单元。
            genome_key = "\t".join((taxid, organism, segments))
            virus_genome_id = "VGENOME:" + hashlib.sha1(
                genome_key.encode("utf-8", errors="replace")).hexdigest()[:20]
            for token in re.split(r"\s*;\s*", segments):
                accs = _VIRUS_ACC_RE.findall(token)
                if not accs:
                    continue
                acc = next((a for a in accs if "_" in a), accs[0])
                if acc in seen:
                    continue
                seen.add(acc)
                seg_name = token.split(":", 1)[0].strip() if ":" in token else ""
                yield {
                    "entity_id": "VIRUSSEQ:" + acc,
                    "entity_type": "virus_sequence", "klass": "virus",
                    "group_name": "viral", "source": "ncbi_genome_reports",
                    "dataset_role": "test_only", "evaluation_collection": "virus_segment",
                    "source_native_id": acc, "metadata_origin": os.path.basename(path),
                    "taxid": taxid, "species_taxid": taxid, "organism_name": organism,
                    "bioproject": lower.get("bioproject_accession", ""),
                    "biosample": lower.get("biosample_accession", ""),
                    "download_url": ("https://www.ncbi.nlm.nih.gov/sviewer/viewer.fcgi"
                                     "?id={}&db=nuccore&report=fasta&retmode=text").format(acc),
                    "source_filename": acc + ".fna", "sequence_accession": acc,
                    # Size(Kb) 是整套病毒基因组长度，不能伪装成单 segment 长度。
                    "sequence_length": "",
                    "virus_group": lower.get("group", ""),
                    "virus_subgroup": lower.get("subgroup", ""),
                    "segment_name": seg_name, "host": lower.get("host", ""),
                    "virus_genome_id": virus_genome_id,
                    "virus_reported_genome_length": (
                        str(int(size_kb * 1000)) if size_kb else ""),
                    "is_refseq_viral": int(acc.startswith(("NC_", "NG_"))),
                    "definition": organism,
                    "source_record_url": "{}#L{}".format(path, line_no),
                }


def _virus_genome_type(value):
    """Normalize NCBI Genome Reports virus group labels."""
    text = re.sub(r"[^a-z0-9]+", "", str(value or "").lower())
    if "dsdna" in text:
        return "dsDNA"
    if "ssdna" in text:
        return "ssDNA"
    if "dsrna" in text:
        return "dsRNA"
    if "ssrna" in text:
        return "ssRNA"
    if "retro" in text:
        return "retro"
    return "unknown"


def _virus_metadata_qc(row, virus_cfg):
    """Cheap pre-download QC. Sequence-level QC is repeated after download."""
    flags = []
    acc = row.get("sequence_accession", "")
    if not acc or not re.match(r"^[A-Z][A-Z0-9_]*\d+(?:\.\d+)?$", acc):
        flags.append("INVALID_ACCESSION")
    if virus_cfg.get("require_taxid", True) and not row.get("taxid"):
        flags.append("MISSING_TAXID")
    if not row.get("download_url"):
        flags.append("MISSING_DOWNLOAD_URL")
    allowed = set(virus_cfg.get("allowed_genome_types", []))
    genome_type = _virus_genome_type(row.get("virus_group"))
    if allowed and genome_type not in allowed:
        flags.append("GENOME_TYPE_{}".format(genome_type))
    reported_len = _to_float(row.get("virus_reported_genome_length"))
    min_len = _to_float(virus_cfg.get("min_len", 0))
    if min_len and reported_len and reported_len < min_len:
        flags.append("REPORTED_GENOME_TOO_SHORT")
    return (not flags), flags, genome_type


# --------------------------------------------------------------------------- #
# 2. build-index : 解���所有 summary + organelle → canonical 总表
# --------------------------------------------------------------------------- #

def stage_build_index(cfg, paths):
    log("== build-index ==")
    sources = _asm_sources(cfg)
    fil = cfg["filters"]
    rows = []
    seen = set()
    seen_accession_bases = set()
    exact_dupes = 0
    audit_rows = []
    mag_excluded_rows = []
    for src in sources:
        for g in cfg["groups"]:
            if g["kind"] != "asm":
                continue
            group, klass = g["group"], g["klass"]
            path = os.path.join(paths.meta, "assembly_summary.{}.{}.txt".format(src, group))
            n_in = n_keep = n_dup = n_hard = n_mag = n_mag_filtered = 0
            species = set()
            for m in _iter_summary(path, group, klass, src):
                n_in += 1
                evidence = _mag_evidence(m)
                if evidence:
                    n_mag += 1
                    policy = fil.get("mag_policy", "exclude" if fil.get("exclude_mag") else "test_only")
                    if policy == "test_only":
                        m["dataset_role"] = "test_only"
                        m["evaluation_collection"] = "mag_{}".format(klass)
                    if policy == "exclude":
                        n_mag_filtered += 1
                        mag_excluded_rows.append({
                            "entity_id": m.get("entity_id", ""), "source": src,
                            "group_name": group, "assembly_accession": m.get("assembly_accession", ""),
                            "source_native_id": m.get("source_native_id", ""),
                            "organism_name": m.get("organism_name", ""), "mag_evidence": evidence,
                        })
                if m.get("species_taxid"):
                    species.add(m["species_taxid"])
                if not _passes_hard_filter(m, fil):
                    n_hard += 1
                    continue
                # 完整 accession（含版本号）精确去重。
                if m["entity_id"] in seen:
                    n_dup += 1
                    exact_dupes += 1
                    continue
                seen.add(m["entity_id"])
                acc = m.get("assembly_accession", "")
                if acc:
                    seen_accession_bases.add(acc.rsplit(".", 1)[0])
                rows.append(m)
                n_keep += 1
            log("  {:<8} {:<22} in={:<8} kept={:<8} exact_dup={}".format(
                src, group, n_in, n_keep, n_dup))
            audit_rows.append({
                "source": src, "group_name": group,
                "summary_file": os.path.basename(path),
                "file_exists": int(os.path.exists(path)),
                "file_size": os.path.getsize(path) if os.path.exists(path) else 0,
                "parsed_records": n_in, "hard_filtered": n_hard,
                "exact_accession_duplicates": n_dup, "pre_pair_kept": n_keep,
                "derived_from_metagenome": n_mag,
                "mag_excluded": n_mag_filtered,
                "non_metagenome": n_in - n_mag,
                "distinct_species_taxids": len(species),
            })
    if exact_dupes:
        log("  exact accession 去重: {} 条".format(exact_dupes))

    write_tsv(os.path.join(paths.index, "metadata_inventory_stats.tsv"),
              audit_rows,
              ["source", "group_name", "summary_file", "file_exists", "file_size",
               "parsed_records", "hard_filtered", "exact_accession_duplicates",
               "pre_pair_kept", "derived_from_metagenome", "non_metagenome",
               "mag_excluded", "distinct_species_taxids"])

    # 合并 JGI 标准化 metadata。优先使用 NCBI assembly accession 去重；
    # JGI-only 记录使用 JGI:<source_native_id> 作为稳定主键。
    jgi_path = os.path.join(paths.index, "jgi_entities.tsv")
    jgi_in = jgi_added = jgi_dup = jgi_mag_filtered = 0
    if os.path.exists(jgi_path):
        for m in read_tsv(jgi_path):
            jgi_in += 1
            # jgi_import.py 的既有输出可能早于 dataset_role 字段；Model A isolate 默认可训练。
            if not m.get("dataset_role"):
                m["dataset_role"] = "train_candidate"
            evidence = _mag_evidence(m)
            policy = fil.get("mag_policy", "exclude" if fil.get("exclude_mag") else "test_only")
            if evidence and policy == "test_only":
                m["dataset_role"] = "test_only"
                m["evaluation_collection"] = "mag_{}".format(m.get("klass") or "unknown")
            if evidence and policy == "exclude":
                jgi_mag_filtered += 1
                mag_excluded_rows.append({
                    "entity_id": m.get("entity_id", ""), "source": "jgi",
                    "group_name": m.get("group_name", ""),
                    "assembly_accession": m.get("assembly_accession", ""),
                    "source_native_id": m.get("source_native_id", ""),
                    "organism_name": m.get("organism_name", ""), "mag_evidence": evidence,
                })
                continue
            key = m.get("assembly_accession") or m.get("entity_id")
            acc = m.get("assembly_accession", "")
            base = acc.rsplit(".", 1)[0] if acc else ""
            # exact accession、同一 accession base 的其他版本、以及 paired accession
            # 已由 NCBI inventory 覆盖时，都不重复加入 GOLD。
            if (not key or key in seen
                    or (base and base in seen_accession_bases)
                    or (m.get("paired_accession") and m.get("paired_accession") in seen)):
                jgi_dup += 1
                continue
            seen.add(key)
            if base:
                seen_accession_bases.add(base)
            rows.append(m)
            jgi_added += 1
        log("  JGI merged: input={} added={} duplicate={} MAG_excluded={}".format(
            jgi_in, jgi_added, jgi_dup, jgi_mag_filtered))

    # 注意：jgi_mag_candidates.tsv 仍不自动合并。GOLD 缺 completeness/contamination，
    # 只有 jgi_entities.tsv（Model A isolate）进入此处。
    virus_report = os.path.join(paths.reports_meta, "viruses.txt")
    virus_added = virus_dup = 0
    if (cfg.get("virus", {}).get("enabled") and
            cfg.get("virus", {}).get("include_genome_report_segments")):
        for m in _iter_virus_report(virus_report, cfg["virus"].get("min_len", 0)):
            if m["entity_id"] in seen:
                virus_dup += 1
                continue
            seen.add(m["entity_id"])
            rows.append(m)
            virus_added += 1
        log("  virus report merged: added={} duplicate={}".format(virus_added, virus_dup))
    write_tsv(os.path.join(paths.index, "jgi_merge_stats.tsv"), [{
        "input_records": jgi_in, "added_to_canonical": jgi_added,
        "duplicate_ncbi_or_jgi": jgi_dup, "mag_excluded": jgi_mag_filtered,
    }], ["input_records", "added_to_canonical", "duplicate_ncbi_or_jgi", "mag_excluded"])

    write_tsv(os.path.join(paths.index, "mag_excluded.tsv"), mag_excluded_rows,
              ["entity_id", "source", "group_name", "assembly_accession",
               "source_native_id", "organism_name", "mag_evidence"])

    # 合并 organelle 总表（如已运行 fetch-organelle）
    org_master = os.path.join(paths.master, "organelle_entities.tsv")
    if os.path.exists(org_master):
        org = read_tsv(org_master)
        rows.extend(org)
        log("  organelle merged: {} rows".format(len(org)))

    rows = _canonicalize_gca_gcf(rows)
    write_tsv(paths.master_tsv, rows, MASTER_COLUMNS)
    log("build-index done: {} entities".format(len(rows)))


def _passes_hard_filter(m, fil):
    if m["entity_type"] != "assembly":
        return True
    policy = fil.get("mag_policy", "exclude" if fil.get("exclude_mag") else "test_only")
    if policy == "exclude" and _mag_evidence(m):
        return False
    if m.get("source") in ("jgi", "jgi_gold"):
        # JGI metadata 可以先进入 inventory；没有下载 URL 的记录只会在 plan-downloads 跳过。
        return bool(m.get("source_native_id") or m.get("assembly_accession"))
    if not m["assembly_accession"] or not re.match(r"^GC[AF]_\d+\.\d+$", m["assembly_accession"]):
        return False
    if not m["ftp_path"] or m["ftp_path"] == "na":
        return False
    if fil["require_latest"] and m["version_status"] and m["version_status"] != "latest":
        return False
    if fil["require_full"] and m["genome_rep"] and m["genome_rep"].lower() != "full":
        return False
    d = m["seq_rel_date"]
    if fil["fetch_before"] and d and d >= fil["fetch_before"]:
        return False
    if fil["fetch_after"] and d and d < fil["fetch_after"]:
        return False
    return True


def _canonicalize_gca_gcf(rows):
    """identical 的 GCA/GCF 只保留一份物理序列（默认优先 GCF）。"""
    by_acc = {r["assembly_accession"]: r for r in rows if r.get("assembly_accession")}
    drop = set()
    for acc, r in by_acc.items():
        if r.get("paired_comparison") != "identical":
            continue
        paired = r.get("paired_accession", "")
        if paired and paired in by_acc and acc not in drop and paired not in drop:
            # 保留 GCF，丢弃 GCA
            loser = acc if acc.startswith("GCA_") else paired
            drop.add(loser)
    kept = [r for r in rows if r.get("assembly_accession", "__x__") not in drop
            or r["entity_type"] != "assembly"]
    if drop:
        log("  GCA/GCF identical 去重: 丢弃 {} 份".format(len(drop)))
    return kept


# --------------------------------------------------------------------------- #
# 3. apply-qc : 关联 ANI/size/type 等报告，生成 qc_status + score + split_group
# --------------------------------------------------------------------------- #

def _report(paths, name):
    return os.path.join(load_reports_dir(paths), name)


_reports_dir_cache = {}


def load_reports_dir(paths):
    return _reports_dir_cache.get("dir", "")


def _read_report_headered(path):
    """NCBI 报告：跳过 ## 注释，以最后一个 # 开头行为表头，逐行 yield dict。"""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8", errors="replace") as fh:
        header = None
        for line in fh:
            if line.startswith("#"):
                header = line.lstrip("#").rstrip("\n").split("\t")
                header = [h.strip() for h in header]
                continue
            if header is None:
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header):
                parts += [""] * (len(header) - len(parts))
            yield dict(zip(header, parts))



def _norm_feature_key(value):
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _flatten_feature_json(obj, prefix="", out=None):
    out = {} if out is None else out
    if isinstance(obj, dict):
        for key, value in obj.items():
            name = prefix + "." + str(key) if prefix else str(key)
            _flatten_feature_json(value, name, out)
    elif isinstance(obj, list):
        out[prefix] = ";".join(str(x) for x in obj if x not in (None, ""))
    else:
        out[prefix] = obj
    return out


def _feature_pick(row, aliases):
    normalized = {_norm_feature_key(k): v for k, v in row.items()}
    for alias in aliases:
        value = normalized.get(_norm_feature_key(alias))
        if value not in (None, "", "na", "NA"):
            return str(value).strip()
    return ""


def _load_assembly_features(path):
    """Load optional NCBI Datasets JSONL or TSV/CSV keyed by assembly accession."""
    if not path or not os.path.exists(path):
        return {}
    records = []
    if path.lower().endswith((".jsonl", ".ndjson")):
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(_flatten_feature_json(json.loads(line)))
    else:
        with open(path, encoding="utf-8", errors="replace", newline="") as fh:
            sample = fh.read(65536); fh.seek(0)
            try:
                delim = csv.Sniffer().sniff(sample, delimiters="\t,;").delimiter
            except csv.Error:
                delim = "\t"
            records = list(csv.DictReader(fh, delimiter=delim))
    out = {}
    for row in records:
        acc = _feature_pick(row, ["assembly_accession", "accession", "assembly accession",
                                  "current_accession", "assemblyInfo.accession"])
        if not re.match(r"^GC[AF]_\d+\.\d+$", acc or ""):
            continue
        out[acc] = {
            "contig_n50": _feature_pick(row, ["contig_n50", "contig N50",
                "assemblyStats.contigN50", "assmstats-contig-n50"]),
            "scaffold_n50": _feature_pick(row, ["scaffold_n50", "scaffold N50",
                "assemblyStats.scaffoldN50", "assmstats-scaffold-n50"]),
            "sequencing_technology": _feature_pick(row, ["sequencing_technology",
                "sequencing technology", "sequencing_tech", "assemblyInfo.sequencingTech",
                "assminfo-sequencing-tech"]),
        }
    return out


def _is_third_generation(technology):
    text = str(technology or "").lower()
    terms = ("pacbio", "pac bio", "smrt", "hifi", "sequel", "revio",
             "nanopore", "oxford nanopore", "promethion", "minion", "gridion")
    return any(term in text for term in terms)


def _non_isolate_evidence(row):
    text = " ".join(str(row.get(k, "") or "") for k in
        ("excluded_from_refseq", "assembly_type", "organism_name",
         "infraspecific_name", "mag_evidence", "definition")).lower()
    for term in ("derived from metagenome", "metagenome-assembled", "metagenome assembled",
                 "metagenomic", "environmental sample", "single-cell", "single cell genome"):
        if term in text:
            return term
    return ""

def stage_apply_qc(cfg, paths):
    log("== apply-qc ==")
    _reports_dir_cache["dir"] = cfg["reports_dir"]
    rd = cfg["reports_dir"]
    rows = read_tsv(paths.master_tsv)
    if not rows:
        log("!! 总表为空，先运行 build-index")
        return

    feature_path = os.path.expanduser(cfg.get("select", {}).get("feature_metadata_path", ""))
    assembly_features = _load_assembly_features(feature_path)
    if feature_path:
        log("  assembly features: {} records from {}".format(len(assembly_features), feature_path))
        if not assembly_features:
            log("  !! N50/technology feature file missing or empty; ranking will mark missing values")

    # ---- 加载报告 ----
    ani_type_bad = set()
    ani_syn = set()
    contamination = {}
    ani_group = {}          # species_taxid -> group id (indistinguishable)
    size_range = {}         # species_taxid -> (min,max)
    type_strain = set()     # assembly acc
    reference = set()       # assembly acc

    def rp(name):
        return os.path.join(rd, name)

    for r in _read_report_headered(rp("prokaryote_ANI_type_not_matching.txt")):
        acc = _g(r, "assembly", "assembly_accession", "Assembly", "#assembly")
        if acc:
            ani_type_bad.add(acc)
    for r in _read_report_headered(rp("prokaryote_ANI_suspect_heterotypic_synonyms.txt")):
        acc = _g(r, "assembly", "assembly_accession", "Assembly")
        if acc:
            ani_syn.add(acc)
    for r in _read_report_headered(rp("prokaryote_ANI_contamination.txt")):
        acc = _g(r, "assembly", "assembly_accession", "Assembly")
        frac = _g(r, "contamination", "contam_fraction", "fraction")
        if acc:
            try:
                contamination[acc] = float(frac)
            except ValueError:
                contamination[acc] = 0.0
    gid = 0
    for r in _read_report_headered(rp("prokaryote_ANI_indistinguishable_groups.txt")):
        gid += 1
        sp = _g(r, "species_taxid", "taxid", "species")
        if sp:
            ani_group[sp] = "ANIG_{}".format(r.get("group", gid))
    for r in _read_report_headered(rp("species_genome_size.txt")):
        sp = _g(r, "species_taxid", "taxid", "#species_taxid")
        lo = _g(r, "min_ungapped_length", "minimum_ungapped_length", "min")
        hi = _g(r, "max_ungapped_length", "maximum_ungapped_length", "max")
        if sp:
            try:
                size_range[sp] = (float(lo), float(hi))
            except ValueError:
                pass
    for r in _read_report_headered(rp("prokaryote_type_strain_report.txt")):
        acc = _g(r, "assembly", "assembly_accession", "Assembly")
        if acc:
            type_strain.add(acc)
    for r in _read_report_headered(rp("reference_assemblies_log.txt")):
        acc = _g(r, "assembly", "assembly_accession", "Assembly")
        if acc:
            reference.add(acc)

    log("  reports: type_bad={} syn={} contam={} anigroup={} size={} type_strain={} ref={}".format(
        len(ani_type_bad), len(ani_syn), len(contamination), len(ani_group),
        len(size_range), len(type_strain), len(reference)))

    qc_rows = []
    counts = {"accepted": 0, "quarantine": 0, "rejected": 0}
    for m in rows:
        acc = m.get("assembly_accession", "")
        sp = m.get("species_taxid", "")
        feat = assembly_features.get(acc, {})
        for key in ("contig_n50", "scaffold_n50", "sequencing_technology"):
            if feat.get(key) and not m.get(key):
                m[key] = feat[key]
        contig_n50 = _to_float(m.get("contig_n50"))
        scaffold_n50 = _to_float(m.get("scaffold_n50"))
        if contig_n50:
            m["n50_value"], m["n50_metric"] = str(int(contig_n50)), "contig_n50"
        elif scaffold_n50:
            m["n50_value"], m["n50_metric"] = str(int(scaffold_n50)), "scaffold_n50"
        else:
            m["n50_value"], m["n50_metric"] = "", "missing"
        m["is_third_generation"] = int(_is_third_generation(m.get("sequencing_technology")))
        flags = []
        score = 0.0
        status = "accepted"

        if m["entity_type"] == "organelle":
            # organelle 已在 fetch-organelle 阶段做过基本清洗
            score = 5.0
            qc_rows.append({
                "entity_id": m["entity_id"], "qc_status": "accepted",
                "score": "{:.2f}".format(score), "flags": "",
                "split_group_id": sp or m["entity_id"],
            })
            counts["accepted"] += 1
            continue

        if m["entity_type"] == "virus_sequence":
            passed, virus_flags, genome_type = _virus_metadata_qc(
                m, cfg.get("virus", {}))
            m["assembly_type"] = genome_type
            # RefSeq curated accession、明确 host 和明确 genome type 用于代表选择排序。
            score = 5.0
            score += 4.0 if str(m.get("is_refseq_viral", "0")) in ("1", "true", "True") else 0.0
            score += 1.0 if m.get("host") else 0.0
            score += 1.0 if genome_type != "unknown" else 0.0
            status = "accepted" if passed else "rejected"
            qc_rows.append({
                "entity_id": m["entity_id"], "qc_status": status,
                "score": "{:.2f}".format(score),
                "flags": ",".join(virus_flags),
                "split_group_id": m.get("virus_genome_id") or sp or m["entity_id"],
            })
            counts[status] += 1
            continue

        # ---- quarantine 级 ----
        if acc in ani_type_bad:
            flags.append("TYPE_NOT_MATCHING"); status = "quarantine"
        if acc in ani_syn:
            flags.append("SUSPECT_SYNONYM"); status = "quarantine"
        cf_frac = contamination.get(acc)
        if cf_frac is not None and cf_frac >= cfg["qc"]["contamination_frac_quarantine"]:
            flags.append("CONTAMINATION"); status = "quarantine"
        gsize = _to_float(m.get("genome_size_ungapped") or m.get("genome_size"))
        if sp in size_range and gsize:
            lo, hi = size_range[sp]
            if gsize < lo * cfg["qc"]["size_low_ratio"]:
                flags.append("SIZE_BELOW_RANGE"); status = "quarantine"
            elif gsize > hi * cfg["qc"]["size_high_ratio"]:
                flags.append("SIZE_ABOVE_RANGE"); status = "quarantine"

        # ---- 软惩罚 / 正向加分 ----
        n50_value = _to_float(m.get("n50_value"))
        if n50_value:
            score += min(math.log10(n50_value + 1), 8)
        if m.get("source") == "refseq":
            score += 2
        if str(m.get("is_third_generation", "0")) in ("1", "true", "True"):
            score += 2
        level = (m.get("assembly_level") or "").lower()
        if level in ("complete genome", "chromosome"):
            score += 3
        elif level == "scaffold":
            score -= 1
        elif level == "contig":
            score -= 2
        if m.get("refseq_category") in ("reference genome", "representative genome"):
            score += 3
        if acc in reference:
            score += 3
        if acc in type_strain or (m.get("relation_to_type_material") or ""):
            score += 2
        if cf_frac:
            score -= min(cf_frac * 10, 3)
        if m.get("excluded_from_refseq"):
            score -= 1

        # split group：ANI group → species_taxid → entity_id
        sgid = ani_group.get(sp) or ("SP_" + sp if sp else m["entity_id"])

        qc_rows.append({
            "entity_id": m["entity_id"], "qc_status": status,
            "score": "{:.2f}".format(score), "flags": ",".join(flags),
            "split_group_id": sgid,
        })
        counts[status] += 1

    # Persist N50/technology enrichment into canonical inventory.
    write_tsv(paths.master_tsv, rows, MASTER_COLUMNS)
    write_tsv(paths.qc_tsv, qc_rows, QC_COLUMNS)
    # 分状态导出
    idx = {q["entity_id"]: q for q in qc_rows}
    for st in ("accepted", "quarantine", "rejected"):
        sub = [m for m in rows if idx.get(m["entity_id"], {}).get("qc_status") == st]
        write_tsv(os.path.join(paths.qc, st + ".tsv"), sub, MASTER_COLUMNS)
    log("apply-qc done: {}".format(counts))


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


# --------------------------------------------------------------------------- #
# 4. select : 基于 score 排序 + 可选 taxonomy-aware quota
# --------------------------------------------------------------------------- #

def stage_select(cfg, paths):
    log("== select ==")
    rows = read_tsv(paths.master_tsv)
    qc = {q["entity_id"]: q for q in read_tsv(paths.qc_tsv)}
    accepted = [m for m in rows if qc.get(m["entity_id"], {}).get("qc_status") == "accepted"]
    scfg = cfg.get("select", {})
    weights = {
        "log10_n50": 12.0, "refseq": 8.0, "third_generation": 6.0,
        "reference_category": 4.0, "type_material": 2.0,
        "assembly_level": 2.0, "qc_score": 1.0,
    }
    weights.update(scfg.get("weights", {}))

    def species_key(m):
        if m.get("species_taxid"):
            return "SP:" + m["species_taxid"]
        if m.get("taxid"):
            return "TX:" + m["taxid"]
        name = (m.get("organism_name") or "").strip().lower()
        if name:
            return "NAME:" + name
        return "ENTITY:" + m["entity_id"]

    def rank_info(m):
        n50 = _to_float(m.get("n50_value") or m.get("contig_n50") or m.get("scaffold_n50"))
        refseq = int(m.get("source") == "refseq" or m.get("assembly_accession", "").startswith("GCF_"))
        third = int(str(m.get("is_third_generation", "0")) in ("1", "true", "True")
                    or _is_third_generation(m.get("sequencing_technology")))
        refcat = int((m.get("refseq_category") or "") in
                     ("reference genome", "representative genome"))
        typemat = int(bool(m.get("relation_to_type_material") or m.get("type_strain")))
        level_text = (m.get("assembly_level") or "").lower()
        level_rank = {"complete genome": 4, "chromosome": 3, "scaffold": 2, "contig": 1}.get(level_text, 0)
        qscore = _to_float(qc.get(m["entity_id"], {}).get("score"))
        total = (weights["log10_n50"] * (math.log10(n50 + 1) if n50 else 0.0)
                 + weights["refseq"] * refseq
                 + weights["third_generation"] * third
                 + weights["reference_category"] * refcat
                 + weights["type_material"] * typemat
                 + weights["assembly_level"] * level_rank
                 + weights["qc_score"] * qscore)
        components = "n50={};refseq={};third_gen={};refcat={};type={};level={};qc={}".format(
            int(n50), refseq, third, refcat, typemat, level_rank, qscore)
        return total, n50, refseq, third, refcat, typemat, level_rank, qscore, components

    isolate_groups, mag_groups, virus_assembly_groups = {}, {}, {}
    virus_sequence_rows = []
    passthrough_train, passthrough_eval = [], []
    audit = []
    for m in accepted:
        if m.get("entity_type") == "virus_sequence":
            virus_sequence_rows.append(m)
            continue
        if m.get("entity_type") != "assembly":
            (passthrough_eval if m.get("dataset_role") == "test_only" else passthrough_train).append(m)
            continue
        if m.get("klass") == "virus":
            virus_assembly_groups.setdefault(species_key(m), []).append(m)
            continue
        evidence = _mag_evidence(m)
        non_iso = _non_isolate_evidence(m)
        if evidence:
            m["is_mag"] = 1
            m["mag_evidence"] = evidence
            m["dataset_role"] = "test_only"
            m["evaluation_collection"] = m.get("evaluation_collection") or "mag_{}".format(m.get("klass") or "unknown")
            mag_groups.setdefault(species_key(m), []).append(m)
        elif m.get("dataset_role") == "test_only":
            passthrough_eval.append(m)
        elif non_iso:
            info = rank_info(m)
            audit.append({"entity_id": m["entity_id"], "partition": "excluded_non_isolate",
                          "species_key": species_key(m), "rank": "", "selected": 0,
                          "selection_score": "{:.6f}".format(info[0]), "n50_value": int(info[1]),
                          "n50_metric": m.get("n50_metric", ""), "source": m.get("source", ""),
                          "sequencing_technology": m.get("sequencing_technology", ""),
                          "is_third_generation": info[3], "qc_score": info[7],
                          "score_components": info[8], "reason": "NON_ISOLATE_EVIDENCE:" + non_iso})
        else:
            isolate_groups.setdefault(species_key(m), []).append(m)

    selected_isolates, selected_mags = [], []

    def choose(groups, partition, limit):
        chosen = []
        for sp, candidates in groups.items():
            ranked = sorted(candidates,
                key=lambda m: (rank_info(m)[0], rank_info(m)[1], rank_info(m)[2],
                               rank_info(m)[3], m.get("assembly_accession") or m["entity_id"]),
                reverse=True)
            for rank, m in enumerate(ranked, 1):
                info = rank_info(m); keep = rank <= limit
                if keep:
                    chosen.append(m)
                reason = "SELECTED_BEST_PER_SPECIES" if keep else "LOWER_RANK_SAME_SPECIES"
                if not info[1]: reason += ";MISSING_N50"
                if not m.get("sequencing_technology"): reason += ";UNKNOWN_TECH"
                audit.append({"entity_id": m["entity_id"], "partition": partition,
                    "species_key": sp, "rank": rank, "selected": int(keep),
                    "selection_score": "{:.6f}".format(info[0]), "n50_value": int(info[1]),
                    "n50_metric": m.get("n50_metric", ""), "source": m.get("source", ""),
                    "sequencing_technology": m.get("sequencing_technology", ""),
                    "is_third_generation": info[3], "qc_score": info[7],
                    "score_components": info[8], "reason": reason})
        return chosen

    isolate_limit = 1 if scfg.get("one_isolate_per_species", True) else 10**9
    mag_limit = 1 if scfg.get("one_mag_per_species", True) else 10**9
    selected_isolates = choose(isolate_groups, "train_isolate", isolate_limit)
    selected_mags = choose(mag_groups, "mag_evaluation", mag_limit)
    selected_virus_assemblies = choose(
        virus_assembly_groups, "virus_assembly_test",
        1 if scfg.get("one_isolate_per_species", True) else 10**9)

    # Genome Reports 病毒：先以 virus_genome_id 聚合完整 segment 集，再按 species
    # 选择一个代表 genome。max_genomes 限制 genome 数，不截断被选 genome 的 segments。
    vcfg = cfg.get("virus", {})
    genome_rows = {}
    for m in virus_sequence_rows:
        gid = m.get("virus_genome_id") or m["entity_id"]
        genome_rows.setdefault(gid, []).append(m)

    genome_candidates = []
    for gid, segments in genome_rows.items():
        first = segments[0]
        qscore = max(_to_float(qc.get(x["entity_id"], {}).get("score")) for x in segments)
        curated = sum(str(x.get("is_refseq_viral", "0")) in ("1", "true", "True") for x in segments)
        host_known = int(any(x.get("host") for x in segments))
        reported_len = max(_to_float(x.get("virus_reported_genome_length")) for x in segments)
        score = qscore + 4.0 * int(curated > 0) + host_known + min(math.log10(reported_len + 1), 7)
        genome_candidates.append({
            "gid": gid, "segments": segments,
            "species_key": species_key(first), "score": score,
            "curated_segments": curated, "host_known": host_known,
            "reported_length": int(reported_len),
            "genome_type": _virus_genome_type(first.get("virus_group")),
        })

    by_species = {}
    for g in genome_candidates:
        by_species.setdefault(g["species_key"], []).append(g)
    chosen_genomes = []
    virus_audit = []
    one_per_species = vcfg.get("one_genome_per_species", True)
    for sp, candidates in by_species.items():
        ranked = sorted(candidates, key=lambda x: (x["score"], x["curated_segments"],
                        x["reported_length"], x["gid"]), reverse=True)
        for rank, g in enumerate(ranked, 1):
            keep = rank == 1 or not one_per_species
            if keep:
                chosen_genomes.append(g)
            virus_audit.append({
                "virus_genome_id": g["gid"], "species_key": sp, "rank": rank,
                "selected": int(keep), "selection_score": "{:.6f}".format(g["score"]),
                "segment_count": len(g["segments"]),
                "curated_segment_count": g["curated_segments"],
                "host_known": g["host_known"], "genome_type": g["genome_type"],
                "reported_genome_length": g["reported_length"],
                "reason": "SELECTED_BEST_PER_SPECIES" if keep else "LOWER_RANK_SAME_SPECIES",
            })

    max_genomes = int(vcfg.get("max_genomes", 0) or 0)
    chosen_genomes.sort(key=lambda x: (x["score"], x["curated_segments"],
                        x["reported_length"], x["gid"]), reverse=True)
    if max_genomes:
        retained = {g["gid"] for g in chosen_genomes[:max_genomes]}
        chosen_genomes = chosen_genomes[:max_genomes]
        for a in virus_audit:
            if a["selected"] and a["virus_genome_id"] not in retained:
                a["selected"] = 0
                a["reason"] = "GLOBAL_GENOME_CAP"

    selected_virus_sequences = [segment for g in chosen_genomes for segment in g["segments"]]
    write_tsv(os.path.join(paths.master, "selected_virus_test.tsv"),
              selected_virus_assemblies + selected_virus_sequences, MASTER_COLUMNS)
    write_tsv(os.path.join(paths.master, "virus_selection_audit.tsv"), virus_audit,
              ["virus_genome_id", "species_key", "rank", "selected", "selection_score",
               "segment_count", "curated_segment_count", "host_known", "genome_type",
               "reported_genome_length", "reason"])

    selected = (selected_isolates + passthrough_train + selected_mags + passthrough_eval
                + selected_virus_assemblies + selected_virus_sequences)
    caps = scfg.get("per_class_cap", {})
    if any(caps.values()):
        kept, counts = [], {}
        for m in sorted(selected, key=lambda x: rank_info(x)[0] if x.get("entity_type") == "assembly" else 0, reverse=True):
            if m.get("dataset_role") == "test_only":
                kept.append(m); continue
            cap = caps.get(m.get("klass"), 0)
            if cap and counts.get(m.get("klass"), 0) >= cap:
                continue
            kept.append(m); counts[m.get("klass")] = counts.get(m.get("klass"), 0) + 1
        selected = kept
        selected_ids = {m["entity_id"] for m in selected}
        selected_isolates = [m for m in selected_isolates if m["entity_id"] in selected_ids]

    assert all(not _mag_evidence(m) and m.get("dataset_role") != "test_only" for m in selected_isolates)
    assert all(m.get("dataset_role") == "test_only" and _mag_evidence(m) for m in selected_mags)
    assert len({species_key(m) for m in selected_isolates}) == len(selected_isolates)

    write_tsv(os.path.join(paths.master, "selected_train_isolates.tsv"), selected_isolates, MASTER_COLUMNS)
    write_tsv(os.path.join(paths.master, "selected_mag_test.tsv"), selected_mags, MASTER_COLUMNS)
    write_tsv(os.path.join(paths.master, "selected.tsv"), selected, MASTER_COLUMNS)
    audit_cols = ["entity_id", "partition", "species_key", "rank", "selected",
                  "selection_score", "n50_value", "n50_metric", "source",
                  "sequencing_technology", "is_third_generation", "qc_score",
                  "score_components", "reason"]
    write_tsv(os.path.join(paths.master, "selection_audit.tsv"), audit, audit_cols)
    stats = {"accepted_candidates": len(accepted), "selected_train_isolates": len(selected_isolates),
             "selected_mag_test": len(selected_mags), "passthrough_train": len(passthrough_train),
             "passthrough_evaluation": len(passthrough_eval),
             "excluded_non_isolate": sum(r["partition"] == "excluded_non_isolate" for r in audit)}
    with open(os.path.join(paths.master, "selection_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)
    log("select done: {}".format(stats))



# --------------------------------------------------------------------------- #
# 5. plan-downloads : 生成 md5-aware 下载 manifest（按类别拆分）
# --------------------------------------------------------------------------- #

MANIFEST_COLUMNS = [
    "entity_id", "source", "klass", "group_name", "file_type", "url", "backup_url",
    "md5_url", "expected_md5", "local_dir", "filename", "fna_stem", "size_hint",
]


def _filename_stem(filename):
    for suffix in ("_genomic.fna.gz", "_genomic.fna", ".fna.gz", ".fna"):
        if filename.endswith(suffix):
            return filename[:-len(suffix)]
    return filename


def stage_plan_downloads(cfg, paths):
    log("== plan-downloads ==")
    sel_path = os.path.join(paths.master, "selected.tsv")
    rows = read_tsv(sel_path) or read_tsv(paths.master_tsv)
    by_class = {}
    skipped = 0
    jgi_pending = []
    for m in rows:
        if m["entity_type"] == "virus_sequence":
            url = m.get("download_url", "")
            filename = m.get("source_filename") or (m.get("sequence_accession") + ".fna")
            if not url:
                skipped += 1
                continue
            rec = {
                "entity_id": m["entity_id"], "source": m["source"],
                "klass": "virus", "group_name": "viral",
                "file_type": "viral_segment_fna", "url": url, "backup_url": "",
                "md5_url": "", "expected_md5": "",
                "local_dir": os.path.join("sequence", "virus", m.get("sequence_accession", "unknown")),
                "filename": filename, "fna_stem": _filename_stem(filename),
                "size_hint": "0",
            }
            by_class.setdefault("virus", []).append(rec)
            continue
        if m["entity_type"] != "assembly":
            continue
        if m.get("source") in ("jgi", "jgi_gold"):
            url = m.get("download_url", "")
            filename = m.get("source_filename", "")
            if not filename and url:
                filename = os.path.basename(urlparse(url).path)
            if not url or not filename.endswith((".fna", ".fna.gz", "_genomic.fna", "_genomic.fna.gz")):
                skipped += 1
                jgi_pending.append({
                    "entity_id": m.get("entity_id", ""),
                    "source_native_id": m.get("source_native_id", ""),
                    "organism_name": m.get("organism_name", ""),
                    "reason": "MISSING_DIRECT_FASTA_URL_OR_FILENAME",
                    "download_url": url, "source_filename": filename,
                })
                continue
            native = m.get("source_native_id") or m["entity_id"].replace(":", "_")
            rec = {
                "entity_id": m["entity_id"], "source": "jgi",
                "klass": m["klass"], "group_name": m["group_name"],
                "file_type": "genomic_fna", "url": url, "backup_url": "",
                "md5_url": "", "expected_md5": m.get("expected_md5", ""),
                "local_dir": os.path.join("jgi", re.sub(r"[^A-Za-z0-9_.-]", "_", native)),
                "filename": filename, "fna_stem": _filename_stem(filename),
                "size_hint": str(_safe_int(
                    m.get("source_file_size") or m.get("genome_size_ungapped")
                    or m.get("genome_size"), 0)),
            }
            by_class.setdefault(m["klass"], []).append(rec)
            continue
        ftp = m["ftp_path"].rstrip("/")
        if not ftp:
            continue
        dirname = ftp.split("/")[-1]
        # HTTPS 主路径 + FTP 协议 fallback
        https = ftp if ftp.startswith("http") else "https:" + ftp.split(":", 1)[-1]
        ftp_proto = re.sub(r"^https?://", "ftp://", ftp)
        rec = {
            "entity_id": m["entity_id"],
            "source": m.get("source", ""),
            "klass": m["klass"],
            "group_name": m["group_name"],
            "file_type": "genomic_fna",
            "url": "{}/{}_genomic.fna.gz".format(https, dirname),
            "backup_url": "{}/{}_genomic.fna.gz".format(ftp_proto, dirname),
            "md5_url": "{}/md5checksums.txt".format(https),
            "expected_md5": "",
            "local_dir": os.path.join("assembly", dirname),
            "filename": "{}_genomic.fna.gz".format(dirname),
            "fna_stem": dirname,
            "size_hint": str(_safe_int(
                m.get("genome_size_ungapped") or m.get("genome_size")
                or m.get("source_file_size"), 0)),
        }
        by_class.setdefault(m["klass"], []).append(rec)

    all_rows = []
    for klass, recs in by_class.items():
        write_tsv(os.path.join(paths.manifests, "download_{}.tsv".format(klass)),
                  recs, MANIFEST_COLUMNS)
        all_rows.extend(recs)
    write_tsv(os.path.join(paths.manifests, "download_all.tsv"),
              all_rows, MANIFEST_COLUMNS)
    write_tsv(os.path.join(paths.manifests, "jgi_download_pending.tsv"), jgi_pending,
              ["entity_id", "source_native_id", "organism_name", "reason",
               "download_url", "source_filename"])
    log("plan-downloads done: {} files across {} classes".format(
        len(all_rows), len(by_class)))
    if skipped:
        log("  JGI skipped without direct .fna[.gz] URL/filename: {}".format(skipped))


# --------------------------------------------------------------------------- #
# 6. download : 并行下载 + md5checksums 定位 + 断点续传
# --------------------------------------------------------------------------- #

_md5_cache = {}
_md5_lock = threading.Lock()


def _get_expected_md5(md5_url, filename, tmpdir, dl):
    """下载并缓存 md5checksums.txt，返回指定文件的 expected md5。"""
    with _md5_lock:
        if md5_url in _md5_cache:
            table = _md5_cache[md5_url]
        else:
            table = None
    if table is None:
        local = os.path.join(tmpdir, "md5_" + hashlib.md5(md5_url.encode()).hexdigest())
        table = {}
        if fetch_text(md5_url, local, timeout=120, retries=dl["max_retries"],
                      delay=dl["retry_delay"]):
            with open(local, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    parts = line.split()
                    if len(parts) >= 2:
                        table[os.path.basename(parts[1].lstrip("./"))] = parts[0]
        with _md5_lock:
            _md5_cache[md5_url] = table
    return table.get(filename)


# --------------------------------------------------------------------------- #
# NCBI EFetch batch downloader（仅 virus_sequence）
# --------------------------------------------------------------------------- #

_ncbi_api_rate_lock = threading.Lock()
_ncbi_api_next_request = 0.0


def _ncbi_api_key(cfg):
    """只从环境变量读取 API key；绝不返回给日志。"""
    api = cfg.get("ncbi_api") or {}
    env_name = api.get("api_key_env", "NCBI_API_KEY")
    return os.environ.get(env_name, "").strip()


def _ncbi_api_wait(cfg):
    """跨 worker 的全局限速，默认留在 API-key 10 req/s 上限以内。"""
    global _ncbi_api_next_request
    api = cfg.get("ncbi_api") or {}
    rps = max(0.1, float(api.get("max_requests_per_second", 8.0)))
    with _ncbi_api_rate_lock:
        now = time.monotonic()
        wait = max(0.0, _ncbi_api_next_request - now)
        _ncbi_api_next_request = max(now, _ncbi_api_next_request) + 1.0 / rps
    if wait:
        time.sleep(wait)


def _virus_accession(rec):
    value = rec.get("fna_stem") or _filename_stem(rec.get("filename", ""))
    return value.strip()


def _fasta_accession(header):
    """兼容 `>NC_...` 与 `>ref|NC_...|` 等 EFetch header。"""
    token = header.split()[0]
    parts = [p for p in token.split("|") if p]
    for part in parts:
        if _VIRUS_ACC_RE.fullmatch(part):
            return part
    return token


def _parse_fasta_text(text):
    header = None
    seq = []
    for line in text.splitlines():
        if line.startswith(">"):
            if header is not None:
                yield _fasta_accession(header), header, "".join(seq)
            header = line[1:].strip()
            seq = []
        elif header is not None:
            seq.append(line.strip())
    if header is not None:
        yield _fasta_accession(header), header, "".join(seq)


def _write_single_fasta(path, header, seq):
    with open(path, "w", encoding="ascii", newline="\n") as fh:
        fh.write(">" + header + "\n")
        for i in range(0, len(seq), 80):
            fh.write(seq[i:i + 80] + "\n")


def _download_virus_api_batch(batch, cfg, paths, writer_q):
    """通过一次 EFetch POST 获取一批 accession，返回逐 entity 的结果列表。"""
    api = cfg.get("ncbi_api") or {}
    key = _ncbi_api_key(cfg)
    vcfg = cfg.get("virus", {})
    results = []
    pending = []

    # 文件级 resume：批请求前先排除已经完整落盘的 accession。
    for rec in batch:
        dest_dir = os.path.join(paths.store, rec["local_dir"])
        dest = os.path.join(dest_dir, rec["filename"])
        if os.path.exists(dest) and fasta_composition_sane(
                dest, vcfg.get("max_n_fraction", 0.05),
                vcfg.get("max_other_fraction", 0.01)):
            results.append((rec["entity_id"], "cached_no_md5", ""))
        else:
            pending.append(rec)
    if not pending:
        return results

    ids = [_virus_accession(r) for r in pending]
    payload = {
        "db": "nuccore", "id": ",".join(ids),
        "rettype": "fasta", "retmode": "text",
        "tool": api.get("tool", "tiara2_pipeline"),
    }
    if key:
        payload["api_key"] = key
    email = os.environ.get(api.get("email_env", "NCBI_EMAIL"), "").strip()
    if email:
        payload["email"] = email

    endpoint = api.get("efetch_url", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi")
    retries = max(1, int(api.get("max_retries", 3)))
    timeout = max(int(api.get("connect_timeout", 20)),
                  int(api.get("read_timeout", 180)))
    text = None
    for attempt in range(1, retries + 1):
        try:
            _ncbi_api_wait(cfg)
            request = Request(
                endpoint, data=urlencode(payload).encode("ascii"), method="POST",
                headers={"User-Agent": api.get("tool", "tiara2_pipeline") + "/1.0",
                         "Accept": "text/plain"})
            with urlopen(request, timeout=timeout) as response:
                raw = response.read()
            text = raw.decode("utf-8", errors="replace")
            if text.lstrip().startswith(">"):
                break
            text = None
            raise RuntimeError("EFetch response is not FASTA")
        except Exception as exc:  # noqa
            # 不输出请求 body/API key。
            log("    EFetch batch err ({}/{}; {} ids): {}".format(
                attempt, retries, len(ids), exc))
            if attempt < retries:
                time.sleep(float(api.get("retry_delay", 3)) * attempt)

    if text is None:
        if api.get("fallback_to_sviewer", True):
            results.extend(_download_one(r, cfg, paths, writer_q) for r in pending)
        else:
            results.extend((r["entity_id"], "failed", "") for r in pending)
        return results

    fetched = {}
    for acc, header, seq in _parse_fasta_text(text):
        fetched[acc] = (header, seq)
        fetched.setdefault(acc.split(".")[0], (header, seq))

    for rec, acc in zip(pending, ids):
        item = fetched.get(acc) or fetched.get(acc.split(".")[0])
        if not item:
            results.append((rec["entity_id"], "failed", ""))
            continue
        dest_dir = os.path.join(paths.store, rec["local_dir"])
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, rec["filename"])
        part = dest + ".part"
        header, seq = item
        try:
            _write_single_fasta(part, header, seq)
            if not fasta_composition_sane(
                    part, vcfg.get("max_n_fraction", 0.05),
                    vcfg.get("max_other_fraction", 0.01)):
                os.remove(part)
                results.append((rec["entity_id"], "failed", ""))
                continue
            os.replace(part, dest)
            _backup(dest, dest_dir, rec, cfg, paths)
            results.append((rec["entity_id"], "ok", "ncbi_efetch_batch"))
        except Exception as exc:  # noqa
            if os.path.exists(part):
                os.remove(part)
            log("    EFetch write err {}: {}".format(rec["entity_id"], exc))
            results.append((rec["entity_id"], "failed", ""))
    return results


def _run_virus_api_pool(jobs, cfg, paths, writer_q, on_result):
    api = cfg.get("ncbi_api") or {}
    batch_size = max(1, min(200, int(api.get("batch_size", 100))))
    workers = max(1, int(api.get("workers", 4)))
    batches = [jobs[i:i + batch_size] for i in range(0, len(jobs), batch_size)]
    log("  [virus-api] start: {} jobs, {} batches, workers={}, batch_size={}".format(
        len(jobs), len(batches), workers, batch_size))
    with cf.ThreadPoolExecutor(max_workers=workers,
                               thread_name_prefix="ncbi_efetch") as ex:
        inflight = {}
        batch_iter = iter(batches)
        for batch in itertools.islice(batch_iter, workers * 2):
            inflight[ex.submit(_download_virus_api_batch,
                               batch, cfg, paths, writer_q)] = batch
        while inflight:
            completed, _ = cf.wait(inflight, return_when=cf.FIRST_COMPLETED)
            for fut in completed:
                source_batch = inflight.pop(fut)
                try:
                    batch_results = fut.result()
                except Exception as exc:  # noqa
                    log("    [virus-api] worker error: {}".format(exc))
                    batch_results = [(r["entity_id"], "failed", "")
                                     for r in source_batch]
                for eid, result, url in batch_results:
                    on_result(eid, result, url)
                try:
                    batch = next(batch_iter)
                except StopIteration:
                    batch = None
                if batch is not None:
                    inflight[ex.submit(_download_virus_api_batch,
                                       batch, cfg, paths, writer_q)] = batch
    log("  [virus-api] done")


def _download_one(rec, cfg, paths, writer_q):
    dl = cfg["download"]
    dest_dir = os.path.join(paths.store, rec["local_dir"])
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, rec["filename"])
    expected = rec.get("expected_md5") or (
        _get_expected_md5(rec["md5_url"], rec["filename"], paths.tmp, dl)
        if rec.get("md5_url") else None)

    # resume 时也要重校 expected MD5（不能仅看文件存在）
    compressed = dest.endswith(".gz")
    compression_ok = lambda p: gzip_ok(p) if compressed else os.path.getsize(p) > 0
    if os.path.exists(dest):
        if expected and md5_of(dest) == expected and compression_ok(dest):
            return (rec["entity_id"], "cached", "")
        if not expected and compression_ok(dest):
            if rec.get("klass") == "virus":
                vcfg = cfg.get("virus", {})
                cached_ok = fasta_composition_sane(
                    dest, vcfg.get("max_n_fraction", 0.05),
                    vcfg.get("max_other_fraction", 0.01))
            else:
                cached_ok = fasta_sane(dest)
            if cached_ok:
                return (rec["entity_id"], "cached_no_md5", "")

    # size-aware 下载参数（timeout / connections）
    profile = download_profile(rec, cfg)
    # 默认关闭 FTP fallback：backup_url 与主 URL 指向同一 ftp.ncbi.nlm.nih.gov，
    # 并非独立镜像，反复 fallback 只会放大超时。需要时在 config 中开启。
    enable_ftp = bool(dl.get("enable_ftp_fallback", False))
    urls = [rec["url"]]
    if enable_ftp and rec.get("backup_url"):
        urls.append(rec["backup_url"])
    for url in urls:
        if not url:
            continue
        ok = http_download(url, dest, profile["timeout"], dl["max_retries"],
                           dl["retry_delay"], resume=True,
                           connections_per_file=profile["connections_per_file"])
        if not ok:
            continue
        if not compression_ok(dest):
            os.remove(dest); continue
        if expected and md5_of(dest) != expected:
            os.remove(dest); continue
        if rec.get("klass") == "virus":
            vcfg = cfg.get("virus", {})
            sequence_ok = fasta_composition_sane(
                dest, vcfg.get("max_n_fraction", 0.05),
                vcfg.get("max_other_fraction", 0.01))
        else:
            sequence_ok = fasta_sane(dest)
        if not sequence_ok:
            os.remove(dest); continue
        # backup
        _backup(dest, dest_dir, rec, cfg, paths)
        return (rec["entity_id"], "ok", url)
    return (rec["entity_id"], "failed", "")


def _backup(dest, dest_dir, rec, cfg, paths):
    if not cfg["storage"].get("enable_backup"):
        return
    bdir = os.path.join(paths.backup_root, "store", rec["local_dir"])
    try:
        os.makedirs(bdir, exist_ok=True)
        bdest = os.path.join(bdir, rec["filename"])
        shutil.copy2(dest, bdest)
        if md5_of(bdest) != md5_of(dest):
            log("    !! backup md5 mismatch: {}".format(rec["entity_id"]))
    except Exception as exc:  # noqa
        log("    backup err {}: {}".format(rec["entity_id"], exc))


def _run_download_pool(label, jobs, conc, cfg, paths, writer_q, on_result):
    """单个 size 通道的下载池：有界 Future（conc*4 inflight）+ 回调写状态。"""
    if not jobs:
        return
    conc = max(1, int(conc))
    max_inflight = max(conc * 4, conc)
    job_iter = iter(jobs)
    log("  [{}] start: {} jobs, concurrency={}".format(label, len(jobs), conc))
    with cf.ThreadPoolExecutor(max_workers=conc,
                               thread_name_prefix="dl_" + label) as ex:
        inflight = {}
        for rec in itertools.islice(job_iter, max_inflight):
            inflight[ex.submit(_download_one, rec, cfg, paths, writer_q)] = rec
        while inflight:
            completed, _ = cf.wait(inflight, return_when=cf.FIRST_COMPLETED)
            for fut in completed:
                source_rec = inflight.pop(fut, {})
                try:
                    eid, res, url = fut.result()
                except Exception as exc:  # noqa
                    eid, res, url = source_rec.get("entity_id", "UNKNOWN"), "failed", ""
                    log("    [{}] worker error {}: {}".format(label, eid, exc))
                on_result(eid, res, url)
                try:
                    rec = next(job_iter)
                except StopIteration:
                    rec = None
                if rec is not None:
                    inflight[ex.submit(_download_one, rec, cfg, paths, writer_q)] = rec
    log("  [{}] done".format(label))


def stage_download(cfg, paths, only_class=None):
    log("== download ==")
    manifest = os.path.join(paths.manifests, "download_all.tsv")
    recs = read_tsv(manifest)
    if only_class:
        recs = [r for r in recs if r["klass"] == only_class]
    if not recs:
        log("!! manifest 为空，先运行 plan-downloads")
        return
    status_path = paths.status_tsv
    done = {r["entity_id"] for r in read_tsv(status_path)
            if r.get("result") in ("ok", "cached", "cached_no_md5")}
    todo = [r for r in recs if r["entity_id"] not in done]
    log("  total={} done={} todo={}".format(len(recs), len(done), len(todo)))

    dl = cfg["download"]
    writer_q = queue.Queue()
    pending_results = []
    merged = {r["entity_id"]: r for r in read_tsv(status_path)}
    flush_every = int(dl.get("status_flush_every", 500))
    lock = threading.Lock()
    counters = {"ok": 0, "failed": 0, "n": 0}

    def flush_status_locked():
        if not pending_results:
            return
        for row in pending_results:
            merged[row["entity_id"]] = row
        write_tsv(status_path, list(merged.values()),
                  ["entity_id", "result", "url", "ts"])
        pending_results[:] = []

    def on_result(eid, res, url):
        # 线程安全：三个 size 通道共享状态，需加锁。
        with lock:
            pending_results.append({
                "entity_id": eid, "result": res, "url": url,
                "ts": datetime.now(timezone.utc).isoformat()})
            counters["n"] += 1
            if res in ("ok", "cached", "cached_no_md5"):
                counters["ok"] += 1
            else:
                counters["failed"] += 1
            n = counters["n"]
            if n % 50 == 0:
                log("  progress {}/{} (ok={} failed={})".format(
                    n, len(todo), counters["ok"], counters["failed"]))
            if n % flush_every == 0:
                flush_status_locked()

    # size-aware：将 todo 拆分为三个通道，小文件高并发、大文件低并发。
    size_aware = bool(dl.get("size_aware", True))
    if not size_aware:
        conc = int(dl.get("concurrency", 16))
        _run_download_pool("all", todo, conc, cfg, paths, writer_q, on_result)
        with lock:
            flush_status_locked()
        log("download done: ok/cached={} failed={}".format(
            counters["ok"], counters["failed"]))
        return

    # 有 API key 时，virus_sequence 从普通 small 池中抽出，走批量 EFetch。
    api_cfg = cfg.get("ncbi_api") or {}
    api_enabled = bool(api_cfg.get("enabled", True))
    api_key = _ncbi_api_key(cfg) if api_enabled else ""
    virus_api_jobs = []
    ordinary_todo = []
    for rec in todo:
        if (api_enabled and api_key and
                rec.get("file_type") == "viral_segment_fna"):
            virus_api_jobs.append(rec)
        else:
            ordinary_todo.append(rec)
    if api_enabled and not api_key and any(
            r.get("file_type") == "viral_segment_fna" for r in todo):
        log("  !! {} 未设置：病毒序列回退为普通 URL 下载".format(
            api_cfg.get("api_key_env", "NCBI_API_KEY")))

    buckets = {"small": [], "medium": [], "large": []}
    for rec in ordinary_todo:
        buckets[download_size_class(rec, cfg)].append(rec)
    log("  split: virus_api={} small={} medium={} large={}".format(
        len(virus_api_jobs), len(buckets["small"]),
        len(buckets["medium"]), len(buckets["large"])))

    pool_specs = [
        ("small", buckets["small"], dl.get("small_concurrency", 16)),
        ("medium", buckets["medium"], dl.get("medium_concurrency", 4)),
        ("large", buckets["large"], dl.get("large_concurrency", 2)),
    ]
    active = [(lbl, jobs, c) for lbl, jobs, c in pool_specs if jobs]
    # API 通道与三个 size 通道并行；共享状态写入由 on_result 加锁保护。
    outer_workers = len(active) + (1 if virus_api_jobs else 0)
    with cf.ThreadPoolExecutor(max_workers=max(1, outer_workers),
                               thread_name_prefix="dl_pool") as outer:
        futs = [outer.submit(_run_download_pool, lbl, jobs, c,
                             cfg, paths, writer_q, on_result)
                for lbl, jobs, c in active]
        if virus_api_jobs:
            futs.append(outer.submit(_run_virus_api_pool, virus_api_jobs,
                                     cfg, paths, writer_q, on_result))
        for f in cf.as_completed(futs):
            f.result()
    with lock:
        flush_status_locked()
    log("download done: ok/cached={} failed={}".format(
        counters["ok"], counters["failed"]))


# --------------------------------------------------------------------------- #
# 7. verify : 对已下载文件重新 gzip + FASTA 校验
# --------------------------------------------------------------------------- #

def stage_verify(cfg, paths):
    log("== verify ==")
    recs = read_tsv(os.path.join(paths.manifests, "download_all.tsv"))
    problems = []

    def check(rec):
        dest = os.path.join(paths.store, rec["local_dir"], rec["filename"])
        if not os.path.exists(dest):
            return (rec["entity_id"], "missing")
        if dest.endswith(".gz") and not gzip_ok(dest):
            return (rec["entity_id"], "gzip_bad")
        if rec.get("klass") == "virus":
            vcfg = cfg.get("virus", {})
            ok = fasta_composition_sane(
                dest, vcfg.get("max_n_fraction", 0.05),
                vcfg.get("max_other_fraction", 0.01))
        else:
            ok = fasta_sane(dest)
        if not ok:
            return (rec["entity_id"], "fasta_bad")
        return None

    with cf.ThreadPoolExecutor(max_workers=cfg["download"]["concurrency"]) as ex:
        for r in ex.map(check, recs):
            if r:
                problems.append({"entity_id": r[0], "problem": r[1]})
    write_tsv(os.path.join(paths.qc, "verify_problems.tsv"),
              problems, ["entity_id", "problem"])
    log("verify done: {} problems / {} files".format(len(problems), len(recs)))


# --------------------------------------------------------------------------- #
# 8. export : 从总表派生各类别表 + accession 清单
# --------------------------------------------------------------------------- #

def stage_export(cfg, paths):
    log("== export ==")
    inventory = read_tsv(paths.master_tsv)
    selected = read_tsv(os.path.join(paths.master, "selected.tsv"))
    selected_ids = {m["entity_id"] for m in selected}
    classes = sorted({m["klass"] for m in inventory})
    all_acc = []
    for klass in classes:
        full = [m for m in inventory if m["klass"] == klass]
        chosen = [m for m in selected if m["klass"] == klass]
        write_tsv(os.path.join(paths.classes, klass + ".tsv"), full, MASTER_COLUMNS)
        write_tsv(os.path.join(paths.classes, "selected_" + klass + ".tsv"), chosen, MASTER_COLUMNS)
        accs = [m["entity_id"] for m in chosen]
        _write_lines(os.path.join(paths.accessions, klass + ".txt"), accs)
        all_acc.extend(accs)
    _write_lines(os.path.join(paths.accessions, "all.txt"), all_acc)
    log("export done: {} classes, {} selected accessions".format(len(classes), len(all_acc)))



def _write_lines(path, lines):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + ("\n" if lines else ""))
    log("  wrote {} lines -> {}".format(len(lines), path))


# --------------------------------------------------------------------------- #
# 9. split : taxonomy-aware（ANI group → species_taxid）确定性划分
# --------------------------------------------------------------------------- #

def stage_split(cfg, paths):
    log("== split ==")
    selected_path = os.path.join(paths.master, "selected.tsv")
    rows = read_tsv(selected_path)
    if not rows:
        raise RuntimeError("selected.tsv is empty; run select before split")
    qc = {q["entity_id"]: q for q in read_tsv(paths.qc_tsv)}
    test_pct = cfg["split"]["test_pct"]
    seed = cfg["split"]["seed"]
    out = []
    for m in rows:
        sgid = qc.get(m["entity_id"], {}).get("split_group_id") or m.get("species_taxid") or m["entity_id"]
        is_mag = bool(_mag_evidence(m))
        if m.get("dataset_role") == "test_only" or is_mag:
            split = "evaluation"
            method = "forced_mag_evaluation" if is_mag else "forced_test_only"
        else:
            if is_mag:
                raise RuntimeError("MAG reached isolate split: {}".format(m["entity_id"]))
            h = hashlib.md5(("{}:{}".format(seed, sgid)).encode()).hexdigest()
            bucket = int(h[:8], 16) % 100
            split = "test" if bucket < test_pct else "train"
            method = "ani_or_species_hash"
        out.append({"entity_id": m["entity_id"], "klass": m["klass"],
                    "split_group_id": sgid, "split": split,
                    "split_method": method, "seed": seed})
    write_tsv(os.path.join(paths.master, "split_assignments.tsv"), out,
              ["entity_id", "klass", "split_group_id", "split", "split_method", "seed"])
    selected_by_id = {m["entity_id"]: m for m in rows}
    assert not any(_mag_evidence(selected_by_id[r["entity_id"]]) and r["split"] != "evaluation" for r in out)
    n_test = sum(r["split"] == "test" for r in out)
    n_eval = sum(r["split"] == "evaluation" for r in out)
    log("split done: train={} test={} evaluation={}".format(len(out)-n_test-n_eval, n_test, n_eval))



# --------------------------------------------------------------------------- #
# 10. fetch-organelle : RefSeq bulk FASTA + GBFF 流式解析
# --------------------------------------------------------------------------- #

ORGANELLE_SOURCES = {
    "mitochondrion": {
        "fna": ["mitochondrion.1.1.genomic.fna.gz"],
        "gbff": ["mitochondrion.1.genomic.gbff.gz"],
        "base": FTP + "/genomes/refseq/mitochondrion/",
        "organelle_type": "mitochondrion",
    },
    "plastid": {
        "fna": ["plastid.1.1.genomic.fna.gz", "plastid.1.2.genomic.fna.gz",
                "plastid.2.1.genomic.fna.gz", "plastid.2.2.genomic.fna.gz",
                "plastid.3.1.genomic.fna.gz"],
        "gbff": ["plastid.1.genomic.gbff.gz", "plastid.2.genomic.gbff.gz",
                 "plastid.3.genomic.gbff.gz"],
        "base": FTP + "/genomes/refseq/plastid/",
        "organelle_type": "plastid",
    },
}

ORGANELLE_RECORD_COLUMNS = [
    "group_name", "sequence_accession", "accession_base", "organelle_type",
    "taxid", "species_taxid", "species_taxid_method", "organism_name",
    "definition", "gbff_source_file", "fasta_source_file",
    "gbff_length", "fasta_length", "metadata_matched", "fasta_present",
    "selected", "filter_reason", "local_fasta",
]

ORGANELLE_STATS_COLUMNS = [
    "group_name", "bulk_fna_files", "bulk_gbff_files", "gbff_records",
    "duplicate_gbff_accessions", "fasta_records", "duplicate_fasta_accessions",
    "metadata_matched", "missing_metadata", "missing_taxid", "below_min_len",
    "species_cap_filtered", "kept_records", "distinct_taxids",
]


def stage_fetch_organelle(cfg, paths):
    log("== fetch-organelle ==")
    dl = cfg["download"]
    minlen = cfg["organelle"]["min_len"]
    max_per_sp = cfg["organelle"]["max_per_species"]
    checksum_rows = []
    master_rows = []
    inventory_rows = []
    stats_rows = []

    groups = [g for g in cfg["groups"] if g["kind"] == "organelle"]
    for g in groups:
        name = g["group"]
        klass = g["klass"]
        spec = ORGANELLE_SOURCES[name]
        raw_dir = os.path.join(paths.store, "organelle", "_bulk", name)
        os.makedirs(raw_dir, exist_ok=True)

        # ---- 下载 bulk 文件（断点续传 + gzip 校验 + 本地 hash）----
        for fn in spec["fna"] + spec["gbff"]:
            dest = os.path.join(raw_dir, fn)
            url = spec["base"] + fn
            if os.path.exists(dest) and gzip_ok(dest):
                log("  cached {}".format(fn))
            else:
                log("  fetch {}".format(url))
                if not http_download(url, dest, dl["timeout"], dl["max_retries"],
                                     dl["retry_delay"], resume=True):
                    log("  !! 失败 {}".format(url)); continue
                if not gzip_ok(dest):
                    log("  !! gzip 损坏 {}".format(fn)); os.remove(dest); continue
            checksum_rows.append({
                "file": fn, "organelle_type": name,
                "md5": md5_of(dest), "sha256": sha256_of(dest),
                "size": os.path.getsize(dest),
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
            })

        # ---- 解析 GBFF 获取 metadata ----
        meta = {}   # accession(no version) -> dict
        gbff_records = duplicate_gbff = 0
        for fn in spec["gbff"]:
            path = os.path.join(raw_dir, fn)
            if os.path.exists(path):
                n_rec, n_dup = _parse_gbff(
                    path, spec["organelle_type"], meta, source_file=fn)
                gbff_records += n_rec
                duplicate_gbff += n_dup
        log("  {} GBFF metadata records: {}".format(name, len(meta)))

        # ---- 流式遍历 FASTA；每条都写 inventory，再按规则选择训练记录 ----
        per_species = {}
        kept = 0
        fasta_records = duplicate_fasta = matched = missing_meta = 0
        missing_taxid = below_min = cap_filtered = 0
        seen_fasta = set()
        taxids = set()
        for fn in spec["fna"]:
            path = os.path.join(raw_dir, fn)
            if not os.path.exists(path):
                continue
            for acc, header, seq in _iter_fasta(path):
                fasta_records += 1
                base = acc.split(".")[0]
                info = meta.get(base) or meta.get(acc) or {}
                is_match = bool(info)
                if is_match:
                    matched += 1
                else:
                    missing_meta += 1
                taxid = info.get("taxid", "")
                sp = info.get("species_taxid") or taxid
                if taxid:
                    taxids.add(taxid)
                else:
                    missing_taxid += 1

                reasons = []
                selected = True
                if acc in seen_fasta:
                    duplicate_fasta += 1
                    reasons.append("DUPLICATE_FASTA_ACCESSION")
                    selected = False
                seen_fasta.add(acc)
                if len(seq) < minlen:
                    below_min += 1
                    reasons.append("BELOW_MIN_LENGTH")
                    selected = False

                # 缺 taxid 时按 accession 独立计配额，不能把所有 NA 当成一个物种。
                quota_key = sp if sp else "ACC:" + acc
                if selected and max_per_sp and per_species.get(quota_key, 0) >= max_per_sp:
                    cap_filtered += 1
                    reasons.append("SPECIES_CAP")
                    selected = False

                local_fasta = ""
                if selected:
                    per_species[quota_key] = per_species.get(quota_key, 0) + 1
                    kept += 1
                    out_path = os.path.join(paths.store, "organelle", acc + ".fna.gz")
                    with gzip.open(out_path, "wt", encoding="utf-8") as fh:
                        fh.write(">{} {}\n".format(acc, info.get("organism_name", "")))
                        for i in range(0, len(seq), 80):
                            fh.write(seq[i:i + 80] + "\n")
                    local_fasta = os.path.relpath(out_path, paths.root)
                    master_rows.append({
                        "entity_id": acc,
                        "entity_type": "organelle",
                        "klass": klass,
                        "group_name": name,
                        "source": "refseq_bulk",
                        "assembly_accession": "",
                        "taxid": taxid,
                        # GBFF 只有 taxon xref；在 taxonomy enrichment 前作为代理值。
                        "species_taxid": sp,
                        "organism_name": info.get("organism_name", ""),
                        "sequence_accession": acc,
                        "organelle_type": info.get("organelle_type", spec["organelle_type"]),
                        "sequence_length": len(seq),
                        "refseq_source_file": fn,
                        "definition": info.get("definition", ""),
                    })

                inventory_rows.append({
                    "group_name": name,
                    "sequence_accession": acc,
                    "accession_base": base,
                    "organelle_type": info.get("organelle_type", spec["organelle_type"]),
                    "taxid": taxid,
                    "species_taxid": sp,
                    "species_taxid_method": "gbff_taxid_proxy" if sp else "missing",
                    "organism_name": info.get("organism_name", ""),
                    "definition": info.get("definition", ""),
                    "gbff_source_file": info.get("source_file", ""),
                    "fasta_source_file": fn,
                    "gbff_length": info.get("length", ""),
                    "fasta_length": len(seq),
                    "metadata_matched": int(is_match),
                    "fasta_present": 1,
                    "selected": int(selected),
                    "filter_reason": ",".join(reasons),
                    "local_fasta": local_fasta,
                })

        # GBFF 有记录但 FASTA 分卷中没有匹配序列的情况也保留在 inventory。
        seen_bases = {a.split(".")[0] for a in seen_fasta}
        for base, info in meta.items():
            if base in seen_bases:
                continue
            taxid = info.get("taxid", "")
            inventory_rows.append({
                "group_name": name,
                "sequence_accession": info.get("accession", base),
                "accession_base": base,
                "organelle_type": info.get("organelle_type", spec["organelle_type"]),
                "taxid": taxid,
                "species_taxid": info.get("species_taxid") or taxid,
                "species_taxid_method": "gbff_taxid_proxy" if taxid else "missing",
                "organism_name": info.get("organism_name", ""),
                "definition": info.get("definition", ""),
                "gbff_source_file": info.get("source_file", ""),
                "fasta_source_file": "",
                "gbff_length": info.get("length", ""),
                "fasta_length": "",
                "metadata_matched": 1,
                "fasta_present": 0,
                "selected": 0,
                "filter_reason": "FASTA_NOT_FOUND",
                "local_fasta": "",
            })

        stats_rows.append({
            "group_name": name,
            "bulk_fna_files": len(spec["fna"]),
            "bulk_gbff_files": len(spec["gbff"]),
            "gbff_records": gbff_records,
            "duplicate_gbff_accessions": duplicate_gbff,
            "fasta_records": fasta_records,
            "duplicate_fasta_accessions": duplicate_fasta,
            "metadata_matched": matched,
            "missing_metadata": missing_meta,
            "missing_taxid": missing_taxid,
            "below_min_len": below_min,
            "species_cap_filtered": cap_filtered,
            "kept_records": kept,
            "distinct_taxids": len(taxids),
        })
        log("  {} kept sequences: {}".format(name, kept))

    write_tsv(os.path.join(paths.master, "organelle_entities.tsv"),
              master_rows, MASTER_COLUMNS)
    write_tsv(os.path.join(paths.master, "organelle_source_checksums.tsv"),
              checksum_rows,
              ["file", "organelle_type", "md5", "sha256", "size", "downloaded_at"])
    write_tsv(os.path.join(paths.master, "organelle_all_records.tsv"),
              inventory_rows, ORGANELLE_RECORD_COLUMNS)
    write_tsv(os.path.join(paths.master, "organelle_parse_stats.tsv"),
              stats_rows, ORGANELLE_STATS_COLUMNS)
    log("fetch-organelle done: {} organelle records".format(len(master_rows)))


def _iter_fasta(path):
    """流式读取 gzip FASTA，逐条 yield (accession, header, sequence)。"""
    opener = gzip.open if path.endswith(".gz") else open
    acc = header = None
    seq = []
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith(">"):
                if acc is not None:
                    yield acc, header, "".join(seq)
                header = line[1:].strip()
                acc = header.split()[0]
                seq = []
            else:
                seq.append(line.strip())
        if acc is not None:
            yield acc, header, "".join(seq)


def _parse_gbff(path, default_type, meta, source_file=""):
    """流式解析 GBFF：每个 record 提取 VERSION/ORGANISM/taxon/organelle/length。"""
    opener = gzip.open if path.endswith(".gz") else open
    cur = {}
    records = duplicates = 0
    in_source = False
    organism_lines = []
    in_organism = False
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("LOCUS"):
                cur = {"organelle_type": default_type}
                parts = line.split()
                # LOCUS  <name> <length> bp ...
                for i, p in enumerate(parts):
                    if p == "bp" and i > 0:
                        cur["length"] = parts[i - 1]
                organism_lines = []
                in_organism = False
                in_source = False
            elif line.startswith("VERSION"):
                cur["accession"] = line.split()[1] if len(line.split()) > 1 else ""
            elif line.startswith("DEFINITION"):
                cur["definition"] = line[len("DEFINITION"):].strip()
            elif line.startswith("  ORGANISM"):
                cur["organism"] = line[len("  ORGANISM"):].strip()
                in_organism = True
            elif line.strip().startswith("/db_xref=\"taxon:"):
                m = re.search(r'taxon:(\d+)', line)
                if m:
                    cur["taxid"] = m.group(1)
            elif line.strip().startswith("/organelle="):
                m = re.search(r'/organelle="([^"]+)"', line)
                if m:
                    val = m.group(1).lower()
                    if "mitochond" in val:
                        cur["organelle_type"] = "mitochondrion"
                    elif "plastid" in val or "chloroplast" in val:
                        cur["organelle_type"] = "plastid"
            elif in_organism and line.startswith("            "):
                pass  # taxonomy lineage lines；qualifier 已在上方优先解析
            elif line.startswith("//"):
                acc = cur.get("accession", "")
                if acc:
                    records += 1
                    base = acc.split(".")[0]
                    if base in meta:
                        duplicates += 1
                    meta[base] = {
                        "accession": acc,
                        "taxid": cur.get("taxid", ""),
                        "species_taxid": cur.get("taxid", ""),
                        "species_taxid_method": "gbff_taxid_proxy",
                        "organism_name": cur.get("organism", ""),
                        "organelle_type": cur.get("organelle_type", default_type),
                        "definition": cur.get("definition", ""),
                        "length": cur.get("length", ""),
                        "source_file": source_file or os.path.basename(path),
                    }
                cur = {}
    return records, duplicates


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# 11. organize : 物���下游 03_04 期望的 raw/<group>/<split>/ + select/
#     - raw/ 里全部用 symlink 指向 store/（不重复占磁盘）
#     - group 名 = NCBI group（与 groups.tsv / 下游 GROUP_META 一致）
#     - taxid.tsv 的 key = 下游 accession_from_path 解析出的 stem
# --------------------------------------------------------------------------- #

SUPERGROUP_FALLBACK = {
    "bacteria": ("prok", "Bacteria"),
    "archaea": ("prok", "Archaea"),
    "fungi": ("euk", "Opisthokonta-Fungi"),
    "protozoa": ("euk", "Protist(SAR/Excavata/Amoebozoa)"),
    "plant": ("euk", "Archaeplastida"),
    "invertebrate": ("euk", "Opisthokonta-Metazoa"),
    "vertebrate_other": ("euk", "Opisthokonta-Metazoa"),
    "vertebrate_mammalian": ("euk", "Host-Mammalia"),
    "mitochondrion": ("euk", "Organelle-Mito"),
    "plastid": ("euk", "Organelle-Plastid"),
}


def _clean_symlinks(root):
    """删除 raw/ 下所有 symlink（保留真实文件与目录），使 organize 幂等。"""
    if not os.path.isdir(root):
        return
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            if os.path.islink(p):
                os.unlink(p)


def _rel_symlink(target, link):
    link_dir = os.path.dirname(link)
    os.makedirs(link_dir, exist_ok=True)
    rel = os.path.relpath(target, link_dir)
    if os.path.islink(link) or os.path.exists(link):
        try:
            os.unlink(link)
        except OSError:
            pass
    os.symlink(rel, link)


def stage_organize(cfg, paths):
    log("== organize ==")
    master = {m["entity_id"]: m for m in read_tsv(paths.master_tsv)}
    qc = {q["entity_id"]: q for q in read_tsv(paths.qc_tsv)}
    split = {r["entity_id"]: r["split"]
             for r in read_tsv(os.path.join(paths.index, "split_assignments.tsv"))}
    manifest = {r["entity_id"]: r
                for r in read_tsv(os.path.join(paths.manifests, "download_all.tsv"))}
    status = {r["entity_id"]: r.get("result")
              for r in read_tsv(paths.status_tsv)}

    _clean_symlinks(paths.raw)
    _clean_symlinks(paths.evaluation)
    taxid_rows = []
    evaluation_rows = []
    linked = 0
    missing = 0
    per_group = {}
    for eid, m in master.items():
        if qc.get(eid, {}).get("qc_status") != "accepted":
            continue
        sp = split.get(eid)
        if sp not in ("train", "test", "evaluation"):
            continue
        group = m.get("group_name") or m.get("klass")
        if m.get("entity_type") == "assembly":
            rec = manifest.get(eid)
            if not rec:
                continue
            if status.get(eid) not in ("ok", "cached", "cached_no_md5"):
                missing += 1
                continue
            target = os.path.join(paths.store, rec["local_dir"], rec["filename"])
            stem = rec.get("fna_stem") or rec["filename"][:-len("_genomic.fna.gz")]
            linkname = rec["filename"]
        elif m.get("entity_type") == "organelle":
            linkname = eid + ".fna.gz"
            target = os.path.join(paths.store, "organelle", linkname)
            stem = eid
        else:  # virus_sequence 等 direct-sequence 实体
            rec = manifest.get(eid)
            if not rec:
                continue
            if status.get(eid) not in ("ok", "cached", "cached_no_md5"):
                missing += 1
                continue
            target = os.path.join(paths.store, rec["local_dir"], rec["filename"])
            stem = rec.get("fna_stem") or _filename_stem(rec["filename"])
            linkname = rec["filename"]
        if not os.path.exists(target):
            missing += 1
            continue
        if sp == "evaluation":
            collection = m.get("evaluation_collection") or group
            link = os.path.join(paths.evaluation, collection, linkname)
            evaluation_rows.append({
                "entity_id": eid, "collection": collection, "klass": m.get("klass", ""),
                "group_name": group, "source": m.get("source", ""),
                "species_taxid": m.get("species_taxid", ""),
                "organism_name": m.get("organism_name", ""), "path": link,
            })
        else:
            link = os.path.join(paths.raw, group, sp, linkname)
        _rel_symlink(target, link)
        linked += 1
        per_group[(group, sp)] = per_group.get((group, sp), 0) + 1
        if sp != "evaluation":
            taxid_rows.append((stem, m.get("species_taxid") or "NA"))

    # select/groups.tsv（group\tlabel\tsupergroup；下游依此推类别）
    groups_path = os.path.join(paths.select, "groups.tsv")
    with open(groups_path, "w", encoding="utf-8") as fh:
        fh.write("# group\tlabel\tsupergroup\n")
        for g in cfg["groups"]:
            if g.get("dataset_role") == "test_only" or g.get("klass") == "virus":
                continue
            grp = g["group"]
            fb = SUPERGROUP_FALLBACK.get(grp, ("euk", grp))
            fh.write("{}\t{}\t{}\n".format(
                grp, g.get("label") or fb[0], g.get("supergroup") or fb[1]))

    # select/taxid.tsv（key = 下游 accession_from_path 解析出的 stem）
    taxid_path = os.path.join(paths.select, "taxid.tsv")
    seen_stem = set()
    with open(taxid_path, "w", encoding="utf-8") as fh:
        for stem, sp in taxid_rows:
            if stem in seen_stem:
                continue
            seen_stem.add(stem)
            fh.write("{}\t{}\n".format(stem, sp))

    write_tsv(os.path.join(paths.evaluation, "metadata.tsv"), evaluation_rows,
              ["entity_id", "collection", "klass", "group_name", "source",
               "species_taxid", "organism_name", "path"])

    for (grp, sp), n in sorted(per_group.items()):
        log("    {:<22} {:<6} {}".format(grp, sp, n))
    log("  wrote {}".format(groups_path))
    log("  wrote {} ({} entries)".format(taxid_path, len(seen_stem)))
    log("organize done: linked={} missing_files={}".format(linked, missing))


def stage_status(cfg, paths):
    log("== status ==")
    def count(path):
        return max(0, sum(1 for _ in open(path, encoding="utf-8")) - 1) \
            if os.path.exists(path) else 0
    log("  master:      {}".format(count(paths.master_tsv)))
    log("  qc_flags:    {}".format(count(paths.qc_tsv)))
    for st in ("accepted", "quarantine", "rejected"):
        log("    {:<11} {}".format(st, count(os.path.join(paths.qc, st + ".tsv"))))
    log("  selected:    {}".format(count(os.path.join(paths.master, "selected.tsv"))))
    log("  manifest:    {}".format(count(os.path.join(paths.manifests, "download_all.tsv"))))
    log("  dl_status:   {}".format(count(paths.status_tsv)))
    def link_count(root):
        if not os.path.isdir(root):
            return 0
        return sum(1 for dp, _dn, fns in os.walk(root)
                   for fn in fns if os.path.islink(os.path.join(dp, fn)))
    log("  raw links:   {}".format(link_count(paths.raw)))
    log("  eval links:  {}".format(link_count(paths.evaluation)))
    log("  groups.tsv:  {}".format(count(os.path.join(paths.select, "groups.tsv")) + 1
                                    if os.path.exists(os.path.join(paths.select, "groups.tsv")) else 0))
    log("  taxid.tsv:   {}".format(count(os.path.join(paths.select, "taxid.tsv")) + 1
                                    if os.path.exists(os.path.join(paths.select, "taxid.tsv")) else 0))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

STAGES = {
    "fetch-metadata": stage_fetch_metadata,
    "build-index": stage_build_index,
    "apply-qc": stage_apply_qc,
    "select": stage_select,
    "plan-downloads": stage_plan_downloads,
    "download": stage_download,
    "verify": stage_verify,
    "export": stage_export,
    "split": stage_split,
    "organize": stage_organize,
    "fetch-organelle": stage_fetch_organelle,
    "status": stage_status,
}

RUN_ORDER = ["fetch-metadata", "build-index", "apply-qc",
             "select", "plan-downloads", "download", "verify",
             "export", "split", "organize"]


def main():
    ap = argparse.ArgumentParser(description="Tiara2 NCBI 数据统一管线（无 SQL）")
    ap.add_argument("stage", choices=list(STAGES.keys()) + ["run"],
                    help="要执行的阶段，或 run 一键跑全流程")
    ap.add_argument("--config", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config.json"))
    ap.add_argument("--stages", default="",
                    help="run 时指定子集，逗号分隔")
    ap.add_argument("--class", dest="only_class", default="",
                    help="download 阶段只处理某个类别")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = Paths(cfg)
    log("config: {}".format(args.config))
    log("primary_root: {}".format(paths.root))

    if args.stage == "run":
        order = args.stages.split(",") if args.stages else RUN_ORDER
        for st in order:
            st = st.strip()
            if st in STAGES:
                STAGES[st](cfg, paths)
            else:
                log("!! 未知阶段: {}".format(st))
    elif args.stage == "download":
        stage_download(cfg, paths, only_class=args.only_class or None)
    else:
        STAGES[args.stage](cfg, paths)


if __name__ == "__main__":
    main()
