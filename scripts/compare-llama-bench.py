#!/usr/bin/env python3

"""
Creates tables from llama-bench data written to an SQLite database.
Compares performance between different commits.
"""

import logging
import argparse
import heapq
import sys
import os
from glob import glob
import sqlite3
from typing import List, Dict, Tuple, Optional, Set
from contextlib import contextmanager
from functools import lru_cache

try:
    import git
    from tabulate import tabulate
except ImportError as e:
    print("Required Python libraries: GitPython, tabulate.")
    raise e

logger = logging.getLogger("compare-llama-bench")

# Configuration constants
KEY_PROPERTIES: List[str] = [
    "cpu_info", "gpu_info", "n_gpu_layers", "cuda", "vulkan", "kompute",
    "metal", "sycl", "rpc", "gpu_blas", "blas", "model_filename",
    "model_type", "n_batch", "n_ubatch", "embeddings", "n_threads",
    "type_k", "type_v", "use_mmap", "no_kv_offload", "split_mode",
    "main_gpu", "tensor_split", "flash_attn", "n_prompt", "n_gen"
]

BOOL_PROPERTIES: List[str] = [
    "cuda", "vulkan", "kompute", "metal", "sycl", "gpu_blas", "blas",
    "embeddings", "use_mmap", "no_kv_offload", "flash_attn"
]

PRETTY_NAMES: Dict[str, str] = {
    "cuda": "CUDA", "vulkan": "Vulkan", "kompute": "Kompute",
    "metal": "Metal", "sycl": "SYCL", "rpc": "RPC",
    "gpu_blas": "GPU BLAS", "blas": "BLAS", "cpu_info": "CPU",
    "gpu_info": "GPU", "model_filename": "File", "model_type": "Model",
    "model_size": "Model Size [GiB]", "model_n_params": "Num. of Par.",
    "n_batch": "Batch size", "n_ubatch": "Microbatch size",
    "n_threads": "Threads", "type_k": "K type", "type_v": "V type",
    "n_gpu_layers": "GPU layers", "split_mode": "Split mode",
    "main_gpu": "Main GPU", "no_kv_offload": "NKVO",
    "flash_attn": "FlashAttention", "tensor_split": "Tensor split",
    "use_mmap": "Use mmap", "embeddings": "Embeddings"
}

DEFAULT_SHOW = ["model_type"]
DEFAULT_HIDE = ["model_filename"]
GPU_NAME_STRIP = ["NVIDIA GeForce ", "Tesla ", "AMD Radeon "]
MODEL_SUFFIX_REPLACE = {" - Small": "_S", " - Medium": "_M", " - Large": "_L"}

@contextmanager
def database_connection(db_path: str):
    """Safe database connection context manager."""
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        yield conn
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        raise
    finally:
        if conn:
            conn.close()

def get_git_repo(path: str = ".") -> Optional[git.Repo]:
    """Safely get git repository."""
    try:
        return git.Repo(path, search_parent_directories=True)
    except git.InvalidGitRepositoryError:
        return None
    except Exception as e:
        logger.error(f"Git error: {e}")
        return None

@lru_cache(maxsize=128)
def get_commit_name(repo: Optional[git.Repo], hexsha8: str) -> str:
    """Get human-readable name for a commit with caching."""
    if not repo or not hexsha8:
        return hexsha8
    try:
        # Check branches
        for head in repo.heads:
            if head.commit.hexsha[:8] == hexsha8:
                return head.name
        # Check tags
        for tag in repo.tags:
            if tag.commit.hexsha[:8] == hexsha8:
                return tag.name
        return hexsha8
    except Exception as e:
        logger.error(f"Error getting commit name: {e}")
        return hexsha8

def get_commit_hexsha8(repo: Optional[git.Repo], name: str) -> Optional[str]:
    """Search for a commit given a human-readable name."""
    if not repo or not name:
        return None
    try:
        # Check branches
        for head in repo.heads:
            if head.name == name:
                return head.commit.hexsha[:8]
        # Check tags
        for tag in repo.tags:
            if tag.name == name:
                return tag.commit.hexsha[:8]
        # Check direct commit hash
        try:
            commit = repo.commit(name)
            return commit.hexsha[:8]
        except (git.BadName, ValueError):
            return None
    except Exception as e:
        logger.error(f"Error resolving commit: {e}")
        return None

def find_parent_in_data(commit: git.Commit, available_commits: List[Tuple[str]]) -> Optional[str]:
    """Find the most recent parent with data."""
    if not commit:
        return None
        
    heap: List[Tuple[int, git.Commit]] = [(0, commit)]
    seen_hexsha8 = set()
    
    try:
        while heap:
            depth, current_commit = heapq.heappop(heap)
            current_hexsha8 = current_commit.hexsha[:8]
            
            if (current_hexsha8,) in available_commits:
                return current_hexsha8
                
            for parent in current_commit.parents:
                parent_hexsha8 = parent.hexsha[:8]
                if parent_hexsha8 not in seen_hexsha8:
                    seen_hexsha8.add(parent_hexsha8)
                    heapq.heappush(heap, (depth + 1, parent))
        return None
    except Exception as e:
        logger.error(f"Error finding parent commit: {e}")
        return None

def get_all_parent_hexsha8s(commit: git.Commit) -> Set[str]:
    """Get all parent commit hashes recursively."""
    if not commit:
        return set()
    
    visited = set()
    to_visit = [commit]
    
    try:
        while to_visit:
            current = to_visit.pop(0)
            current_hash = current.hexsha[:8]
            if current_hash not in visited:
                visited.add(current_hash)
                to_visit.extend(current.parents)
        return visited
    except Exception as e:
        logger.error(f"Error getting parent commits: {e}")
        return set()

def get_rows(
    cursor: sqlite3.Cursor,
    properties: List[str],
    baseline_hash: str,
    compare_hash: str
) -> List[Tuple]:
    """
    Get benchmark comparison rows with safe query construction.
    
    Args:
        cursor: Database cursor
        properties: Properties to compare
        baseline_hash: Baseline commit hash
        compare_hash: Comparison commit hash
        
    Returns:
        List of comparison result tuples
    """
    try:
        # Validate inputs
        if not all(p in KEY_PROPERTIES for p in properties):
            invalid = [p for p in properties if p not in KEY_PROPERTIES]
            raise ValueError(f"Invalid properties: {', '.join(invalid)}")

        select_cols = [f"tb.{p}" for p in properties]
        select_cols.extend(["tb.n_prompt", "tb.n_gen", "AVG(tb.avg_ts)", "AVG(tc.avg_ts)"])
        select_string = ", ".join(select_cols)

        join_conditions = [f"tb.{p} = tc.{p}" for p in KEY_PROPERTIES]
        join_conditions.extend(["tb.build_commit = ?", "tc.build_commit = ?"])
        join_string = " AND ".join(join_conditions)

        group_cols = [f"tb.{p}" for p in properties]
        group_cols.extend(["tb.n_gen", "tb.n_prompt"])
        group_string = ", ".join(group_cols)

        query = f"""
            SELECT {select_string} 
            FROM test tb 
            JOIN test tc ON {join_string}
            GROUP BY {group_string}
            ORDER BY {group_string};
        """

        return cursor.execute(query, (baseline_hash, compare_hash)).fetchall()
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        raise

def process_comparison_data(
    cursor: sqlite3.Cursor,
    show: List[str],
    baseline_hash: str,
    compare_hash: str,
    name_baseline: str,
    name_compare: str
) -> Tuple[List[List], List[str]]:
    """Process comparison data and prepare table output."""
    rows = get_rows(cursor, show, baseline_hash, compare_hash)
    if not rows:
        raise ValueError("No comparison data found")

    table = []
    for row in rows:
        n_prompt = int(row[-4])
        n_gen = int(row[-3])
        
        # Format test name
        if n_prompt != 0 and n_gen == 0:
            test_name = f"pp{n_prompt}"
        elif n_prompt == 0 and n_gen != 0:
            test_name = f"tg{n_gen}"
        else:
            test_name = f"pp{n_prompt}+tg{n_gen}"

        # Convert row data
        formatted_row = list(row[:-4])  # Regular columns
        
        # Add test name and metrics
        formatted_row.extend([
            test_name,
            float(row[-2]),  # baseline t/s
            float(row[-1]),  # compare t/s
            float(row[-1]) / float(row[-2])  # speedup
        ])
        
        table.append(formatted_row)

    # Post-process table data
    for bool_property in BOOL_PROPERTIES:
        if bool_property in show:
            ip = show.index(bool_property)
            for row_table in table:
                row_table[ip] = "Yes" if int(row_table[ip]) == 1 else "No"

    if "model_type" in show:
        ip = show.index("model_type")
        for old, new in MODEL_SUFFIX_REPLACE.items():
            for row_table in table:
                row_table[ip] = row_table[ip].replace(old, new)

    if "model_size" in show:
        ip = show.index("model_size")
        for row_table in table:
            row_table[ip] = float(row_table[ip]) / 1024 ** 3

    if "gpu_info" in show:
        ip = show.index("gpu_info")
        for row_table in table:
            for prefix in GPU_NAME_STRIP:
                row_table[ip] = row_table[ip].replace(prefix, "")

            gpu_names = row_table[ip].split("/")
            if len(gpu_names) >= 2 and len(set(gpu_names)) == 1:
                row_table[ip] = f"{len(gpu_names)}x {gpu_names[0]}"

    # Prepare headers
    headers = [PRETTY_NAMES[p] for p in show]
    headers.extend(["Test", f"t/s {name_baseline}", f"t/s {name_compare}", "Speedup"])

    return table, headers

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-b", "--baseline", help="Baseline commit (branch, tag, or hash)")
    parser.add_argument("-c", "--compare", help="Comparison commit (branch, tag, or hash)")
    parser.add_argument("-i", "--input", help="Input SQLite file")
    parser.add_argument("-o", "--output", default="pipe", help="Output format")
    parser.add_argument("-s", "--show", help="Columns to show (comma-separated)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(levelname)s: %(message)s'
    )

    # Find input file
    input_file = args.input
    if not input_file:
        if os.path.exists("./llama-bench.sqlite"):
            input_file = "llama-bench.sqlite"
        else:
            sqlite_files = glob("*.sqlite")
            if len(sqlite_files) == 1:
                input_file = sqlite_files[0]

    if not input_file:
        logger.error("No suitable input file found")
        sys.exit(1)

    try:
        repo = get_git_repo()
        
        with database_connection(input_file) as conn:
            cursor = conn.cursor()
            available_commits = cursor.execute(
                "SELECT DISTINCT build_commit FROM test;"
            ).fetchall()

            # Resolve baseline commit
            hexsha8_baseline = None
            if args.baseline:
                if (args.baseline,) in available_commits:
                    hexsha8_baseline = args.baseline
                if not hexsha8_baseline and repo:
                    hexsha8_baseline = get_commit_hexsha8(repo, args.baseline)
            elif repo:
                hexsha8_baseline = find_parent_in_data(
                    repo.heads.master.commit,
                    available_commits
                )

            if not hexsha8_baseline:
                logger.error("Could not resolve baseline commit")
                sys.exit(1)

            # Resolve compare commit
            hexsha8_compare = None
            if args.compare:
                if (args.compare,) in available_commits:
                    hexsha8_compare = args.compare
                if not hexsha8_compare and repo:
                    hexsha8_compare = get_commit_hexsha8(repo, args.compare)
            elif repo:
                master_parents = get_all_parent_hexsha8s(repo.heads.master.commit)
                builds_timestamp = cursor.execute(
                    "SELECT build_commit, test_time FROM test ORDER BY test_time DESC;"
                ).fetchall()
                
                for hexsha8, _ in builds_timestamp:
                    if hexsha8 not in master_parents:
                        hexsha8_compare = hexsha8
                        break

            if not hexsha8_compare:
                logger.error("Could not resolve comparison commit")
                sys.exit(1)

            # Get human-readable names
            name_baseline = get_commit_name(repo, hexsha8_baseline)
            name_compare = get_commit_name(repo, hexsha8_compare)

            # Determine columns to show
            if args.show:
                show = args.show.split(",")
                unknown_cols = [p for p in show if p not in KEY_PROPERTIES[:-2]]
                if unknown_cols:
                    logger.error(f"Unknown columns: {', '.join(unknown_cols)}")
                    sys.exit(1)
            else:
                show = determine_columns_to_show(cursor, hexsha8_baseline, hexsha8_compare)

            # Process data and create table
            table, headers = process_comparison_data(
                cursor,
                show,
                hexsha8_baseline,
                hexsha8_compare,
                name_baseline,
                name_compare
            )

            # Output results
            print(tabulate(
                table,
                headers=headers,
                floatfmt=".2f",
                tablefmt=args.output
            ))

    except Exception as e:
        logger.error(f"Error during execution: {e}")
        if args.verbose:
            logger.exception("Detailed error information:")
        sys.exit(1)

if __name__ == "__main__":
    main()
