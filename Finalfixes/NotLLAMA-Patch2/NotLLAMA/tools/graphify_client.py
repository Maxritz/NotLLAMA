"""Python GraphifyClient — mirrors include/rdna4_graphify.hpp.
Queries the local graphify knowledge graph via subprocess."""

import dataclasses
import json
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional


@dataclasses.dataclass
class GraphifyConfig:
    graphPath: str = "graphify-out/graph.json"
    autoUpdate: bool = True
    queryBeforeRead: bool = True
    preferredMode: str = "dfs"
    tokenBudget: int = 1500
    budgetCap: bool = True
    cacheSize: int = 32


@dataclasses.dataclass
class GraphQueryResult:
    question: str = ""
    answer: str = ""
    sourceNodes: list[str] = dataclasses.field(default_factory=list)
    sourceLocations: list[str] = dataclasses.field(default_factory=list)
    confidence: float = 0.0
    queryTimeMs: int = 0


class _LRUCache:
    def __init__(self, maxsize: int):
        self._maxsize = maxsize
        self._data: OrderedDict = OrderedDict()

    def get(self, key: str) -> Optional[GraphQueryResult]:
        if key not in self._data:
            return None
        self._data.move_to_end(key)
        return self._data[key]

    def put(self, key: str, value: GraphQueryResult):
        self._data[key] = value
        self._data.move_to_end(key)
        if len(self._data) > self._maxsize:
            self._data.popitem(last=False)

    def clear(self):
        self._data.clear()


class GraphifyClient:
    """Query the local graphify knowledge graph via CLI subprocess."""

    def __init__(self, config: GraphifyConfig):
        self._config = config
        self._cache = _LRUCache(config.cacheSize)

    @staticmethod
    def is_available() -> bool:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "graphify", "--help"],
                capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return False

    def query(self, question: str) -> GraphQueryResult:
        t0 = time.perf_counter()

        cache_key = (question, self._config.graphPath)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        cmd = [
            sys.executable, "-m", "graphify", "query", question,
            "--graph", self._config.graphPath,
            "--budget", str(self._config.tokenBudget)
        ]
        if self._config.preferredMode == "dfs":
            cmd.append("--dfs")

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            )
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            query_result = self._parse_output(question, result, elapsed_ms)
        except subprocess.TimeoutExpired:
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            query_result = GraphQueryResult(
                question=question,
                answer="(timeout)",
                confidence=0.0,
                queryTimeMs=elapsed_ms,
            )
        except Exception as e:
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            query_result = GraphQueryResult(
                question=question,
                answer=f"(error: {e})",
                confidence=0.0,
                queryTimeMs=elapsed_ms,
            )

        self._cache.put(cache_key, query_result)
        return query_result

    def _parse_output(self, question: str, result: subprocess.CompletedProcess,
                       elapsed_ms: int) -> GraphQueryResult:
        raw = result.stdout.strip()
        if not raw:
            raw = result.stderr.strip()
            if not raw:
                return GraphQueryResult(
                    question=question, answer="(empty)", confidence=0.0,
                    queryTimeMs=elapsed_ms)

        try:
            data = json.loads(raw)
            return GraphQueryResult(
                question=question,
                answer=data.get("answer", raw),
                sourceNodes=data.get("sourceNodes", []),
                sourceLocations=data.get("sourceLocations", []),
                confidence=data.get("confidence", 0.5),
                queryTimeMs=elapsed_ms,
            )
        except (json.JSONDecodeError, TypeError):
            return GraphQueryResult(
                question=question,
                answer=raw,
                confidence=0.5,
                queryTimeMs=elapsed_ms,
            )

    def is_stale(self) -> bool:
        graph = Path(self._config.graphPath)
        if not graph.is_file():
            return True
        graph_mtime = graph.stat().st_mtime
        src_dirs = [
            Path("src"),
            Path("include"),
            Path("tools"),
        ]
        for d in src_dirs:
            if d.is_dir():
                for f in d.rglob("*"):
                    if f.is_file() and f.suffix in (".cpp", ".hpp", ".comp", ".py"):
                        if f.stat().st_mtime > graph_mtime:
                            return True
        return False

    def update_graph(self) -> bool:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "graphify", "build",
                 "--out", self._config.graphPath],
                capture_output=True, text=True, timeout=120
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return False

    def get_related_nodes(self, symbol: str, depth: int = 2) -> list[str]:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "graphify", "query",
                 f"find {symbol} depth {depth}",
                 "--graph", self._config.graphPath,
                 "--budget", "500"],
                capture_output=True, text=True, timeout=15
            )
            data = json.loads(result.stdout)
            nodes = data.get("sourceNodes", [])
            if isinstance(nodes, list):
                return nodes
            return [result.stdout.strip()] if result.stdout.strip() else []
        except Exception:
            return []

    def clear_cache(self):
        self._cache.clear()


if __name__ == "__main__":
    cfg = GraphifyConfig()
    client = GraphifyClient(cfg)
    print(f"Graphify available: {client.is_available()}")
