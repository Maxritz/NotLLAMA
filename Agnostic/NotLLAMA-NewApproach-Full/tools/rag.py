#!/usr/bin/env python3
"""
RAG — Retrieval-Augmented Generation for RDNA4-native inference engine.

Uses graphify knowledge graph as primary RAG source, augmented with
web/GitHub search for external references.

Usage:
    python tools/rag.py query "How does DPP work on RDNA4?"
    python tools/rag.py graph "VBUFFER"
    python tools/rag.py code "attention"
    python tools/rag.py github "RDNA4 wave32 persistent kernel"
    python tools/rag.py web "Vulkan compute shader RDNA4"
    python tools/rag.py context "softmax DPP reduction"
"""

import sys
import json
import os
from pathlib import Path
from collections import defaultdict

# === Graph Layer ===

class GraphRAG:
    """Knowledge graph RAG backend using graphify output."""

    def __init__(self, graph_path=None):
        if graph_path is None:
            # Default to graphify-out/graph.json relative to project root
            root = Path(__file__).parent.parent
            graph_path = root / "graphify-out" / "graph.json"

        self.graph_path = Path(graph_path)
        self.G = None
        self._load()

    def _load(self):
        import networkx as nx
        from networkx.readwrite import json_graph

        if not self.graph_path.exists():
            print(f"Warning: graph not found at {self.graph_path}", file=sys.stderr)
            self.G = nx.Graph()
            return

        data = json.loads(self.graph_path.read_text(encoding="utf-8"))
        self.G = json_graph.node_link_graph(data, edges="links")

        # Build reverse index: label -> node_id
        self.label_index = {}
        self.type_index = defaultdict(list)
        self.community_index = defaultdict(list)
        self.source_index = defaultdict(list)

        for nid, attrs in self.G.nodes(data=True):
            label = attrs.get("label", nid).lower()
            self.label_index[label] = nid
            self.label_index[nid] = nid

            ft = attrs.get("file_type", "unknown")
            self.type_index[ft].append(nid)

            comm = attrs.get("community", -1)
            self.community_index[comm].append(nid)

            src = attrs.get("source_file", "")
            if src:
                self.source_index[src].append(nid)

    def search(self, query, k=10):
        """Find nodes matching query string (fuzzy label match)."""
        query_lower = query.lower()
        results = []

        for nid, attrs in self.G.nodes(data=True):
            label = attrs.get("label", nid).lower()
            # Score: exact match > partial match > word overlap
            score = 0
            if query_lower == label:
                score = 100
            elif query_lower in label:
                score = 80
            elif label in query_lower:
                score = 60
            else:
                # Word overlap
                query_words = set(query_lower.split())
                label_words = set(label.split())
                overlap = len(query_words & label_words)
                if overlap > 0:
                    score = 40 * overlap / max(len(query_words), 1)

            if score > 0:
                results.append((score, nid, attrs))

        results.sort(reverse=True)
        return results[:k]

    def neighbors(self, node_id, depth=1):
        """Get neighboring nodes up to depth hops."""
        if node_id not in self.G:
            return []

        visited = {node_id}
        current = {node_id}
        result = []

        for d in range(depth):
            next_level = set()
            for n in current:
                for neighbor in self.G.neighbors(n):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_level.add(neighbor)
                        attrs = self.G.nodes[neighbor]
                        result.append((d + 1, neighbor, attrs))
            current = next_level

        return result

    def community_nodes(self, community_id):
        """Get all nodes in a community."""
        return [(nid, self.G.nodes[nid]) for nid in self.community_index.get(community_id, [])]

    def path_between(self, source, target, max_depth=5):
        """Find shortest path between two nodes."""
        import networkx as nx
        if source not in self.G or target not in self.G:
            return []
        try:
            path = nx.shortest_path(self.G, source, target)
            return [(nid, self.G.nodes[nid]) for nid in path]
        except nx.NetworkXNoPath:
            return []

    def subgraph_context(self, query, depth=2, max_nodes=50):
        """Get a contextual subgraph around matching nodes."""
        matches = self.search(query, k=5)
        if not matches:
            return {}

        context_nodes = set()
        for _, nid, _ in matches:
            context_nodes.add(nid)
            neighbors = self.neighbors(nid, depth=depth)
            for _, nnid, _ in neighbors[:max_nodes // len(matches)]:
                context_nodes.add(nnid)

        # Build context
        subgraph_data = {
            "query": query,
            "matches": [(nid, attrs.get("label", nid), attrs.get("file_type", "?"))
                       for _, nid, attrs in matches],
            "context_nodes": [],
        }

        for nid in context_nodes:
            attrs = self.G.nodes[nid]
            edges = []
            for u, v, edata in self.G.edges(nid, data=True):
                neighbor = v if u == nid else u
                neighbor_label = self.G.nodes[neighbor].get("label", neighbor)
                edges.append({
                    "to": neighbor_label,
                    "relation": edata.get("relation", "?"),
                    "confidence": edata.get("confidence", "?"),
                })
            subgraph_data["context_nodes"].append({
                "id": nid,
                "label": attrs.get("label", nid),
                "type": attrs.get("file_type", "?"),
                "community": attrs.get("community", -1),
                "edges": edges[:10],  # Top 10 edges
            })

        return subgraph_data

    def god_nodes(self, top=10):
        """Return most connected nodes."""
        degrees = sorted(self.G.degree(), key=lambda x: x[1], reverse=True)
        return [(nid, deg, self.G.nodes[nid].get("label", nid))
                for nid, deg in degrees[:top]]

    def community_summary(self):
        """Summarize all communities."""
        summaries = []
        for comm_id in sorted(self.community_index.keys()):
            nodes = self.community_index[comm_id]
            labels = [self.G.nodes[n].get("label", n) for n in nodes[:5]]
            summaries.append({
                "id": comm_id,
                "size": len(nodes),
                "samples": labels,
            })
        return sorted(summaries, key=lambda x: x["size"], reverse=True)

    def code_context(self, keyword, max_files=5):
        """Find code files and their context."""
        results = []
        for nid in self.type_index.get("code", []):
            attrs = self.G.nodes[nid]
            label = attrs.get("label", "").lower()
            if keyword.lower() in label:
                src = attrs.get("source_file", "")
                # Get neighbors for context
                neighbors = self.neighbors(nid, depth=1)
                neighbor_labels = [self.G.nodes[n].get("label", n) for _, n, _ in neighbors[:5]]
                results.append({
                    "node": attrs.get("label", nid),
                    "source": src,
                    "neighbors": neighbor_labels,
                })
                if len(results) >= max_files:
                    break
        return results


# === Web/GitHub Search Layer ===

def search_github(query, max_results=5):
    """Search GitHub for code/repositories matching query."""
    import subprocess
    import shutil

    if not shutil.which("gh"):
        return [{"error": "gh CLI not installed. Install from https://cli.github.com/"}]

    try:
        result = subprocess.run(
            ["gh", "search", "repos", query, "--limit", str(max_results), "--json",
             "name,description,url,stargazersCount,language"],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode != 0:
            return [{"error": result.stderr}]

        repos = json.loads(result.stdout)
        return [{
            "name": r.get("name", ""),
            "description": r.get("description", ""),
            "url": r.get("url", ""),
            "stars": r.get("stargazersCount", 0),
            "language": r.get("language", ""),
        } for r in repos]
    except Exception as e:
        return [{"error": str(e)}]


def search_github_code(query, max_results=5):
    """Search GitHub code matching query."""
    import subprocess
    import shutil

    if not shutil.which("gh"):
        return [{"error": "gh CLI not installed"}]

    try:
        result = subprocess.run(
            ["gh", "search", "code", query, "--limit", str(max_results), "--json",
             "repository,path,textMatches"],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode != 0:
            return [{"error": result.stderr}]

        return json.loads(result.stdout)
    except Exception as e:
        return [{"error": str(e)}]


def search_web(query, max_results=5):
    """Search web for documentation and references."""
    import subprocess
    import shutil

    # Try curl with DuckDuckGo lite (no API key needed)
    try:
        import urllib.parse
        encoded = urllib.parse.quote(query)
        url = f"https://lite.duckduckgo.com/lite/?q={encoded}"
        result = subprocess.run(
            ["curl", "-sL", "--max-time", "10", url],
            capture_output=True, text=True, timeout=15
        )
        # Parse simple results
        lines = result.stdout.split("\n")
        results = []
        for line in lines:
            line = line.strip()
            if line.startswith("http") and len(line) > 10:
                results.append({"url": line})
                if len(results) >= max_results:
                    break
        return results
    except Exception as e:
        return [{"error": str(e)}]


# === Unified RAG ===

class RAG:
    """Unified RAG: graph + external sources."""

    def __init__(self, graph_path=None):
        self.graph = GraphRAG(graph_path)

    def query(self, question, max_context=20):
        """Full RAG query: graph context + external references."""
        results = {
            "question": question,
            "graph_context": self.graph.subgraph_context(question, depth=2, max_nodes=max_context),
            "god_nodes": self.graph.god_nodes(top=5),
            "external": {},
        }

        # Search GitHub for relevant code
        gh_results = search_github(question + " RDNA4 vulkan compute", max_results=3)
        if gh_results and "error" not in gh_results[0]:
            results["external"]["github_repos"] = gh_results

        gh_code = search_github_code(question + " RDNA4 DPP wave32", max_results=3)
        if gh_code and "error" not in gh_code[0]:
            results["external"]["github_code"] = gh_code

        return results

    def graph_query(self, query, k=10):
        """Pure graph search."""
        return self.graph.search(query, k=k)

    def get_context(self, concept):
        """Get rich context around a concept."""
        return self.graph.subgraph_context(concept, depth=2, max_nodes=30)

    def find_path(self, concept_a, concept_b):
        """Find path between two concepts."""
        matches_a = self.graph.search(concept_a, k=1)
        matches_b = self.graph.search(concept_b, k=1)
        if matches_a and matches_b:
            nid_a = matches_a[0][1]
            nid_b = matches_b[0][1]
            return self.graph.path_between(nid_a, nid_b)
        return []

    def code_context(self, keyword):
        """Find code context for a keyword."""
        return self.graph.code_context(keyword)

    def communities(self):
        """List all communities."""
        return self.graph.community_summary()

    def help(self):
        """Print usage."""
        print("""
RAG — RDNA4 Knowledge Graph + External Search

Commands:
  query <text>        Full RAG query (graph + GitHub + web)
  graph <text>        Search graph nodes
  code <text>         Search code files
  path <A> <B>        Find path between concepts
  context <concept>   Rich context around a concept
  god                 Most connected nodes
  communities         List all communities
  github <query>      Search GitHub repos
  github-code <query> Search GitHub code
  web <query>         Search web

Examples:
  python tools/rag.py query "DPP register shuffle RDNA4"
  python tools/rag.py graph "VBUFFER"
  python tools/rag.py path "softmax" "DPP"
  python tools/rag.py code "attention"
  python tools/rag.py context "persistent kernel"
""")


# === CLI ===

def main():
    if len(sys.argv) < 2:
        RAG().help()
        return

    cmd = sys.argv[1].lower()
    args = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else ""

    rag = RAG()

    if cmd == "query":
        if not args:
            print("Usage: rag query <question>")
            return
        result = rag.query(args)
        print(json.dumps(result, indent=2, default=str))

    elif cmd == "graph":
        if not args:
            print("Usage: rag graph <search term>")
            return
        results = rag.graph_query(args)
        for score, nid, attrs in results:
            print(f"  [{score:.0f}] {attrs.get('label', nid)} ({attrs.get('file_type', '?')})")

    elif cmd == "code":
        if not args:
            print("Usage: rag code <keyword>")
            return
        results = rag.code_context(args)
        for r in results:
            print(f"  {r['node']}")
            print(f"    source: {r['source']}")
            if r['neighbors']:
                print(f"    uses: {', '.join(r['neighbors'][:5])}")

    elif cmd == "path":
        parts = args.split(" ", 1)
        if len(parts) < 2:
            print("Usage: rag path <concept_a> <concept_b>")
            return
        path = rag.find_path(parts[0], parts[1])
        if path:
            for i, (nid, attrs) in enumerate(path):
                arrow = "  -> " if i > 0 else "     "
                print(f"{arrow}{attrs.get('label', nid)}")
        else:
            print("No path found")

    elif cmd == "context":
        if not args:
            print("Usage: rag context <concept>")
            return
        ctx = rag.get_context(args)
        if ctx.get("matches"):
            print(f"Matches for '{args}':")
            for nid, label, ftype in ctx["matches"]:
                print(f"  {label} ({ftype})")
        if ctx.get("context_nodes"):
            print(f"\nContext ({len(ctx['context_nodes'])} nodes):")
            for n in ctx["context_nodes"][:15]:
                edge_str = ""
                if n["edges"]:
                    edge_str = f" -> {n['edges'][0]['to']} ({n['edges'][0]['relation']})"
                print(f"  {n['label']} [{n['type']}]{edge_str}")

    elif cmd == "god":
        for nid, deg, label in rag.graph.god_nodes(top=15):
            print(f"  {label} (degree: {deg})")

    elif cmd == "communities":
        for c in rag.communities():
            print(f"  Community {c['id']} ({c['size']} nodes): {', '.join(c['samples'][:5])}")

    elif cmd == "github":
        if not args:
            print("Usage: rag github <query>")
            return
        results = search_github(args)
        for r in results:
            if "error" in r:
                print(f"  Error: {r['error']}")
            else:
                print(f"  {r['name']} ({r.get('stars', 0)}*")
                if r.get('description'):
                    print(f"    {r['description'][:80]}")
                print(f"    {r['url']}")

    elif cmd == "github-code":
        if not args:
            print("Usage: rag github-code <query>")
            return
        results = search_github_code(args)
        for r in results:
            if "error" in r:
                print(f"  Error: {r['error']}")
            else:
                repo = r.get("repository", {}).get("name", "?")
                path = r.get("path", "?")
                print(f"  {repo}/{path}")

    elif cmd == "web":
        if not args:
            print("Usage: rag web <query>")
            return
        results = search_web(args)
        for r in results:
            if "error" in r:
                print(f"  Error: {r['error']}")
            else:
                print(f"  {r.get('url', '?')}")

    else:
        print(f"Unknown command: {cmd}")
        RAG().help()


if __name__ == "__main__":
    main()
