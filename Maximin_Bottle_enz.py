from collections import defaultdict, deque
import networkx as nx
import numpy as np
from tqdm import tqdm
import time

def generate_one_rr_set(G, root):
    """
    生成以 root 为根的一个 RR 集（反向可达集）
    """
    rr_set = {root}
    queue = deque([root])
    visited = {root}

    while queue:
        current = queue.popleft()
        for neighbor in G.predecessors(current):
            if neighbor in visited:
                continue
            prob = G[neighbor][current]['prob']
            if np.random.random() < prob:
                rr_set.add(neighbor)
                queue.append(neighbor)
                visited.add(neighbor)

    return rr_set


def generate_reverse_reachable_sets(G, communities, node_community, theta=5, random_seed=42, show_progress=False):
    """
    为每个节点生成 RR 集并统计
    """
    np.random.seed(random_seed)
    nodes = list(G.nodes())

    num_communities = len(communities)
    node_cnt = {node: np.zeros(num_communities) for node in nodes}
    community_rr_sets = {comm_id: [] for comm_id in communities.keys()}
    node_rr_sets = {node: [] for node in nodes}

    it_nodes = tqdm(nodes, desc="生成RR集") if show_progress else nodes
    for v in it_nodes:
        v_community = node_community.get(v)
        for _ in range(theta):
            rr_set = generate_one_rr_set(G, v)
            community_rr_sets[v_community].append(rr_set)
            node_rr_sets[v].append(rr_set)
            for u in rr_set:
                node_cnt[u][v_community] += 1

    return community_rr_sets, node_rr_sets, node_cnt


def calculate_node_utility(nodes, node_cnt, community_rr_sets):
    """
    计算每个节点的效用向量
    """
    num_communities = len(community_rr_sets)
    community_rr_counts = {cid: len(rrs) for cid, rrs in community_rr_sets.items()}

    node_utility = {}
    for node in nodes:
        utility_vector = np.zeros(num_communities)
        for cid in range(num_communities):
            utility_vector[cid] = node_cnt[node][cid] / community_rr_counts[cid]
        node_utility[node] = utility_vector

    return node_utility


def _build_node_rr_indices(community_rr_sets):
    """
    构建倒排索引：node -> [(community_id, rr_index), ...]
    用于快速找到包含某节点的所有 RR 集，避免每轮全量扫描。
    """
    mapping = defaultdict(list)
    for cid, rrs in community_rr_sets.items():
        for idx, rr in enumerate(rrs):
            for u in rr:
                mapping[u].append((cid, idx))
    return mapping


def fairness_maximization(node_utility,
                          community_rr_sets,
                          node_rr_sets,
                          node_cnt,
                          community_to_attribute,
                          k=10,
                          verbose=0):
    """
    公平性影响力最大化（精简输出版）
    保持输出不变；内部优化：使用倒排索引与增量更新减少计算。
    """
    import numpy as np

    nodes = list(node_utility.keys())
    num_communities = len(community_rr_sets)
    community_rr_counts = {cid: len(rrs) for cid, rrs in community_rr_sets.items()}

    # 倒排索引：node -> [(cid, idx)]
    node_rr_indices = _build_node_rr_indices(community_rr_sets)

    S = []
    U = np.zeros(num_communities)

    first_candidate_no_change_count = 0
    rounds_compare = 0
    rounds_all_change = 0
    bottleneck_change_count = 0

    # 第 1 轮
    maximin_values = {n: float(np.min(node_utility[n])) for n in nodes}
    best_maximin = max(maximin_values.values())
    level1 = [n for n, v in maximin_values.items() if np.isclose(v, best_maximin)]
    if len(level1) > 1:
        sums = [(n, float(np.sum(node_utility[n]))) for n in level1]
        best_sum = max(s for _, s in sums)
        level2 = [n for n, s in sums if np.isclose(s, best_sum)]
        s1 = min(level2, key=lambda x: int(x)) if len(level2) > 1 else level2[0]
    else:
        s1 = level1[0]

    S.append(s1)
    U = node_utility[s1].copy()

    # 增量更新：仅更新受影响社区与节点
    affected_pairs = node_rr_indices.get(s1, [])
    impacted_by_cid = defaultdict(set)
    for cid, idx in affected_pairs:
        rr = community_rr_sets[cid][idx]
        for u in rr:
            if u != s1:
                node_cnt[u][cid] -= 1
                impacted_by_cid[cid].add(u)
    for cid, nodes_impacted in impacted_by_cid.items():
        denom = community_rr_counts[cid]
        for u in nodes_impacted:
            node_utility[u][cid] = node_cnt[u][cid] / denom

    # 后续轮次
    for _round in range(2, k + 1):
        bottleneck = int(np.argmin(U))
        remaining = [n for n in nodes if n not in S]
        ordered = sorted(remaining, key=lambda x: node_utility[x][bottleneck], reverse=True)

        scanned = []
        first_keep_found = False
        a_contrib = None

        for idx, n in enumerate(ordered):
            temp_U = U + node_utility[n]
            min_val = float(np.min(temp_U))
            min_indices = [j for j, val in enumerate(temp_U) if np.isclose(val, min_val)]
            sum_val = float(np.sum(temp_U))
            contrib = float(node_utility[n][bottleneck])
            scanned.append((n, temp_U, min_val, sum_val, min_indices, contrib))

            if not first_keep_found:
                if bottleneck in min_indices:
                    first_keep_found = True
                    a_contrib = contrib
                    if idx == 0:
                        first_candidate_no_change_count += 1
                    else:
                        rounds_compare += 1
            else:
                if not np.isclose(contrib, a_contrib):
                    break
        else:
            if not first_keep_found:
                rounds_compare += 1
                rounds_all_change += 1

        best_maximin = max(item[2] for item in scanned)
        level1 = [it for it in scanned if np.isclose(it[2], best_maximin)]
        if len(level1) > 1:
            best_sum = max(it[3] for it in level1)
            level2 = [it for it in level1 if np.isclose(it[3], best_sum)]
            selected = min(level2, key=lambda x: int(x[0]))[0] if len(level2) > 1 else level2[0][0]
        else:
            selected = level1[0][0]

        prev_bottleneck = bottleneck
        new_U = U + node_utility[selected]
        new_min_val = float(np.min(new_U))
        new_min_indices = [j for j, val in enumerate(new_U) if np.isclose(val, new_min_val)]
        if prev_bottleneck not in new_min_indices:
            bottleneck_change_count += 1

        U = new_U
        S.append(selected)

        # 增量更新：仅更新受影响社区与节点
        affected_pairs = node_rr_indices.get(selected, [])
        impacted_by_cid.clear()
        for cid, idx in affected_pairs:
            rr = community_rr_sets[cid][idx]
            for u in rr:
                if u != selected and u not in S:
                    node_cnt[u][cid] -= 1
                    impacted_by_cid[cid].add(u)
        for cid, nodes_impacted in impacted_by_cid.items():
            denom = community_rr_counts[cid]
            for u in nodes_impacted:
                node_utility[u][cid] = node_cnt[u][cid] / denom

    # 覆盖率统计（复用倒排索引）
    total_rr_sets = sum(len(rrs) for rrs in community_rr_sets.values())
    covered_rr_sets = set()
    for seed in S:
        for cid, idx in node_rr_indices.get(seed, []):
            covered_rr_sets.add((cid, idx))
    total_influence = len(covered_rr_sets) / total_rr_sets
    community_influence = {}
    for cid, rrs in community_rr_sets.items():
        covered_comm_rr_sets = sum(1 for (c, _) in covered_rr_sets if c == cid)
        community_influence[cid] = covered_comm_rr_sets / len(rrs)

    if verbose:
        print("公平性算法结束")
        print("种子数量:", len(S), "总影响力:", f"{total_influence:.6f}")

    return {
        "seed_set": S,
        "final_utility": U,
        "total_influence": total_influence,
        "community_influence": community_influence,
        "first_candidate_no_change_count": first_candidate_no_change_count,
        "rounds_compare": rounds_compare,
        "rounds_all_change": rounds_all_change,
        "bottleneck_change_count": bottleneck_change_count
    }


def maximin_greedy_maximization(node_utility,
                                community_rr_sets,
                                node_rr_sets,
                                node_cnt,
                                community_to_attribute,
                                k=10,
                                verbose=0):
    """
    全局 maximin 贪心影响力最大化（精简输出版）
    内部优化：使用倒排索引与增量更新；最终输出不变。
    """
    import numpy as np

    nodes = list(node_utility.keys())
    num_communities = len(community_rr_sets)
    community_rr_counts = {cid: len(rrs) for cid, rrs in community_rr_sets.items()}

    # 倒排索引：node -> [(cid, idx)]
    node_rr_indices = _build_node_rr_indices(community_rr_sets)

    S = []
    U = np.zeros(num_communities)

    for _round in range(1, k + 1):
        remaining_nodes = [v for v in nodes if v not in S]

        maximin_values = {}
        temp_U_cache = {}
        sum_values = {}
        for v in remaining_nodes:
            tU = U + node_utility[v]
            temp_U_cache[v] = tU
            maximin_values[v] = float(np.min(tU))
            sum_values[v] = float(np.sum(tU))

        best_maximin = max(maximin_values.values())
        level1 = [v for v, mv in maximin_values.items() if np.isclose(mv, best_maximin)]

        if len(level1) == 1:
            chosen = level1[0]
        else:
            best_sum = max(sum_values[v] for v in level1)
            level2 = [v for v in level1 if np.isclose(sum_values[v], best_sum)]
            chosen = level2[0] if len(level2) == 1 else min(level2, key=lambda x: int(x))

        U = temp_U_cache[chosen]
        S.append(chosen)

        # 增量更新：仅更新受影响社区与节点
        affected_pairs = node_rr_indices.get(chosen, [])
        impacted_by_cid = defaultdict(set)
        for cid, idx in affected_pairs:
            rr = community_rr_sets[cid][idx]
            for u in rr:
                if u != chosen and u not in S:
                    node_cnt[u][cid] -= 1
                    impacted_by_cid[cid].add(u)
        for cid, nodes_impacted in impacted_by_cid.items():
            denom = community_rr_counts[cid]
            for u in nodes_impacted:
                node_utility[u][cid] = node_cnt[u][cid] / denom

    # 覆盖率统计（复用倒排索引）
    total_rr_sets = sum(len(rrs) for rrs in community_rr_sets.values())
    covered_rr_sets = set()
    for seed in S:
        for cid, idx in node_rr_indices.get(seed, []):
            covered_rr_sets.add((cid, idx))
    total_influence = len(covered_rr_sets) / total_rr_sets
    community_influence = {}
    for cid, rrs in community_rr_sets.items():
        covered_comm_rr_sets = sum(1 for (c, _) in covered_rr_sets if c == cid)
        community_influence[cid] = covered_comm_rr_sets / len(rrs)

    if verbose:
        print("全局贪心算法结束")
        print("种子数量:", len(S), "总影响力:", f"{total_influence:.6f}")

    return {
        "seed_set": S,
        "final_utility": U,
        "total_influence": total_influence,
        "community_influence": community_influence
    }


def load_graph_from_txt(nodes_file, edges_file, prob_model="WIC"):
    """
    从文本文件构造与原流程兼容的数据结构：
      nodes_file: 行格式 => node_id community_id
      edges_file: 行格式 => u v   (有向边 u -> v)
    返回:
      G, communities(int索引), node_community(int), community_to_attribute(int->原标签)
    """
    import networkx as nx
    from collections import defaultdict

    G = nx.DiGraph()
    raw_node_comm = {}
    raw_comm_nodes = defaultdict(list)

    # 1. 读取节点
    with open(nodes_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            nid, cid_raw = parts[0], parts[1]
            raw_node_comm[nid] = cid_raw
            raw_comm_nodes[cid_raw].append(nid)
            G.add_node(nid, raw_comm=cid_raw)

    # 2. 读取边
    with open(edges_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = parts[0], parts[1]
            if u not in G:
                G.add_node(u, raw_comm=raw_node_comm.get(u))
            if v not in G:
                G.add_node(v, raw_comm=raw_node_comm.get(v))
            G.add_edge(u, v)

    # 3. 重新映射社区为 0..m-1
    sorted_raw_labels = sorted(raw_comm_nodes.keys())
    label_to_int = {lab: i for i, lab in enumerate(sorted_raw_labels)}
    community_to_attribute = {i: lab for lab, i in label_to_int.items()}  # int -> 原标签
    node_community = {nid: label_to_int[cid_raw] for nid, cid_raw in raw_node_comm.items()}

    # 4. 构建 communities (int id)
    communities = {i: raw_comm_nodes[lab] for lab, i in label_to_int.items()}

    # 5. 赋传播概率 (WIC: 1 / indeg(v))
    if prob_model == "WIC":
        in_degs = {v: G.in_degree(v) for v in G.nodes()}
        for u, v in G.edges():
            deg = in_degs[v]
            if deg > 0:
                G[u][v]['prob'] = 1.0 / deg
            else:
                G[u][v]['prob'] = 0.0
    else:
        raise ValueError("暂只支持 prob_model='WIC'")

    return G, communities, node_community, community_to_attribute


def main_from_txt(nodes_file, edges_file, theta, k, verbose=1):
    """
    使用 txt 节点/边文件直接运行公平性与全局 maximin 贪心两种算法。
    """
    import time
    import numpy as np

    start_time = time.time()

    # A. 构建图与社区
    G, communities, node_community, community_to_attribute = load_graph_from_txt(nodes_file, edges_file)

    # B. 生成 RR 集（关闭进度条，避免冗余输出；需要时可在 generate_* 中开启 show_progress）
    community_rr_sets, node_rr_sets, node_cnt = generate_reverse_reachable_sets(
        G, communities, node_community, theta=theta
    )

    # C. 初始效用
    nodes = list(G.nodes())
    node_utility = calculate_node_utility(nodes, node_cnt, community_rr_sets)

    if k <= 0:
        m = len(communities)
        fairness_result = {
            "seed_set": [],
            "final_utility": np.zeros(m),
            "total_influence": 0.0,
            "community_influence": {cid: 0.0 for cid in communities.keys()},
            "first_candidate_no_change_count": 0,
            "rounds_compare": 0,
            "rounds_all_change": 0,
            "bottleneck_change_count": 0
        }
        greedy_result = {
            "seed_set": [],
            "final_utility": np.zeros(m),
            "total_influence": 0.0,
            "community_influence": {cid: 0.0 for cid in communities.keys()}
        }
        return {
            "fairness": fairness_result,
            "greedy": greedy_result,
            "overlap_ratio": None,
            "common_seeds": []
        }

    # D. 公平性算法
    t1 = time.time()
    fairness_result = fairness_maximization(
        node_utility.copy(),
        community_rr_sets,
        node_rr_sets,
        {nid: vec.copy() for nid, vec in node_cnt.items()},
        community_to_attribute,
        k=k,
        verbose=0
    )
    fairness_time = time.time() - t1

    # E. 全局贪心
    t2 = time.time()
    greedy_result = maximin_greedy_maximization(
        node_utility.copy(),
        community_rr_sets,
        node_rr_sets,
        {nid: vec.copy() for nid, vec in node_cnt.items()},
        community_to_attribute,
        k=k,
        verbose=0
    )
    greedy_time = time.time() - t2

    # F. 对比（输出保持原样）
    common = set(fairness_result["seed_set"]).intersection(greedy_result["seed_set"])
    overlap_ratio = len(common) / k if k > 0 else None

    if verbose:
        total_time = time.time() - start_time
        print("\n=== TXT 数据集结果对比 ===")
        print(f"总运行时间: {total_time:.2f}s | θ={theta} | k={k}")
        print("公平性: time={:.2f}s seeds={} influence={:.6f}".format(
            fairness_time, fairness_result["seed_set"], fairness_result["total_influence"]))
        print("贪心  : time={:.2f}s seeds={} influence={:.6f}".format(
            greedy_time, greedy_result["seed_set"], greedy_result["total_influence"]))
        print(f"\n重叠比例: {overlap_ratio:.2f} ({len(common)}/{k})")
        print("共同种子:", list(common))
        print("\n社区影响力对比 (fair - greedy):")
        for cid in sorted(fairness_result["community_influence"].keys()):
            fv = fairness_result["community_influence"][cid]
            gv = greedy_result["community_influence"].get(cid, 0.0)
            cname = community_to_attribute[cid]
            print(f"  {cid}({cname}): {fv:.4f} vs {gv:.4f} Δ={fv-gv:.4f}")
        print("\n公平性统计:",
              f"first_keep={fairness_result['first_candidate_no_change_count']}",
              f"compare={fairness_result['rounds_compare']}",
              f"all_change={fairness_result['rounds_all_change']}",
              f"bottleneck_change={fairness_result['bottleneck_change_count']}")

    return {
        "fairness": fairness_result,
        "greedy": greedy_result,
        "overlap_ratio": overlap_ratio,
        "common_seeds": list(common),
        "community_to_attribute": community_to_attribute
    }


# 使用示例（请替换路径）
if __name__ == "__main__":
    nodes_path = r"C:\Users\31062\Desktop\科研积累\code\enz_nodes.txt"
    edges_path = r"C:\Users\31062\Desktop\科研积累\code\enz_edges.txt"
    for i in [10, 20, 30, 40, 50]:
        print(f"\n--- 运行 k={i} ---")
        result_txt = main_from_txt(nodes_path, edges_path, theta=5, k=i, verbose=1)