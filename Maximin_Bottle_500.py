from collections import defaultdict, deque
import networkx as nx
import numpy as np
import time
import math


def preprocess_network_by_attribute(G, attribute_name, prob_model="WIC"):
    """
    为网络的每条边添加传播概率，并按指定属性划分社区
    """
    # 确保为有向图
    if not isinstance(G, nx.DiGraph):
        G = nx.DiGraph(G)

    G_proc = G.copy()

    # 计算每个节点的入度
    in_degrees = {node: G_proc.in_degree(node) for node in G_proc.nodes()}

    # 为每条边添加传播概率（WIC：1 / in_degree(v)）
    for u, v in G_proc.edges():
        v_in_degree = in_degrees[v]
        G_proc[u][v]['prob'] = 1.0 / v_in_degree

    # 属性映射
    attribute_mappings = {
        'gender': {'male': 0, 'female': 1},
        'status': {'normal': 0, 'obese': 1, 'overweight': 2},
        'ethnicity': {'asian': 0, 'black': 1, 'latino': 2, 'other': 3, 'white': 4},
        'age': {
            '18-24': 0, '25-29': 1, '30-39': 2, '40-49': 3,
            '50-59': 4, '60-64': 5, '65+': 6
        },
        'region': {
            'acton': 0, 'desert_view_highlands': 1, 'lake_los_angeles': 2,
            'lancaster': 3, 'leona_valley': 4, 'littlerock': 5,
            'northeast_antelope_valley': 6, 'northwest_antelope_valley': 7,
            'northwest_palmdale': 8, 'palmdale': 9, 'quartz_hill': 10,
            'southeast_antelope_valley': 11, 'sun_village': 12
        }
    }

    if attribute_name not in attribute_mappings:
        raise ValueError(f"未知的属性名: {attribute_name}，可选: {list(attribute_mappings.keys())}")

    attribute_to_community = attribute_mappings[attribute_name]
    community_to_attribute = {v: k for k, v in attribute_to_community.items()}

    # 划分社区
    communities = defaultdict(list)
    node_community = {}
    for node, attrs in G_proc.nodes(data=True):
        if attribute_name not in attrs:
            raise ValueError(f"节点 {node} 缺少属性 {attribute_name}")
        attribute_value = attrs[attribute_name]
        if attribute_value not in attribute_to_community:
            raise ValueError(f"属性值 {attribute_value} 不在映射中，请补充映射或修正数据")
        community_id = attribute_to_community[attribute_value]
        communities[community_id].append(node)
        node_community[node] = community_id

    return G_proc, communities, node_community, community_to_attribute


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


def generate_reverse_reachable_sets(G, communities, node_community, theta=5, random_seed=42):
    """
    生成反向可达集(RR集)并统计相关信息。

    输入:
      - G: 带 'prob' 概率的有向图
      - communities: {cid: [nodes,...]} 社区划分
      - node_community: {node: cid} 节点所属社区
      - theta:
          * int: 旧逻辑（保持不变）——每个节点生成 theta 个 RR 集
          * dict[int->int]: 新逻辑（分社区）——每个社区生成 theta[cid] 个 RR 集（根仅从该社区采样）
      - random_seed: 随机种子

    输出:
      - community_rr_sets: {cid: [RR_set, ...]}
      - node_rr_sets: {node: [RR_set, ...]}
      - node_cnt: {node: np.array(m)}，节点被各社区 RR 集覆盖的次数向量
    """
    np.random.seed(random_seed)

    # 节点列表与社区数
    nodes = list(G.nodes())
    num_communities = len(communities)

    # 初始化统计结构
    community_rr_sets = {comm_id: [] for comm_id in communities.keys()}
    node_rr_sets = {node: [] for node in nodes}
    node_cnt = {node: np.zeros(num_communities) for node in nodes}

    # 分社区 θ（dict）模式
    if isinstance(theta, dict):
        for cid, nodes_c in communities.items():
            target = int(theta.get(cid, 0))
            if target <= 0 or len(nodes_c) == 0:
                continue
            for _ in range(target):
                root = np.random.choice(nodes_c)  # 仅在该社区内选根
                rr_set = generate_one_rr_set(G, root)
                community_rr_sets[cid].append(rr_set)
                node_rr_sets[root].append(rr_set)
                for u in rr_set:
                    node_cnt[u][cid] += 1
    else:
        # 保持原逻辑：每个节点各生成 theta 个 RR 集
        theta_int = int(theta)
        for v in nodes:
            v_community = node_community.get(v)
            for _ in range(theta_int):
                rr_set = generate_one_rr_set(G, v)
                community_rr_sets[v_community].append(rr_set)
                node_rr_sets[v].append(rr_set)
                for u in rr_set:
                    node_cnt[u][v_community] += 1

    return community_rr_sets, node_rr_sets, node_cnt


def calculate_node_utility(nodes, node_cnt, community_rr_sets):
    """
    计算每个节点的效用向量（分母为 0 时置 0）
    """
    num_communities = len(community_rr_sets)
    community_rr_counts = {cid: len(rrs) for cid, rrs in community_rr_sets.items()}

    node_utility = {}
    for node in nodes:
        utility_vector = np.zeros(num_communities)
        for cid in range(num_communities):
            denom = community_rr_counts.get(cid, 0)
            utility_vector[cid] = (node_cnt[node][cid] / denom) if denom > 0 else 0.0
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


def NodeSelection(RR_sets, k):
    """
    经典 IMM 的 NodeSelection：在 RR-sets 上做贪心覆盖，返回覆盖最多 RR-sets 的 k 个节点。
    """
    if not RR_sets or k <= 0:
        return set()

    cover_dict = defaultdict(set)  # 节点 -> 出现过的 RR 下标集合
    rr_to_nodes = []               # RR 下标 -> 该 RR 的节点集合
    for idx, rr in enumerate(RR_sets):
        rrs = set(rr)
        rr_to_nodes.append(rrs)
        for v in rrs:
            cover_dict[v].add(idx)

    covered = set()  # 已覆盖的 RR 下标
    S = set()

    for _ in range(k):
        best_node, best_new = None, -1
        for v, idxs in cover_dict.items():
            if v in S:
                continue
            gain = len(idxs - covered)
            if gain > best_new:
                best_new = gain
                best_node = v

        if best_node is None or best_new <= 0:
            break

        S.add(best_node)
        newly_covered = cover_dict[best_node] - covered
        covered.update(newly_covered)

        # 增量更新：仅在受影响的 RR 中移除引用
        for rr_idx in newly_covered:
            for u in rr_to_nodes[rr_idx]:
                if rr_idx in cover_dict[u]:
                    cover_dict[u].discard(rr_idx)

    return S


def estimate_theta_per_community(G,
                                 communities,
                                 k,
                                 epsilon=0.1,
                                 ell=1,
                                 random_seed=42,
                                 verbose=False):
    """
    基于 IMM 两阶段采样的分社区 θ 估计。
    返回:
      - theta_per_comm: dict[cid] = θ_c
      - LB_per_comm: dict[cid] = LB_c
    """
    np.random.seed(random_seed)

    theta_per_comm = {}
    LB_per_comm = {}

    for cid, nodes_c in communities.items():
        n_c = len(nodes_c)

        # 边界：空社区
        if n_c <= 0:
            theta_per_comm[cid] = 0
            LB_per_comm[cid] = 0.0
            continue

        n_eff = max(n_c, 2)
        k_eff = min(k, n_c)

        eps_p = epsilon * math.sqrt(2.0)
        # log_term = ln C(n_c,k_eff) + ℓ ln n_eff + ln 2 + ln(log_2 n_eff)
        log_term = (
            math.log(math.comb(n_c, k_eff))
            + ell * math.log(n_eff)
            + math.log(2.0)
            + math.log(math.log2(n_eff))
        )

        # 阶段一：估计 LB_c
        LB_c = 1.0
        max_i = int(math.log2(n_c)) if n_c > 0 else 0
        for i in range(1, max_i):
            x_c = n_c / (2 ** i)
            theta_i_c = math.ceil((n_c * (2.0 + (2.0 / 3.0) * eps_p) * log_term) / ((eps_p ** 2) * x_c))

            RR_sets_c = []
            for _ in range(theta_i_c):
                root = np.random.choice(nodes_c)
                rr = generate_one_rr_set(G, root)
                RR_sets_c.append(rr)

            S_k_c = NodeSelection(RR_sets_c, k_eff)
            covered_cnt = sum(1 for rr in RR_sets_c if rr & S_k_c)
            f_est_c = (covered_cnt / len(RR_sets_c)) if RR_sets_c else 0.0

            if n_c * f_est_c >= (1.0 + eps_p) * x_c:
                LB_c = (n_c * f_est_c) / (1.0 + eps_p)
                break

        LB_per_comm[cid] = LB_c

        # 阶段二：计算最终 θ_c
        alpha_c = math.sqrt(ell * math.log(n_eff) + math.log(4.0))
        beta_c = math.sqrt((1.0 - 1.0 / math.e) * (math.log(math.comb(n_c, k_eff)) + ell * math.log(n_eff) + math.log(4.0)))
        theta_c = math.ceil((2.0 * n_c * (((1.0 - 1.0 / math.e) * alpha_c) + beta_c) ** 2) / (LB_c * (epsilon ** 2)))

        theta_per_comm[cid] = int(theta_c)

    return theta_per_comm, LB_per_comm


def fairness_maximization(node_utility,
                          community_rr_sets,
                          node_rr_sets,
                          node_cnt,
                          community_to_attribute,
                          k=10,
                          verbose=0):
    """
    公平性影响力最大化算法（按新候选停止规则）
    """
    def _node_key(x):
        try:
            return int(x)
        except Exception:
            return str(x)

    nodes = list(node_utility.keys())
    num_communities = len(community_rr_sets)
    community_rr_counts = {cid: len(rrs) for cid, rrs in community_rr_sets.items()}

    # 覆盖标记 + 倒排索引
    covered = {cid: [False] * len(rrs) for cid, rrs in community_rr_sets.items()}
    node_to_rr = _build_node_rr_indices(community_rr_sets)

    S = []
    U = np.zeros(num_communities)

    first_candidate_no_change_count = 0
    rounds_compare = 0
    rounds_all_change = 0
    bottleneck_change_count = 0

    # 第 1 轮：首个种子
    maximin_values = {n: float(np.min(node_utility[n])) for n in nodes}
    best_maximin = max(maximin_values.values())
    level1 = [n for n, v in maximin_values.items() if np.isclose(v, best_maximin)]
    if len(level1) > 1:
        sums = [(n, float(np.sum(node_utility[n]))) for n in level1]
        best_sum = max(s for _, s in sums)
        level2 = [n for n, s in sums if np.isclose(s, best_sum)]
        s1 = min(level2, key=_node_key) if len(level2) > 1 else level2[0]
    else:
        s1 = level1[0]

    S.append(s1)
    U = node_utility[s1].copy()

    # 覆盖包含 s1 的 RR 集（首次覆盖时扣减）
    for (cid, rr_idx) in node_to_rr.get(s1, []):
        if not covered[cid][rr_idx]:
            covered[cid][rr_idx] = True
            for u in community_rr_sets[cid][rr_idx]:
                if u != s1 and u not in S:
                    node_cnt[u][cid] -= 1

    # 重算未选节点效用
    for u in nodes:
        if u != s1:
            for cid in range(num_communities):
                denom = community_rr_counts.get(cid, 0)
                node_utility[u][cid] = (node_cnt[u][cid] / denom) if denom > 0 else 0.0

    # 后续轮次
    for _round in range(2, k + 1):
        bottleneck = int(np.argmin(U))
        remaining = [n for n in nodes if n not in S]
        ordered = sorted(remaining, key=lambda x: node_utility[x][bottleneck], reverse=True)

        scanned = []  # (node, temp_U, min_val, sum_val, min_indices, contrib)
        first_keep_found = False
        keep_contrib = None

        for idx, n in enumerate(ordered):
            temp_U = U + node_utility[n]
            min_val = float(np.min(temp_U))
            min_indices = [j for j, val in enumerate(temp_U) if np.isclose(val, min_val)]
            sum_val = float(np.sum(temp_U))
            contrib = float(node_utility[n][bottleneck])
            scanned.append((n, temp_U, min_val, sum_val, min_indices, contrib))

            # 遇到第一个“不改变瓶颈”的候选
            if bottleneck in min_indices and not first_keep_found:
                first_keep_found = True
                keep_contrib = contrib
                if idx == 0:
                    first_candidate_no_change_count += 1
                else:
                    rounds_compare += 1

                # 纳入与该候选在瓶颈贡献相同的后续节点
                j = idx + 1
                while j < len(ordered):
                    m = ordered[j]
                    m_contrib = float(node_utility[m][bottleneck])
                    if not np.isclose(m_contrib, keep_contrib):
                        break
                    m_temp_U = U + node_utility[m]
                    m_min_val = float(np.min(m_temp_U))
                    m_min_indices = [t for t, val in enumerate(m_temp_U) if np.isclose(val, m_min_val)]
                    m_sum_val = float(np.sum(m_temp_U))
                    scanned.append((m, m_temp_U, m_min_val, m_sum_val, m_min_indices, m_contrib))
                    j += 1
                break  # 截断

        if not first_keep_found:
            rounds_compare += 1
            rounds_all_change += 1

        # 决策：maximin -> sum -> id
        best_maximin = max(item[2] for item in scanned)
        level1 = [it for it in scanned if np.isclose(it[2], best_maximin)]
        if len(level1) > 1:
            best_sum = max(it[3] for it in level1)
            level2 = [it for it in level1 if np.isclose(it[3], best_sum)]
            selected = min(level2, key=lambda x: _node_key(x[0]))[0] if len(level2) > 1 else level2[0][0]
        else:
            selected = level1[0][0]

        # 记录瓶颈是否改变
        prev_bottleneck = bottleneck
        new_U = U + node_utility[selected]
        new_min_val = float(np.min(new_U))
        new_min_indices = [j for j, val in enumerate(new_U) if np.isclose(val, new_min_val)]
        if prev_bottleneck not in new_min_indices:
            bottleneck_change_count += 1

        # 更新 U / S
        U = new_U
        S.append(selected)

        # 覆盖包含 selected 的 RR 集（首次覆盖时扣减）
        for (cid, rr_idx) in node_to_rr.get(selected, []):
            if not covered[cid][rr_idx]:
                covered[cid][rr_idx] = True
                for u in community_rr_sets[cid][rr_idx]:
                    if u != selected and u not in S:
                        node_cnt[u][cid] -= 1

        # 重算未选节点的效用
        for u in ordered:
            if u != selected:
                for cid in range(num_communities):
                    denom = community_rr_counts.get(cid, 0)
                    node_utility[u][cid] = (node_cnt[u][cid] / denom) if denom > 0 else 0.0

    # 覆盖率统计
    total_rr_sets = sum(len(rrs) for rrs in community_rr_sets.values())
    covered_rr_sets = set()
    for seed in S:
        for cid, rrs in community_rr_sets.items():
            for idx, rr in enumerate(rrs):
                if seed in rr:
                    covered_rr_sets.add((cid, idx))
    total_influence = (len(covered_rr_sets) / total_rr_sets) if total_rr_sets > 0 else 0.0
    community_influence = {}
    for cid, rrs in community_rr_sets.items():
        denom = len(rrs)
        if denom > 0:
            covered_comm_rr_sets = sum(1 for (c, _) in covered_rr_sets if c == cid)
            community_influence[cid] = covered_comm_rr_sets / denom
        else:
            community_influence[cid] = 0.0

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
    全局 maximin 贪心影响力最大化（静默）
    """
    def _node_key(x):
        try:
            return int(x)
        except Exception:
            return str(x)

    nodes = list(node_utility.keys())
    num_communities = len(community_rr_sets)
    community_rr_counts = {cid: len(rrs) for cid, rrs in community_rr_sets.items()}

    # 覆盖标记 + 倒排索引
    covered = {cid: [False] * len(rrs) for cid, rrs in community_rr_sets.items()}
    node_to_rr = _build_node_rr_indices(community_rr_sets)

    S = []
    U = np.zeros(num_communities)

    for _round in range(1, k + 1):
        remaining_nodes = [v for v in nodes if v not in S]
        if not remaining_nodes:
            break

        # 计算所有未选节点的 maximin
        maximin_values = {}
        temp_U_cache = {}
        sum_values = {}
        for v in remaining_nodes:
            tU = U + node_utility[v]
            temp_U_cache[v] = tU
            maximin_values[v] = float(np.min(tU))
            sum_values[v] = float(np.sum(tU))

        # 级别1：maximin 最大
        best_maximin = max(maximin_values.values())
        level1 = [v for v, mv in maximin_values.items() if np.isclose(mv, best_maximin)]

        if len(level1) == 1:
            chosen = level1[0]
        else:
            # 级别2：sum 最大
            best_sum = max(sum_values[v] for v in level1)
            level2 = [v for v in level1 if np.isclose(sum_values[v], best_sum)]
            chosen = level2[0] if len(level2) == 1 else min(level2, key=_node_key)

        # 更新 U / S
        U = temp_U_cache[chosen]
        S.append(chosen)

        # 覆盖包含 chosen 的 RR 集（首次覆盖时扣减）
        for (cid, rr_idx) in node_to_rr.get(chosen, []):
            if not covered[cid][rr_idx]:
                covered[cid][rr_idx] = True
                for u in community_rr_sets[cid][rr_idx]:
                    if u != chosen and u not in S:
                        node_cnt[u][cid] -= 1

        # 重新计算未选节点的效用
        for u in remaining_nodes:
            if u != chosen:
                for cid in range(num_communities):
                    denom = community_rr_counts.get(cid, 0)
                    node_utility[u][cid] = (node_cnt[u][cid] / denom) if denom > 0 else 0.0

    # 覆盖率统计
    total_rr_sets = sum(len(rrs) for rrs in community_rr_sets.values())
    covered_rr_sets = set()
    for seed in S:
        for cid, rrs in community_rr_sets.items():
            for idx, rr in enumerate(rrs):
                if seed in rr:
                    covered_rr_sets.add((cid, idx))
    total_influence = (len(covered_rr_sets) / total_rr_sets) if total_rr_sets > 0 else 0.0
    community_influence = {}
    for cid, rrs in community_rr_sets.items():
        denom = len(rrs)
        if denom > 0:
            covered_comm_rr_sets = sum(1 for (c, _) in covered_rr_sets if c == cid)
            community_influence[cid] = covered_comm_rr_sets / denom
        else:
            community_influence[cid] = 0.0

    return {
        "seed_set": S,
        "final_utility": U,
        "total_influence": total_influence,
        "community_influence": community_influence
    }


# ...existing code...
def main(graph_path, attribute_name, theta, k, verbose=1, epsilon_theta=0.1, ell_theta=1, random_seed=42):
    """
    精简主函数输出：
      - 各社区 θ
      - 两种方法的种子集合与累计效用向量
      - 公平性参数
      - 两种算法的运行时间
    """
    np.random.seed(random_seed)
    t0 = time.time()

    # 读图并确保为有向图
    G = nx.read_gml(graph_path)
    if not isinstance(G, nx.DiGraph):
        G = nx.DiGraph(G)

    # 预处理
    G_proc, communities, node_community, community_to_attribute = preprocess_network_by_attribute(G, attribute_name)
    nodes = list(G_proc.nodes())
    num_communities = len(communities)

    # θ：外部 dict 或 IMM 估计
    if isinstance(theta, dict):
        theta_used = {int(cid): int(val) for cid, val in theta.items()}
    else:
        theta_used, _ = estimate_theta_per_community(
            G_proc, communities, k, epsilon=epsilon_theta, ell=ell_theta, random_seed=random_seed, verbose=False
        )

    # 采样 RR 与初始效用
    community_rr_sets, node_rr_sets, node_cnt = generate_reverse_reachable_sets(
        G_proc, communities, node_community, theta=theta_used, random_seed=random_seed
    )
    node_utility = calculate_node_utility(nodes, node_cnt, community_rr_sets)

    # 运行两种算法（函数内部静默）
    fair_time = 0.0
    greedy_time = 0.0

    if k <= 0:
        fairness_result = {
            "seed_set": [],
            "final_utility": np.zeros(num_communities),
            "first_candidate_no_change_count": 0,
            "rounds_compare": 0,
            "rounds_all_change": 0,
            "bottleneck_change_count": 0
        }
        greedy_result = {
            "seed_set": [],
            "final_utility": np.zeros(num_communities),
        }
    else:
        t_fair_start = time.time()
        fairness_result = fairness_maximization(
            node_utility.copy(),
            community_rr_sets,
            node_rr_sets,
            {nid: vec.copy() for nid, vec in node_cnt.items()},
            community_to_attribute,
            k=k,
            verbose=0
        )
        fair_time = time.time() - t_fair_start

        t_greedy_start = time.time()
        greedy_result = maximin_greedy_maximization(
            node_utility.copy(),
            community_rr_sets,
            node_rr_sets,
            {nid: vec.copy() for nid, vec in node_cnt.items()},
            community_to_attribute,
            k=k,
            verbose=0
        )
        greedy_time = time.time() - t_greedy_start

    # 精简输出
    if verbose:
        print(f"\n=== 运行参数 ===")
        print(f"属性: {attribute_name} | k={k} | epsilon={epsilon_theta} | ell={ell_theta} | seed={random_seed}")

        print("\n=== 各社区 θ ===")
        for cid in sorted(communities.keys()):
            cname = community_to_attribute.get(cid, cid)
            th = int(theta_used.get(cid, 0))
            print(f"  {cid}({cname}): θ={th}")

        print("\n=== 公平性算法（Fair-Maximin） ===")
        print("种子集合:", fairness_result["seed_set"])
        U_fair = fairness_result["final_utility"]
        print("累计效用向量 U_fair:")
        for cid in range(num_communities):
            cname = community_to_attribute.get(cid, cid)
            print(f"  {cid}({cname}): {U_fair[cid]:.6f}")

        print("\n=== 全局 Maximin 贪心 ===")
        print("种子集合:", greedy_result["seed_set"])
        U_greedy = greedy_result["final_utility"]
        print("累计效用向量 U_greedy:")
        for cid in range(num_communities):
            cname = community_to_attribute.get(cid, cid)
            print(f"  {cid}({cname}): {U_greedy[cid]:.6f}")

        print("\n=== 公平性参数（Fair-Maximin） ===")
        print("首个候选即不改变瓶颈的轮数:", fairness_result.get("first_candidate_no_change_count", 0))
        print("发生候选比较/截断的轮数:", fairness_result.get("rounds_compare", 0))
        print("全轮均改变瓶颈的轮数:", fairness_result.get("rounds_all_change", 0))
        print("最终选择后瓶颈社区变化的轮数:", fairness_result.get("bottleneck_change_count", 0))

        print("\n=== 运行时间 ===")
        print(f"公平性算法: {fair_time:.2f}s")
        print(f"全局 Maximin 贪心: {greedy_time:.2f}s")

        total_time = time.time() - t0
        print(f"\n总计运行时间: {total_time:.2f}s")

    return {
        "fairness": fairness_result,
        "greedy": greedy_result,
        "theta": theta_used
    }
# ...existing code...

import contextlib
from Maximin_Bottle_500 import main

if __name__ == "__main__":
    with open("output.txt", "w", encoding="utf-8") as f, contextlib.redirect_stdout(f):
        file_path = "C:\\Users\\31062\\Desktop\\科研积累\\code\\graph_spa_500_0.gml"
        for k in [10, 20, 30, 40, 50]:
            print(f"\n=== 运行 k={k} ===")
            main(file_path, attribute_name="gender", theta=None, k=k, verbose=1)