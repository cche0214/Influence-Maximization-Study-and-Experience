from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import networkx as nx
import numpy as np
from tqdm import tqdm

def load_gml_graph(file_path):
    G = nx.read_gml(file_path)
    return G

def generate_rr_set_general(G, p, start_node, count, record):
    rr_set = set([start_node])
    queue = deque([start_node])

    while queue:
        node = queue.popleft()
        for neighbor in G.predecessors(node):
            if neighbor not in rr_set and np.random.random() < p:
                rr_set.add(neighbor)
                queue.append(neighbor)

    # 更新 count 和 record
    for node in rr_set:
        count[node][start_node] += 1  # 更新 count，记录 v 在以 start_node 为根节点生成的 RR 集中出现的次数
        record[node].append((rr_set, start_node))  # 记录 node 所在的反向可达集编号

    return rr_set, start_node

def generate_rr_sets_general(G, p, num_rr_sets_per_node):
    count = defaultdict(lambda: defaultdict(int))  # count[v][i]: v 在以 i 为根节点生成的 RR 集中出现的次数
    record = defaultdict(list)  # record[v]: v 所在的反向可达集列表 [({rr_set}, root_node), ...]
    record_count = defaultdict(int)

    def process_node(node):
        node_rr_sets = []
        for _ in range(num_rr_sets_per_node):
            rr_set, root_node = generate_rr_set_general(G, p, node, count, record)
            node_rr_sets.append((rr_set, root_node))
            record_count[(frozenset(rr_set),root_node)] +=1
        return node_rr_sets

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_node, node): node for node in G.nodes()}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing nodes", unit="node"):
            future.result()

    return count, record, record_count

def select_seed_nodes_with_rr_optimized(count, record, k, record_count):
    seeds = []
    covered_sets = set()

    for _ in range(k):

        # Step 5: 评估每个节点作为种子的效果
        # 使用生成器表达式统计每个节点的总出现次数
        node_total_counts = defaultdict(int)

        for node, start_nodes in count.items():
            node_total_counts[node] = sum(start_nodes.values())

        # 找到出现次数最多的节点
        best_node = max(node_total_counts, key=lambda x: node_total_counts[x])
        # Step 6: 选择最佳节点作为种子

        seeds.append(best_node)

        # Step 7: 更新 count 和 record，移除包含该节点的反向可达集
        for rr_set, root_node in record[best_node]:
            # 跳过已经移除的反向可达集
            # if rr_set not in covered_sets:
            if record_count[frozenset(rr_set),root_node] > 0:
                covered_sets.add(frozenset(rr_set))
                # 更新 count: 对于反向可达集中的每个节点，减少相应的计数
                for node in rr_set:
                    count[node][root_node] -= 1
                record_count[(frozenset(rr_set),root_node)] -= 1
    # print(count)

    return seeds

def monte_carlo_simulation_general(G, seeds, iterations, prob):
    """使用蒙特卡洛模拟估计种子节点的影响范围。"""
    total_influenced = []  # 记录累计影响节点
    for i in range(iterations):

        influenced = set(seeds)  # 当前轮次被激活的全部节点
        newly_influenced = set(seeds)  # 当前轮次中每一次传播新激活的节点

        while newly_influenced:
            next_influenced = set()
            for node in newly_influenced:
                for neighbor in G.neighbors(node):
                    if neighbor not in influenced and np.random.random() < prob:
                        next_influenced.add(neighbor)

            influenced.update(next_influenced)
            newly_influenced = next_influenced

        total_influenced.append(len(influenced))

    return np.mean(total_influenced)

def main(file_path, k=10):
    G = load_gml_graph(file_path)

    num_rr_sets_per_node = 200
    count, record, record_count = generate_rr_sets_general(G, 0.1, num_rr_sets_per_node)

    seeds = select_seed_nodes_with_rr_optimized(count, record, k, record_count)
    print(f"Selected seed nodes: {seeds}")
    ans = monte_carlo_simulation_general(G, seeds, 10000, 0.1)
    print(f"Monte Carlo: {ans}")


file_path = "datasets/graph_spa_500_0.gml"
main(file_path, k=10)