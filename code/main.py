import simpy
import random
import numpy as np
import csv
import os
from collections import defaultdict
from model_cache import EdgeModelCache, handle_request
from TD3 import TD3, ReplayBuffer

EDGE_COUNT = 10
RUNS = 1000
STATE_DIM = 4
ACTION_DIM = 2
MAX_ACTION = 1.0
CACHE_SIZE = 20.0
BANDWIDTH = 10.0
POWER_PER_EDGE_W = 5.0
LIVE_PRINT = False

class Server:
    def __init__(self, env, name, cpu):
        self.env = env
        self.name = name
        self.cpu = cpu
        self.resource = simpy.Resource(env, capacity=1)
        self.cache = EdgeModelCache(storage_limit=CACHE_SIZE)
        self.td3 = TD3(STATE_DIM, ACTION_DIM, MAX_ACTION)
        self.replay_buffer = ReplayBuffer()
        self.cumulative_processing_time = 0.0
        self.requests_served = 0

    def run_task(self, task_name, duration):
        with self.resource.request() as req:
            yield req
            start = self.env.now
            yield self.env.timeout(duration)
            end = self.env.now
            self.cumulative_processing_time += (end - start)
            self.requests_served += 1

class Model:
    def __init__(self, name, duration, input_size_range, model_size):
        self.name = name
        self.duration = duration
        self.input_size_range = input_size_range
        self.model_size = model_size

def smart_edge_request(env, edge, cloud, model, run_metrics, global_state):
    start_time = env.now
    run_metrics["total_requests"] += 1
    global_state["model_popularity"][model.name] += 1

    result = handle_request(0, model.name, cache=edge.cache)
    if result["status"] == "miss":
        run_metrics["cloud_requests"] += 1
        transfer_time = model.model_size / BANDWIDTH
        simulated_cloud_latency = 0.2
        yield env.timeout(transfer_time + simulated_cloud_latency)
        evicted = edge.cache.add_model(model.name, model.model_size)
        if evicted:
            run_metrics["cache_evictions"] += len(evicted)
        run_metrics["data_downloaded"] += model.model_size
        run_metrics["per_run_downloaded"] += model.model_size
    else:
        run_metrics["cache_hits"] += 1

    yield env.process(edge.run_task(model.name, model.duration))

    end_time = env.now
    total_time = end_time - start_time
    run_metrics["wait_times"].append(total_time)
    global_state["rewards_list"].append(-total_time)
    run_metrics["rewards"].append(-total_time)

    state = np.random.randn(STATE_DIM)
    next_state = np.random.randn(STATE_DIM)
    action = edge.td3.select_action(state)
    reward = -total_time
    done = 0
    edge.replay_buffer.add((state, next_state, action, reward, done))
    if len(edge.replay_buffer.storage) > 32:
        edge.td3.train(edge.replay_buffer, batch_size=32)

def run_simulation():
    env = simpy.Environment()
    cloud = Server(env, "cloud", cpu=5000)
    edges = [Server(env, f"edge_{i+1}", cpu=1000) for i in range(EDGE_COUNT)]

    ai_models = [
        Model("Traffic-Junction-CNN", 2.0, (10, 100), 1.2),
        Model("Traffic-Volume-CNN", 3.0, (100, 400), 11.2),
        Model("Incident-Detection-CNN", 5.0, (100, 400), 15.3),
        Model("Vehicle-Classification-CNN", 2.0, (100, 400), 5.6),
    ]

    edge_biases = {
        f"edge_{i+1}": {
            "Traffic-Junction-CNN": 0.3,
            "Traffic-Volume-CNN": 0.35,
            "Incident-Detection-CNN": 0.2,
            "Vehicle-Classification-CNN": 0.15,
        } for i in range(EDGE_COUNT)
    }

    metrics_rows = []
    global_state = {
        "model_popularity": defaultdict(int),
        "rewards_list": []
    }

    total_data_downloaded = 0.0
    total_data_uploaded = 0.0
    total_waits = []
    total_cloud_requests = 0
    total_requests = 0
    total_cache_hits = 0
    total_cache_evictions = 0
    total_processing_time_all_edges = 0.0

    for run in range(1, RUNS + 1):
        run_metrics = {
            "wait_times": [],
            "data_downloaded": 0.0,
            "per_run_downloaded": 0.0,
            "data_uploaded": 0.0,
            "cache_hits": 0,
            "cloud_requests": 0,
            "cache_evictions": 0,
            "total_requests": 0,
            "rewards": []
        }

        for edge in edges:
            biases = edge_biases[edge.name]
            model_names = list(biases.keys())
            probabilities = list(biases.values())
            model_name = random.choices(model_names, weights=probabilities, k=1)[0]
            model = next(m for m in ai_models if m.name == model_name)
            env.process(smart_edge_request(env, edge, cloud, model, run_metrics, global_state))

        env.run()

        run_total_requests = run_metrics["total_requests"]
        run_cache_hits = run_metrics["cache_hits"]
        run_cloud_requests = run_metrics["cloud_requests"]
        run_evictions = run_metrics["cache_evictions"]
        run_data_downloaded = run_metrics["per_run_downloaded"]
        run_waits = run_metrics["wait_times"]
        run_rewards = run_metrics["rewards"]

        total_data_downloaded += run_data_downloaded
        total_cloud_requests += run_cloud_requests
        total_requests += run_total_requests
        total_cache_hits += run_cache_hits
        total_cache_evictions += run_evictions
        total_waits.extend(run_waits)
        global_state["rewards_list"].extend(run_rewards)

        sim_time = max(run_waits) if run_waits else 1.0

        cpu_usage_per_edge = []
        for edge in edges:
            cpu_usage = (edge.cumulative_processing_time / sim_time) * 100.0 if sim_time > 0 else 0.0
            cpu_usage_per_edge.append(cpu_usage)
            total_processing_time_all_edges += edge.cumulative_processing_time

        total_sim_time = env.now if env.now > 0 else sim_time
        bandwidth_capacity_total = BANDWIDTH * total_sim_time
        bandwidth_util = (run_data_downloaded / bandwidth_capacity_total) * 100.0 if bandwidth_capacity_total > 0 else 0.0
        throughput = (run_data_downloaded / total_sim_time) if total_sim_time > 0 else 0.0

        avg_wait = (sum(run_waits) / len(run_waits)) if run_waits else 0.0
        cache_hit_rate = (run_cache_hits / run_total_requests) * 100.0 if run_total_requests > 0 else 0.0
        cloud_offload_ratio = (run_cloud_requests / run_total_requests) * 100.0 if run_total_requests > 0 else 0.0
        avg_cpu_util = (sum(cpu_usage_per_edge) / len(cpu_usage_per_edge)) if cpu_usage_per_edge else 0.0
        avg_reward = (sum(run_rewards) / len(run_rewards)) if run_rewards else 0.0
        energy_joules = (sum(edge.cumulative_processing_time for edge in edges) * (POWER_PER_EDGE_W))  # assuming 1 time unit = 1 second

        metrics_rows.append({
            "run": run,
            "avg_wait": avg_wait,
            "cache_hit_rate_pct": cache_hit_rate,
            "data_downloaded_MB": run_data_downloaded,
            "data_uploaded_MB": run_metrics["data_uploaded"],
            "bandwidth_util_pct": bandwidth_util,
            "cloud_offload_pct": cloud_offload_ratio,
            "throughput_MB_per_s": throughput,
            "avg_cpu_util_pct": avg_cpu_util,
            "avg_reward": avg_reward,
            "cache_evictions": run_evictions,
            "energy_joules": energy_joules,
            "sim_time": total_sim_time
        })

        if LIVE_PRINT and run % 10 == 0:
            print(f"run {run:4d} avg_wait {avg_wait:.3f} hit_rate {cache_hit_rate:.2f}% downloaded {run_data_downloaded:.2f}MB reward {avg_reward:.3f}")

    grand_total_requests = total_requests if total_requests > 0 else 1
    overall_metrics = {
        "overall_avg_wait": (sum(total_waits) / len(total_waits)) if total_waits else 0.0,
        "overall_cache_hit_rate": (total_cache_hits / grand_total_requests) * 100.0,
        "overall_data_downloaded_MB": total_data_downloaded,
        "overall_cache_evictions": total_cache_evictions,
        "overall_cloud_offload_pct": (total_cloud_requests / grand_total_requests) * 100.0,
        "overall_avg_cpu_util_pct": (total_processing_time_all_edges / (env.now * len(edges))) * 100.0 if env.now > 0 else 0.0,
        "overall_avg_reward": (sum(global_state["rewards_list"]) / len(global_state["rewards_list"])) if global_state["rewards_list"] else 0.0,
        "model_popularity": dict(global_state["model_popularity"])
    }

    out_dir = os.path.join(os.getcwd(), "simulation_output")
    os.makedirs(out_dir, exist_ok=True)
    csv_file = os.path.join(out_dir, "simulation_metrics.csv")
    with open(csv_file, "w", newline="") as f:
        fieldnames = ["run", "sim_time", "avg_wait", "cache_hit_rate_pct", "data_downloaded_MB", "data_uploaded_MB", "bandwidth_util_pct", "cloud_offload_pct", "throughput_MB_per_s", "avg_cpu_util_pct", "avg_reward", "cache_evictions", "energy_joules"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics_rows:
            writer.writerow(row)

    summary_file = os.path.join(out_dir, "simulation_summary.txt")
    with open(summary_file, "w") as sf:
        sf.write("=== Simulation Summary ===\n")
        sf.write(f"Runs: {RUNS}\n")
        sf.write(f"Overall average wait time: {overall_metrics['overall_avg_wait']:.4f} time units\n")
        sf.write(f"Overall cache hit rate: {overall_metrics['overall_cache_hit_rate']:.2f}%\n")
        sf.write(f"Overall data downloaded (MB): {overall_metrics['overall_data_downloaded_MB']:.2f}\n")
        sf.write(f"Overall cache evictions: {overall_metrics['overall_cache_evictions']}\n")
        sf.write(f"Overall cloud offload ratio: {overall_metrics['overall_cloud_offload_pct']:.2f}%\n")
        sf.write(f"Overall avg CPU util (approx): {overall_metrics['overall_avg_cpu_util_pct']:.2f}%\n")
        sf.write(f"Overall avg RL reward: {overall_metrics['overall_avg_reward']:.4f}\n")
        sf.write("Model popularity counts:\n")
        for m, c in overall_metrics["model_popularity"].items():
            sf.write(f"  {m}: {c}\n")

    print("\n=== Simulation finished ===")
    print(f"Saved metrics CSV: {csv_file}")
    print(f"Saved summary: {summary_file}")
    print("\nSummary (console):")
    print(f"Overall average wait time: {overall_metrics['overall_avg_wait']:.4f} time units")
    print(f"Overall cache hit rate: {overall_metrics['overall_cache_hit_rate']:.2f}%")
    print(f"Overall data downloaded (MB): {overall_metrics['overall_data_downloaded_MB']:.2f}")
    print(f"Overall cache evictions: {overall_metrics['overall_cache_evictions']}")
    print(f"Overall cloud offload ratio: {overall_metrics['overall_cloud_offload_pct']:.2f}%")
    print(f"Overall avg CPU util (approx): {overall_metrics['overall_avg_cpu_util_pct']:.2f}%")
    print(f"Overall avg RL reward: {overall_metrics['overall_avg_reward']:.4f}")
    print("Model popularity counts:")
    for m, c in overall_metrics["model_popularity"].items():
        print(f"  {m}: {c}")

if __name__ == "__main__":
    run_simulation()
