import simpy
import random
import numpy as np
from model_cache import EdgeModelCache, handle_request, fetch_from_cloud
from TD3 import TD3, ReplayBuffer

EDGE_COUNT = 10
RUNS = 1000
state_dim = 4
action_dim = 2
max_action = 1.0
cache_size = 20  # MB
bandwidth = 10  # MB per time unit


class Server:
    def __init__(self, env, name, cpu):
        self.env = env
        self.name = name
        self.cpu = cpu
        self.resource = simpy.Resource(env, capacity=1)
        self.cache = EdgeModelCache(
            storage_limit=cache_size, do_file_ops=False
        )
        self.td3 = TD3(state_dim, action_dim, max_action)
        self.replay_buffer = ReplayBuffer()

    def run_task(self, task_name, duration):
        with self.resource.request() as req:
            yield req
            print(
                f"{self.env.now:.2f}: {self.name} starts running {task_name}"
            )
            yield self.env.timeout(duration)
            print(f"{self.env.now:.2f}: {self.name} finished {task_name}")


class Model:
    def __init__(self, name, duration, input_size_range, model_size):
        self.name = name
        self.duration = duration
        self.input_size_range = input_size_range
        self.model_size = model_size  # MB


def smart_edge_request(env, edge, cloud, model, metrics):
    start_time = env.now

    print(f"{start_time:.2f}: {edge.name} wants model {model.name}")

    # simulate cache decision time (small relative to model runtime)
    # decision_time = model.duration * 0.5
    # yield env.timeout(decision_time)

    # check if model is in cache
    user_id = 0
    result = handle_request(user_id, model.name, cache=edge.cache)
    if result["status"] == "miss":
        print(
            f"{env.now:.2f}: - {edge.name} cache miss for {model.name}, fetching from cloud"
        )
        transfer_time = model.model_size / bandwidth
        yield env.timeout(transfer_time)
        fetch_from_cloud(
            model.name, model_size=model.model_size, cache=edge.cache
        )
        metrics["data_received"].append(model.model_size)
        print(f"{env.now:.2f}: {edge.name} fetched {model.name} from cloud")

    else:
        metrics["data_received"].append(0)
        print(f"{env.now:.2f}: + {edge.name} cache hit for {model.name}")

    # run model locally
    yield env.process(edge.run_task(model.name, model.duration))

    end_time = env.now
    total_time = end_time - start_time

    # record metrics
    metrics["wait_times"].append(total_time)

    # simulate TD3 training after task
    state = np.random.randn(state_dim)
    next_state = np.random.randn(state_dim)
    action = edge.td3.select_action(state)
    reward = -total_time
    done = 0
    edge.replay_buffer.add((state, next_state, action, reward, done))
    if len(edge.replay_buffer.storage) > 32:
        edge.td3.train(edge.replay_buffer, batch_size=32)


# define AI models
ai_models = [
    Model("Traffic-Junction-CNN", 2, (10, 100), 1.2),
    Model("Traffic-Volume-CNN", 3, (100, 400), 11.2),
    Model("Incident-Detection-CNN", 5, (100, 400), 15.3),
    Model("Vehicle-Classification-CNN", 2, (100, 400), 5.6),
]

edge_biases = {
    "edge_1": {
        "Traffic-Junction-CNN": 0.6,
        "Traffic-Volume-CNN": 0.25,
        "Incident-Detection-CNN": 0.1,
        "Vehicle-Classification-CNN": 0.05,
    },
    "edge_2": {
        "Traffic-Junction-CNN": 0.1,
        "Traffic-Volume-CNN": 0.65,
        "Incident-Detection-CNN": 0.2,
        "Vehicle-Classification-CNN": 0.05,
    },
    "edge_3": {
        "Traffic-Junction-CNN": 0.15,
        "Traffic-Volume-CNN": 0.2,
        "Incident-Detection-CNN": 0.55,
        "Vehicle-Classification-CNN": 0.1,
    },
    "edge_4": {
        "Traffic-Junction-CNN": 0.05,
        "Traffic-Volume-CNN": 0.15,
        "Incident-Detection-CNN": 0.2,
        "Vehicle-Classification-CNN": 0.6,
    },
    "edge_5": {
        "Traffic-Junction-CNN": 0.3,
        "Traffic-Volume-CNN": 0.45,
        "Incident-Detection-CNN": 0.15,
        "Vehicle-Classification-CNN": 0.1,
    },
    "edge_6": {
        "Traffic-Junction-CNN": 0.4,
        "Traffic-Volume-CNN": 0.4,
        "Incident-Detection-CNN": 0.1,
        "Vehicle-Classification-CNN": 0.1,
    },
    "edge_7": {
        "Traffic-Junction-CNN": 0.05,
        "Traffic-Volume-CNN": 0.25,
        "Incident-Detection-CNN": 0.6,
        "Vehicle-Classification-CNN": 0.1,
    },
    "edge_8": {
        "Traffic-Junction-CNN": 0.1,
        "Traffic-Volume-CNN": 0.15,
        "Incident-Detection-CNN": 0.25,
        "Vehicle-Classification-CNN": 0.5,
    },
    "edge_9": {
        "Traffic-Junction-CNN": 0.3,
        "Traffic-Volume-CNN": 0.3,
        "Incident-Detection-CNN": 0.25,
        "Vehicle-Classification-CNN": 0.15,
    },
    "edge_10": {
        "Traffic-Junction-CNN": 0.55,
        "Traffic-Volume-CNN": 0.3,
        "Incident-Detection-CNN": 0.1,
        "Vehicle-Classification-CNN": 0.05,
    },
}

metrics_total = {"wait_times": [], "data_received": []}


env = simpy.Environment()
cloud = Server(env, "cloud", cpu=5000)
edges = [Server(env, f"edge_{i+1}", cpu=1000) for i in range(EDGE_COUNT)]

for run in range(RUNS):
    print(f"\n--- Simulation run {run+1} ---")

    run_metrics = {"wait_times": [], "data_received": []}

    # schedule parallel requests
    for edge in edges:
        biases = edge_biases[edge.name]
        model_names = list(biases.keys())
        probabilities = list(biases.values())
        model_name = random.choices(model_names, weights=probabilities, k=1)[0]
        model = next(m for m in ai_models if m.name == model_name)
        env.process(smart_edge_request(env, edge, cloud, model, run_metrics))

    env.run()

    # aggregate metrics
    metrics_total["wait_times"].extend(run_metrics["wait_times"])
    metrics_total["data_received"].extend(run_metrics["data_received"])

# calculate averages
avg_wait_time = sum(metrics_total["wait_times"]) / len(
    metrics_total["wait_times"]
)
avg_data_received = sum(metrics_total["data_received"]) / len(
    metrics_total["data_received"]
)

print(f"\nAverage wait time per edge: {avg_wait_time:.2f} time units")
print(f"Average data received per edge: {avg_data_received:.2f} MB")
