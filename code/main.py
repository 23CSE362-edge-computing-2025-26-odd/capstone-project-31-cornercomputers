import simpy
import random

EDGE_COUNT = 10
RUNS = 100_000


class Server:
    def __init__(self, env, name, cpu):
        self.env = env
        self.name = name
        self.cpu = cpu
        self.resource = simpy.Resource(env, capacity=1)

    def run_task(self, task_name, duration):
        with self.resource.request() as req:
            yield req
            print(f"{self.env.now:.2f}: {self.name} starts {task_name}")
            yield self.env.timeout(duration)
            print(f"{self.env.now:.2f}: {self.name} finishes {task_name}")


class Model:
    def __init__(self, name, delay, input_size_range, model_size):
        self.name = name
        self.duration = delay
        self.input_size_range = input_size_range
        self.model_size = model_size  # Irrelevant right now since model isnt being sent anywhere


def dumb_edge_request(
    env, edge, cloud, task_name, task_duration, data_size, bandwidth, metrics
):
    start_time = env.now
    print(
        f"{start_time:.2f}: {edge.name} sends {data_size}MB + request for {task_name} to cloud"
    )

    # simulate transfer to cloud
    transfer_time_to_cloud = data_size / bandwidth
    yield env.timeout(transfer_time_to_cloud)

    # cloud processes data
    yield env.process(cloud.run_task(task_name, task_duration))

    # simulate transfer of result back (assuming same size)
    transfer_time_back = data_size / bandwidth
    yield env.timeout(transfer_time_back)

    end_time = env.now
    total_time = end_time - start_time

    metrics["wait_times"].append(total_time)
    metrics["data_sent"].append(data_size)


env = simpy.Environment()
cloud = Server(env, "cloud", cpu=5000)

# create edge servers
edges = [Server(env, f"edge_{i+1}", cpu=1000) for i in range(EDGE_COUNT)]

# define AI models
ai_models = [
    Model("Traffic-Junction-CNN", 2, (10, 100), 1.2),  # Text data
    Model("Traffic-Volume-CNN", 3, (100, 400), 11.2),  # Video and image
    Model("Incident-Detection-CNN", 5, (100, 400), 15.3),
    Model("Vehicle-Classification-CNN", 2, (100, 400), 5.6),
]

bandwidth = 10  # MB per time unit

metrics_total = {"wait_times": [], "data_sent": []}

for _ in range(RUNS):
    env = simpy.Environment()
    cloud = Server(env, "cloud", cpu=5000)
    edges = [Server(env, f"edge_{i+1}", cpu=1000) for i in range(EDGE_COUNT)]
    ai_models = [
        Model("Traffic-Junction-CNN", 2, (10, 100), 1.2),
        Model("Traffic-Volume-CNN", 3, (100, 400), 11.2),
        Model("Incident-Detection-CNN", 5, (100, 400), 15.3),
        Model("Vehicle-Classification-CNN", 2, (100, 400), 5.6),
    ]
    bandwidth = 10

    # metrics for this run
    run_metrics = {"wait_times": [], "data_sent": []}

    for edge in edges:
        model = random.choice(ai_models)
        data_size = random.randint(*model.input_size_range)
        env.process(
            dumb_edge_request(
                env,
                edge,
                cloud,
                model.name,
                model.duration,
                data_size,
                bandwidth,
                run_metrics,
            )
        )

    env.run()

    # aggregate metrics
    metrics_total["wait_times"].extend(run_metrics["wait_times"])
    metrics_total["data_sent"].extend(run_metrics["data_sent"])

# calculate averages
avg_wait_time = sum(metrics_total["wait_times"]) / len(
    metrics_total["wait_times"]
)
avg_data_sent = sum(metrics_total["data_sent"]) / len(
    metrics_total["data_sent"]
)

print(f"Average wait time per edge: {avg_wait_time:.2f} time units")
print(f"Average data sent per edge: {avg_data_sent:.2f} MB")
