import random
import numpy as np
from model_cache import handle_request, fetch_from_cloud, EdgeModelCache
from td3_module import TD3, ReplayBuffer


STATE_DIM   = 4
ACTION_DIM  = 2
MAX_ACTION  = 1.0
CACHE_LIMIT = 20     # MB
BANDWIDTH   = 10     # MB per time unit
RUNS        = 1000
USERS       = 4

# Cloud catalogue: model name -> size (MB)
MODEL_CATALOG = {
    "Traffic-Junction-CNN": 1.2,
    "Traffic-Volume-CNN":   11.2,
    "Incident-Detection-CNN": 15.3,
    "Vehicle-Classification-CNN": 5.6
}

# Each user’s probability distribution over models
USER_BIAS = [
    {"Traffic-Junction-CNN":0.6, "Traffic-Volume-CNN":0.25,
     "Incident-Detection-CNN":0.1, "Vehicle-Classification-CNN":0.05},
    {"Traffic-Junction-CNN":0.1, "Traffic-Volume-CNN":0.65,
     "Incident-Detection-CNN":0.2, "Vehicle-Classification-CNN":0.05},
    {"Traffic-Junction-CNN":0.15,"Traffic-Volume-CNN":0.2,
     "Incident-Detection-CNN":0.55,"Vehicle-Classification-CNN":0.1},
    {"Traffic-Junction-CNN":0.05,"Traffic-Volume-CNN":0.15,
     "Incident-Detection-CNN":0.2, "Vehicle-Classification-CNN":0.6},
]


def run_user_requests(cache):
    """Generate biased user requests and update cache + metrics."""
    td3   = TD3(STATE_DIM, ACTION_DIM, MAX_ACTION)
    buff  = ReplayBuffer()
    data_fetched = []
    wait_times   = []

    for t in range(RUNS):
        state = np.random.randn(STATE_DIM)
        action = td3.select_action(state)

        # Four concurrent users
        for u_id in range(USERS):
            # pick a model according to user’s bias
            choices, probs = zip(*USER_BIAS[u_id].items())
            model_name = random.choices(choices, weights=probs, k=1)[0]
            model_size = MODEL_CATALOG[model_name]

            result = handle_request(u_id, model_name, cache=cache)
            if result["status"] == "miss":
                # network transfer time (simple size/bandwidth)
                transfer = model_size / BANDWIDTH
                fetch_from_cloud(model_name, model_size=model_size, cache=cache)
                data_fetched.append(model_size)
                wait_times.append(transfer + 0.2)   # add simulated fetch latency
            else:
                data_fetched.append(0)
                wait_times.append(0)

        # RL training step
        next_state = np.random.randn(STATE_DIM)
        reward = -np.mean(wait_times[-USERS:])   # negative latency as reward
        buff.add((state, next_state, action, reward, 0))
        if len(buff.storage) > 32:
            td3.train(buff, batch_size=32)

    return {
        "avg_wait_time": sum(wait_times)/len(wait_times),
        "avg_data_MB":   sum(data_fetched)/len(data_fetched),
        "cache_hit_rate": cache.get_cache_hit_rate()
    }

if __name__ == "__main__":
    cache = EdgeModelCache(storage_limit=CACHE_LIMIT, do_file_ops=False)
    results = run_user_requests(cache)

    print("=== Simulation Results ===")
    print(f"Average wait time:     {results['avg_wait_time']:.4f} time units")
    print(f"Average data fetched:  {results['avg_data_MB']:.4f} MB")
    print(f"Cache hit rate:        {results['cache_hit_rate']:.2%}")

