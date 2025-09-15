from typing import Dict, Any
import time
import json
from model_cache import ModelCache
from TD3 import TD3Agent

AVAILABLE_MODELS = {
    "M1": "Traffic Density Estimation",
    "M2": "Congestion Prediction"
}

class RequestHandler:
    def __init__(self, cache: ModelCache, agent: TD3Agent):
        self.cache = cache
        self.agent = agent

    def handle(self, user_id: str, model_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.time()
        if model_id not in AVAILABLE_MODELS:
            return {"status": "error", "message": f"model {model_id} not supported"}

        state = self._gather_state(user_id, model_id)

        if self.cache.is_cached(model_id):
            output = self._edge_infer(model_id, payload)
            source = "edge"
        else:
            output = self._cloud_infer(model_id, payload)
            source = "cloud"
            reward = self._reward(time.time() - t0, cloud=True)
            self.agent.update(state, action="cloud_fetch", reward=reward)

        return {
            "status": "ok",
            "model": AVAILABLE_MODELS[model_id],
            "source": source,
            "latency": time.time() - t0,
            "result": output
        }

    def _gather_state(self, user_id: str, model_id: str) -> Dict[str, Any]:
        return {
            "user": user_id,
            "model": model_id,
            "cache_size": self.cache.size(),
            "net_quality": self._net_quality()
        }

    def _reward(self, latency: float, cloud: bool) -> float:
        return max(0.0, 1.0 - latency) - (0.2 if cloud else 0.0)

    def _net_quality(self) -> float:
        return 0.9

    def _edge_infer(self, model_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        if model_id == "M1":
            return {"density": self._density_estimate(data)}
        if model_id == "M2":
            return {"congestion": self._congestion_score(data)}

    def _cloud_infer(self, model_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        time.sleep(0.3)
        return {"cloud_inference": True}

    def _density_estimate(self, data):
        return len(data.get("vehicle_boxes", [])) / (data.get("road_length", 1) or 1)

    def _congestion_score(self, data):
        return min(1.0, data.get("avg_wait", 0) / 120.0)


if __name__ == "__main__":
    cache = ModelCache(max_size=2)
    agent = TD3Agent()
    handler = RequestHandler(cache, agent)

    demo = {"vehicle_boxes": [1, 2, 3], "road_length": 0.5}
    print(json.dumps(handler.handle("demo_user", "M1", demo), indent=2))
