import os
import shutil
class EdgeModelCache:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CLOUD_DIR = os.path.join(BASE_DIR, 'cloud')
    CACHE_DIR = os.path.join(BASE_DIR, 'cache')

    def __init__(self, storage_limit, do_file_ops=True):
        self.storage_limit = storage_limit  # GB
        self.cache = {}  # model_id -> model metadata
        self.current_storage = 0
        self.hits = 0
        self.misses = 0
        self.do_file_ops = do_file_ops

    def file_is_cached(self, model_id):
        if not self.do_file_ops:
            return False
        return os.path.exists(os.path.join(self.CACHE_DIR, model_id))

    def file_add_to_cache(self, model_id):
        if not self.do_file_ops:
            return
        src = os.path.join(self.CLOUD_DIR, model_id)
        dst = os.path.join(self.CACHE_DIR, model_id)
        if not os.path.exists(dst):
            shutil.copy(src, dst)

    def file_evict_from_cache(self, model_id):
        if not self.do_file_ops:
            return
        path = os.path.join(self.CACHE_DIR, model_id)
        if os.path.exists(path):
            os.remove(path)

    def add_model(self, model_id, model_size):
        evicted_models = []
        if model_id in self.cache:
            self.update_cache_policy(model_id)
            return evicted_models
        # Evict until there is room or cache is empty
        while self.current_storage + model_size > self.storage_limit:
            if not self.cache:
                # No more models to evict, can't fit this model
                return evicted_models
            evicted = self.evict_model()
            if evicted is None:
                break
            evicted_models.append(evicted)
        if self.current_storage + model_size > self.storage_limit:
            # Still can't fit, don't add
            return evicted_models
        self.cache[model_id] = {
            'size': model_size,
            'last_access': 0
        }
        self.current_storage += model_size
        self.file_add_to_cache(model_id)
        self.update_cache_policy(model_id)
        return evicted_models

    def evict_model(self):
        if not self.cache:
            return None
        lru_model = min(self.cache.items(), key=lambda x: x[1].get('last_access', 0))[0]
        model_size = self.cache[lru_model]['size']
        del self.cache[lru_model]
        self.current_storage -= model_size
        self.file_evict_from_cache(lru_model)
        return lru_model

    def is_cached(self, model_id):
        if model_id in self.cache:
            self.hits += 1
            return True
        else:
            self.misses += 1
            return False

    def get_model(self, model_id):
        return self.cache.get(model_id)

    def update_cache_policy(self, model_id):
        if model_id in self.cache:
            # Increment last_access for all models
            for mid in self.cache:
                self.cache[mid]['last_access'] += 1
            # Reset last_access for accessed model
            self.cache[model_id]['last_access'] = 0

    def get_cache_status(self):
        return {
            'current_storage': self.current_storage,
            'cache': self.cache
        }

    def get_cache_hit_rate(self):
        total_requests = self.hits + self.misses
        if total_requests == 0:
            return 0
        return self.hits / total_requests

# Simulation / environment functions
def handle_request(user_id, model_id, request_info=None, cache=None):
    # Integrate with user request handler: pass user_id, model_id, and cache instance
    if cache is not None and cache.is_cached(model_id):
        cache.update_cache_policy(model_id)
        return {"status": "hit", "source": "cache"}
    else:
        return {"status": "miss", "source": "cloud"}

def fetch_from_cloud(model_id, model_size=1, cache=None, request_info=None):
    import time
    simulated_latency = 0.2  # seconds
    time.sleep(simulated_latency)

    # Simulate fetching model file from cloud to cache
    if cache is not None:
        cache.file_add_to_cache(model_id)
        cache.add_model(model_id, model_size)

    return {
        "model_id": model_id,
        "fetched": True,
        "latency": simulated_latency,
        "from": "cloud",
        "file_path": f"cache/{model_id}"
    }

def allocate_resources(model_id):
    pass

def run_simulation(requests):
    pass
