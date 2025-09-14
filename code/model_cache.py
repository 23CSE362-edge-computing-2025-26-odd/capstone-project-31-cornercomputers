class EdgeModelCache:
    def __init__(self, storage_limit):
        self.storage_limit = storage_limit  # GB
        self.cache = {}  # model_id -> model metadata
        self.current_storage = 0
        self.hits = 0
        self.misses = 0

    def add_model(self, model_id, model_size):
        evicted_models = []
        if model_id in self.cache:
            self.update_cache_policy(model_id)
            return evicted_models
        while self.current_storage + model_size > self.storage_limit:
            evicted = self.evict_model()
            if evicted is None:
                break
            evicted_models.append(evicted)
        if self.current_storage + model_size > self.storage_limit:
            return evicted_models
        self.cache[model_id] = {
            'size': model_size,
            'last_access': 0
        }
        self.current_storage += model_size
        self.update_cache_policy(model_id)
        return evicted_models

    def evict_model(self):
        if not self.cache:
            return None
        # LRU: evict model with smallest last_access
        lru_model = min(self.cache.items(), key=lambda x: x[1].get('last_access', 0))[0]
        model_size = self.cache[lru_model]['size']
        del self.cache[lru_model]
        self.current_storage -= model_size
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
            # Increment last_access for all, reset for accessed
            for mid in self.cache:
                if mid == model_id:
                    self.cache[mid]['last_access'] = 0
                else:
                    self.cache[mid]['last_access'] += 1

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
def handle_request(user_id, model_id):
    pass

def fetch_from_cloud(model_id):
    pass

def allocate_resources(model_id):
    pass

def run_simulation(requests):
    pass
