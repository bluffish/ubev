import os
import pickle
import hashlib
import inspect
import functools

def cached(cache_dir=None):
    def decreator(func):
        nonlocal cache_dir
        func_source = inspect.getsource(func)
        if cache_dir is None:
            cache_dir = os.path.join("cache", func.__name__)
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            hash_obj = {
                "func_name": func.__name__,
                "src": func_source,
                "args": args,
                "kwargs": kwargs,
            }
            hash_data = pickle.dumps(hash_obj)
            hash = hashlib.sha256(hash_data).hexdigest()
            os.makedirs(cache_dir, exist_ok=True)
            if os.path.isfile(os.path.join(cache_dir, f"{hash}.pkl")):
                try:
                    with open(os.path.join(cache_dir, f"{hash}.pkl"), "rb") as f:
                        retval = pickle.load(f)
                    return retval
                except Exception as e:
                    pass

            retval = func(*args, **kwargs)
            with open(os.path.join(cache_dir, f"{hash}.pkl"), "wb") as f:
                pickle.dump(retval, f)
            return retval
        return wrapper
    return decreator
