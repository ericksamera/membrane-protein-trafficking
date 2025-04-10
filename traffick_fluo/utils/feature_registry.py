# traffick_fluo/utils/feature_registry.py

def create_feature_registry():
    registry = []

    def feature(fn):
        registry.append(fn)
        return fn

    return feature, registry
