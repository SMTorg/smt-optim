import json
import warnings

import numpy as np


def json_safe(obj):
    try:
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj

        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, dict):
            safe = {}
            for k, v in obj.items():
                try:
                    safe[str(k)] = json_safe(v)
                except:
                    safe[str(k)] = None
            return safe

        if isinstance(obj, (list, tuple)):
            out = []
            for v in obj:
                try:
                    out.append(json_safe(v))
                except Exception:
                    out.append(None)
            return out

        json.dumps(obj)
        return obj

    except Exception as e:
        try:
            obj_repr = repr(obj)
        except Exception:
            obj_repr = "<unrepresentable object>"
        warnings.warn(f"Failed to convert: {obj_repr}. Error message: {e}")
        return None