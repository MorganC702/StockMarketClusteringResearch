
def _gpu_available() -> bool:
    try:
        import rmm  # RAPIDS memory manager
        import numba.cuda
        return numba.cuda.is_available()
    except Exception:
        return False
