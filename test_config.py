print("=== test_config running ===")

from data_gen.module1_config import load_config

cfg = load_config("config.yaml")
print("Loaded keys:", cfg.keys())
print("duration:", cfg["simulation"]["duration"])
print("seed:", cfg["simulation"]["seed"])
