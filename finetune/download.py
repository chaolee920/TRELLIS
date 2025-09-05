from datasets import load_dataset

ds = load_dataset("404-Gen/404mini", split="train")
print(len(ds))  # Should be around 20,000 entries