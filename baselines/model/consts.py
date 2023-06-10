task_names = [
    "composite",
    "material",
    "object",
    "bootstrap",
    "shape",
    "color",
    "number",
    "pragmatic",
    "relation",
]

task_name2idx = {task_name: idx for idx, task_name in enumerate(task_names)}
idx2task_name = {idx: task_name for idx, task_name in enumerate(task_names)}