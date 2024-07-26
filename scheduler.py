from gpu_queue import JobSubmitter

job_array = [
    "python -c 'import os, time;print(\"GPU num utilized\",os.environ[\"CUDA_VISIBLE_DEVICES\"]);time.sleep(3)'",
    "python -c 'import os, time;print(\"GPU num utilized\",os.environ[\"CUDA_VISIBLE_DEVICES\"]);time.sleep(2)'",
    "python -c 'import os, time;print(\"GPU num utilized\",os.environ[\"CUDA_VISIBLE_DEVICES\"]);time.sleep(0.5)'",
    "python -c 'import os, time;print(\"GPU num utilized\",os.environ[\"CUDA_VISIBLE_DEVICES\"]);time.sleep(0.5)'",
    "python -c 'import os, time;print(\"GPU num utilized\",os.environ[\"CUDA_VISIBLE_DEVICES\"]);time.sleep(3)'",
    "python -c 'import os, time;print(\"GPU num utilized\",os.environ[\"CUDA_VISIBLE_DEVICES\"]);time.sleep(1)'",
]

J = JobSubmitter(job_array, [0, 1, 2])
J.submit_jobs()