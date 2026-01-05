import requests
import pandas as pd
import numpy as np
import concurrent.futures
import time
import os

# --- CONFIGURATION ---
API_URL = "http://localhost:8000"
NUM_JOBS = 50  # Total number of jobs to push to Redis
CONCURRENCY = 10  # How many concurrent users to simulate
FILENAME = "stress_data.csv"


def create_dummy_data():
    """Generates a valid CSV file for the system."""
    print(f"Creating dummy data: {FILENAME}...")
    df = pd.DataFrame({
        "age": np.random.randint(20, 80, 200),
        "income": np.random.normal(50000, 15000, 200),
        "credit_score": np.random.randint(300, 850, 200)
    })
    df.to_csv(FILENAME, index=False)


def send_training_job(user_id):
    """Sends a single training request to the API."""
    try:
        with open(FILENAME, "rb") as f:
            # We use a small epoch count (5) to ensure tasks finish quickly
            # so you can see the 'Success' graph in Flower go up.
            response = requests.post(
                f"{API_URL}/train",
                files={"file": f},
                params={"epochs": 5}
            )

        if response.status_code == 200:
            task_id = response.json().get("task_id")
            print(f"[User {user_id}] ✅ Job Queued: {task_id}")
            return task_id
        else:
            print(f"[User {user_id}] ❌ Failed: {response.text}")
            return None

    except Exception as e:
        print(f"[User {user_id}] 💥 Connection Error: {e}")
        return None


def main():
    create_dummy_data()

    print(f"\n--- 🚀 LAUNCHING STRESS TEST ({NUM_JOBS} Jobs) ---")
    print("Check your Flower Dashboard at: http://localhost:5555\n")

    start_time = time.time()

    # We use ThreadPoolExecutor to simulate multiple users hitting the API at once
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        # Submit all jobs
        futures = [executor.submit(send_training_job, i) for i in range(NUM_JOBS)]

        # Wait for them to be accepted by the API
        concurrent.futures.wait(futures)

    duration = time.time() - start_time
    print(f"\n--- 🏁 TEST COMPLETE in {duration:.2f} seconds ---")
    print("The API handled the load. Now watch the Worker drain the Queue!")

    # Cleanup
    if os.path.exists(FILENAME):
        os.remove(FILENAME)


if __name__ == "__main__":
    main()