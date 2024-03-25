import threading
import subprocess

def run_script(script_name, month):
    # Split the script name and its arguments into separate elements
    subprocess.run(["python", script_name, "-m", month])

if __name__ == "__main__":
    script1_thread = threading.Thread(target=run_script, args=("./monthly_trade.py", "2023_12"))
    script2_thread = threading.Thread(target=run_script, args=("./monthly_trade.py", "2024_01"))
    script3_thread = threading.Thread(target=run_script, args=("./monthly_trade.py", "2024_02"))

    print("Start multithread.")

    script1_thread.start()
    script2_thread.start()
    script3_thread.start()

    script1_thread.join()
    script2_thread.join()
    script3_thread.join()

    print("All scripts have finished executing.")
