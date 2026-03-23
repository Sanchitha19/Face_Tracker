"""
main.py — Entry point for the Face Tracker system.
Run: python main.py
"""

import sys
import os
import time

# Add project root to sys.path for internal imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config_loader import get_config
from src.database import init_db
from src.pipeline import Pipeline
from src.logger import EventLogger


def main():
    print("=" * 60)
    print("  Intelligent Face Tracker — Starting Up")
    print("=" * 60)

    try:
        # Load config
        cfg = get_config()
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    # Init database
    try:
        conn = init_db(cfg["database"]["path"])
    except Exception as e:
        print(f"Error initializing database: {e}")
        sys.exit(1)

    # Init logger (for startup messages)
    try:
        system_logger = EventLogger(
            cfg["logging"]["log_file"],
            cfg["logging"]["image_quality"]
        )
    except Exception as e:
        print(f"Error initializing logger: {e}")
        sys.exit(1)

    system_logger.log_system("System standby. Waiting for START command from dashboard...")
    print("\n  [STANDBY] Waiting for START command from dashboard (app.py)...")

    while True:
        # Check command file
        if os.path.exists("tracker_command.txt"):
            with open("tracker_command.txt", "r") as f:
                cmd = f.read().strip().lower()
            
            if cmd == "start":
                print("\n  [START] Initialization command received!")
                system_logger.log_system("START command received. Launching pipeline.")
                
                # Clear command to avoid immediate restarts
                with open("tracker_command.txt", "w") as f2: f2.write("idle")

                # Reload config (in case it was changed in dashboard)
                cfg = get_config()
                
                # Run pipeline
                pipeline = Pipeline(cfg)
                try:
                    pipeline.run()
                except Exception as e:
                    system_logger.log_system(f"Pipeline error: {e}")
                    print(f"Error: {e}")
                finally:
                    pipeline.cleanup()
                    unique = pipeline.get_visitor_count()
                    system_logger.log_system(f"Tracker stopped. Unique visitors: {unique}")
                    print(f"\n  [STOP] Tracker stopped. Unique visitors: {unique}")
                    print("  Status: STANDBY (Waiting for next START command...)")
        
        time.sleep(1) # Check every second


if __name__ == "__main__":
    main()
