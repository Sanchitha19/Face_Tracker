"""
main.py — Entry point for the Face Tracker system.
Run: python main.py
"""

import sys
import os

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

    system_logger.log_system("System started. Loading models...")

    # Run pipeline
    pipeline = Pipeline(cfg)
    try:
        pipeline.run()
    except KeyboardInterrupt:
        system_logger.log_system("Interrupted by user.")
    except Exception as e:
        system_logger.log_system(f"Fatal error: {e}")
        # traceback would be good here but stick to requirement
        raise
    finally:
        pipeline.cleanup()
        unique = pipeline.get_visitor_count()
        system_logger.log_system(
            f"Session ended. Total unique visitors: {unique}"
        )
        print(f"\n  Total unique visitors this session: {unique}")
        print("=" * 60)


if __name__ == "__main__":
    main()
