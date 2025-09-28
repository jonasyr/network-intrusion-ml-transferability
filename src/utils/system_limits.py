"""
System limits and resource management for preventing crashes
"""

import os
import sys
import psutil
import signal
from typing import Dict, Any
from pathlib import Path


class SystemLimitManager:
    """Manage system resources to prevent crashes and segfaults"""

    def __init__(self):
        self.original_limits = {}
        self.process_limit = None

    def setup_safe_environment(self):
        """Configure environment for safe execution"""
        print("🛡️ Setting up safe execution environment...")

        # Limit thread usage for problematic libraries
        thread_limits = {
            "OMP_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        }

        for var, value in thread_limits.items():
            self.original_limits[var] = os.environ.get(var)
            os.environ[var] = value

        # Set XGBoost specific limits
        os.environ["XGBOOST_NTHREAD"] = "1"

        # Limit memory usage
        available_memory = psutil.virtual_memory().available
        memory_limit = min(available_memory // 2, 16 * 1024**3)  # Max 16GB

        try:
            import resource

            # Set memory limit (Linux/Unix only)
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
            print(f"   ✅ Memory limit set to {memory_limit / (1024**3):.1f}GB")
        except (ImportError, OSError):
            print("   ⚠️ Could not set memory limit")

        # Limit process count
        try:
            import resource

            current_limit = resource.getrlimit(resource.RLIMIT_NPROC)
            new_limit = min(current_limit[0], 50)  # Max 50 processes
            resource.setrlimit(resource.RLIMIT_NPROC, (new_limit, current_limit[1]))
            print(f"   ✅ Process limit set to {new_limit}")
        except (ImportError, OSError):
            print("   ⚠️ Could not set process limit")

        print("   ✅ Safe environment configured")

    def restore_environment(self):
        """Restore original environment settings"""
        for var, original_value in self.original_limits.items():
            if original_value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = original_value

    def monitor_resources(self) -> Dict[str, Any]:
        """Monitor current resource usage"""
        process = psutil.Process()

        return {
            "memory_percent": process.memory_percent(),
            "memory_mb": process.memory_info().rss / (1024**2),
            "cpu_percent": process.cpu_percent(),
            "num_threads": process.num_threads(),
            "num_fds": process.num_fds() if hasattr(process, "num_fds") else 0,
            "children": len(process.children()),
        }

    def check_safety_limits(self) -> bool:
        """Check if we're approaching dangerous resource usage"""
        stats = self.monitor_resources()

        # Warning thresholds
        if stats["memory_percent"] > 80:
            print(f"⚠️ High memory usage: {stats['memory_percent']:.1f}%")
            return False

        if stats["num_threads"] > 50:
            print(f"⚠️ High thread count: {stats['num_threads']}")
            return False

        if stats["children"] > 10:
            print(f"⚠️ High child process count: {stats['children']}")
            return False

        return True


def setup_signal_handlers():
    """Setup signal handlers to catch crashes"""

    def signal_handler(signum, frame):
        print(f"\n🚨 Received signal {signum}")
        print("💾 Attempting to save current state...")

        # Try to save any in-progress results
        try:
            from pathlib import Path

            crash_log = Path("data/results/crash_report.txt")
            crash_log.parent.mkdir(parents=True, exist_ok=True)

            with open(crash_log, "w") as f:
                f.write(f"Crash occurred with signal {signum}\n")
                f.write(f"Frame: {frame}\n")

                # Get current resource usage
                try:
                    stats = SystemLimitManager().monitor_resources()
                    f.write(f"Resource usage at crash:\n")
                    for key, value in stats.items():
                        f.write(f"  {key}: {value}\n")
                except:
                    f.write("Could not get resource stats\n")

        except Exception as e:
            print(f"Failed to save crash report: {e}")

        sys.exit(1)

    # Register handlers for common crash signals
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, "SIGSEGV"):
        signal.signal(signal.SIGSEGV, signal_handler)


# Global instance
system_manager = SystemLimitManager()


def with_safe_execution(func):
    """Decorator to run functions with safe execution limits"""

    def wrapper(*args, **kwargs):
        system_manager.setup_safe_environment()
        setup_signal_handlers()

        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            print(f"🚨 Function failed: {e}")
            stats = system_manager.monitor_resources()
            print(
                f"📊 Resource usage: Memory={stats['memory_percent']:.1f}%, Threads={stats['num_threads']}"
            )
            raise
        finally:
            system_manager.restore_environment()

    return wrapper
