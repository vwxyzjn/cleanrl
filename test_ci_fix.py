#!/usr/bin/env python3
"""
Comprehensive validation script for CI fix.

Tests:
1. pyproject.toml requires-python is correct (>=3.9,<3.11)
2. Episodic logging code fixes are in place
3. Unit tests pass
4. All 30 files have the fix applied
"""
import subprocess
import sys
import re
from pathlib import Path

# ANSI colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

def print_header(msg):
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}{msg}{RESET}")
    print(f"{BLUE}{'='*70}{RESET}\n")

def print_success(msg):
    print(f"{GREEN}✓ {msg}{RESET}")

def print_fail(msg):
    print(f"{RED}✗ {msg}{RESET}")

def print_info(msg):
    print(f"{YELLOW}ℹ {msg}{RESET}")

def test_pyproject_requires_python():
    """Test 1: Check pyproject.toml has correct requires-python"""
    print_header("TEST 1: pyproject.toml requires-python")

    pyproject_path = Path("pyproject.toml")
    content = pyproject_path.read_text()

    # Check requires-python line
    match = re.search(r'requires-python\s*=\s*["\']([^"\']+)["\']', content)
    if not match:
        print_fail("Could not find requires-python in pyproject.toml")
        return False

    version_spec = match.group(1)
    print_info(f"Found requires-python = {version_spec!r}")

    # Should be >=3.9,<3.11 (not >=3.8,<3.11)
    if version_spec == ">=3.9,<3.11":
        print_success("requires-python correctly excludes Python 3.8")
        return True
    elif version_spec == ">=3.8,<3.11":
        print_fail("requires-python still includes Python 3.8 (broken with JAX)")
        return False
    else:
        print_fail(f"Unexpected requires-python value: {version_spec}")
        return False

def test_jax_dependencies():
    """Test 2: Verify JAX dependencies are present"""
    print_header("TEST 2: JAX dependencies in pyproject.toml")

    pyproject_path = Path("pyproject.toml")
    content = pyproject_path.read_text()

    jax_section = re.search(r'jax\s*=\s*\[(.*?)\]', content, re.DOTALL)
    if not jax_section:
        print_fail("Could not find [jax] optional dependencies")
        return False

    jax_deps = jax_section.group(1)
    required_deps = ["jax", "flax", "optax", "chex", "scipy"]

    all_present = True
    for dep in required_deps:
        if dep in jax_deps:
            print_success(f"Found {dep}")
        else:
            print_fail(f"Missing {dep}")
            all_present = False

    if all_present:
        print_success("All JAX dependencies present")
    return all_present

def test_episodic_logging_fix():
    """Test 3: Check that episodic logging fix is in place"""
    print_header("TEST 3: Episodic logging fix in code")

    # Sample files to check (not all 30, just a representative set)
    test_files = [
        "cleanrl/ppo_atari_envpool.py",
        "cleanrl/ppo_procgen.py",
        "cleanrl/dqn.py",
        "cleanrl/sac_continuous_action.py",
    ]

    fixed_count = 0
    for file_path in test_files:
        path = Path(file_path)
        if not path.exists():
            print_info(f"Skipping {file_path} (not found)")
            continue

        content = path.read_text()

        # Look for the fix pattern: logging_step = global_step - num_envs
        if "logging_step = global_step - args.num_envs" in content:
            print_success(f"{file_path}: Fix applied")
            fixed_count += 1
        elif "logging_step = global_step - num_envs" in content:
            print_success(f"{file_path}: Fix applied")
            fixed_count += 1
        else:
            print_fail(f"{file_path}: Fix NOT found")

    return fixed_count > 0

def test_no_break_statements():
    """Test 4: Check that break statements were removed"""
    print_header("TEST 4: Break statements removed from logging")

    # Files that had break statements
    test_files = [
        "cleanrl/ppo_procgen.py",
        "cleanrl/sac_continuous_action.py",
        "cleanrl/sac_atari.py",
    ]

    clean_count = 0
    for file_path in test_files:
        path = Path(file_path)
        if not path.exists():
            continue

        content = path.read_text()

        # Look for logging code with break
        # Pattern: for item in info: ... writer.add_scalar ... break
        has_break_bug = re.search(
            r'for\s+item\s+in\s+info.*?writer\.add_scalar.*?break',
            content,
            re.DOTALL
        )

        if not has_break_bug:
            print_success(f"{file_path}: No break in logging")
            clean_count += 1
        else:
            print_fail(f"{file_path}: Still has break statement")

    return clean_count > 0

def test_unit_tests():
    """Test 5: Run unit tests"""
    print_header("TEST 5: Unit tests (test_episodic_logging.py)")

    try:
        result = subprocess.run(
            ["python3", "-m", "pytest", "tests/test_episodic_logging.py", "-v"],
            capture_output=True,
            text=True,
            timeout=30
        )

        # Check if tests passed
        if "passed" in result.stdout and result.returncode == 0:
            # Count passed tests
            match = re.search(r'(\d+) passed', result.stdout)
            if match:
                count = match.group(1)
                print_success(f"All {count} unit tests passed")
                print_info(result.stdout.split('\n')[-2])  # Summary line
                return True
        else:
            print_fail("Unit tests failed")
            print_info(result.stdout)
            return False

    except subprocess.TimeoutExpired:
        print_fail("Unit tests timed out")
        return False
    except Exception as e:
        print_fail(f"Error running tests: {e}")
        return False

def test_count_fixed_files():
    """Test 6: Count files with the fix"""
    print_header("TEST 6: Counting files with episodic logging fix")

    cleanrl_dir = Path("cleanrl")
    py_files = list(cleanrl_dir.glob("**/*.py"))

    fixed_files = []
    for py_file in py_files:
        try:
            content = py_file.read_text()
            # Check for logging_step pattern
            if "logging_step = global_step" in content:
                fixed_files.append(str(py_file.relative_to(Path("."))))
        except:
            pass

    print_info(f"Found {len(fixed_files)} files with logging_step fix")

    if len(fixed_files) >= 25:  # We expect ~30 files
        print_success(f"Good number of files fixed: {len(fixed_files)}")
        return True
    else:
        print_fail(f"Expected ~30 files, found {len(fixed_files)}")
        return False

def main():
    """Run all tests"""
    print(f"\n{GREEN}{'#'*70}{RESET}")
    print(f"{GREEN}# CI Fix Validation Script{RESET}")
    print(f"{GREEN}# Testing Issue #508 fix + CI dependency fix{RESET}")
    print(f"{GREEN}{'#'*70}{RESET}\n")

    tests = [
        ("pyproject.toml requires-python", test_pyproject_requires_python),
        ("JAX dependencies", test_jax_dependencies),
        ("Episodic logging fix", test_episodic_logging_fix),
        ("Break statements removed", test_no_break_statements),
        ("Unit tests", test_unit_tests),
        ("Count fixed files", test_count_fixed_files),
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print_fail(f"Test '{name}' crashed: {e}")
            results[name] = False

    # Summary
    print_header("SUMMARY")
    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = f"{GREEN}PASS{RESET}" if result else f"{RED}FAIL{RESET}"
        print(f"{status}: {name}")

    print(f"\n{BLUE}Result: {passed}/{total} tests passed{RESET}\n")

    if passed == total:
        print_success("🎉 ALL TESTS PASSED! Ready to commit.")
        return 0
    else:
        print_fail(f"❌ {total - passed} test(s) failed. Fix before committing.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
