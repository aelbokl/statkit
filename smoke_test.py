import sys


def main():
    try:
        import analytikit
        from analytikit import stat_kit, cleaning_kit
    except ModuleNotFoundError as exc:
        missing_name = exc.name or "a required package"
        print(
            f"Missing dependency: {missing_name}. Install requirements with 'pip install -r requirements.txt' and try again.",
            file=sys.stderr,
        )
        return 1

    print(f"analytikit package version: {stat_kit.version()}")
    print(f"cleaning_kit version: {cleaning_kit.version()}")
    print(f"plus/minus helper: {analytikit.plus_minus()}")
    print("Smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
