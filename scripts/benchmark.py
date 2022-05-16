"""Run the local hard-negative benchmark after installing the package."""


def main() -> None:
    try:
        from hard_negative_ft.bench import main as benchmark_main
    except ImportError as error:
        raise SystemExit("Install the package or run with PYTHONPATH=src.") from error

    benchmark_main()


if __name__ == "__main__":
    main()
