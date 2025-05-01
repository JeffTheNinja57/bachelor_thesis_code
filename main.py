import logging

from experiment.run import parse_args, setup_experiment, run_pso_experiment


def main():
    """Entry point of the program."""
    args = parse_args()
    config = setup_experiment(args)

    try:
        logging.info("Starting PSO-CNN experiment...")
        run_pso_experiment(config)
        logging.info("Experiment finished successfully!")
    except Exception as e:
        logging.critical(f"Experiment failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
