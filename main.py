from src.app.transform_query_logs import main as transform_main
from src.app.parse_athena_executions import main as parse_athena_main
from src.app.prediction import main as execute_prediction

from dotenv import load_dotenv

load_dotenv()


def main():
    """
    A menu-driven command-line interface for processing Athena executions, transforming query logs,
    and making predictions.

    The user is presented with a list of actions to choose from:

    - **1. Process Athena Executions**: Invoke the main function from ``parse_athena_executions``.
    - **2. Transform Query Logs**: Invoke the main function from ``transform_query_logs``.
    - **3. Run The Whole Pipeline**: First process Athena executions and then transform query logs.
    - **4. Make Prediction**: Invoke the make_prediction function from ``prediction.py``.
    - **5. Exit**: Exit the application.

    **Note**:
        The user is repeatedly prompted to choose an action until they choose to exit.
        Invalid choices prompt a warning and re-display the menu.

    **Example Usage**::

        Choose an action:
        1. Process Athena Executions
        2. Transform Query Logs
        3. Run The Whole Pipeline
        4. Make Prediction
        5. Exit

        Enter your choice (1/2/3/4/5): 4
        [Output from the Make Prediction function]
        ...

    :return: None
    """
    while True:
        print("\nChoose an action:")
        print("1. Process Athena Executions")
        print("2. Transform Query Logs")
        print("3. Run The Whole Pipeline")
        print("4. Make Prediction")
        print("5. Exit")

        choice = input("\nEnter your choice (1/2/3/4/5): ")

        if choice == "1":
            parse_athena_main()
        elif choice == "2":
            transform_main()
        elif choice == "3":
            parse_athena_main()
            transform_main()
            execute_prediction()
        elif choice == "4":
            execute_prediction()
        elif choice == "5":
            print("Exiting the application. Goodbye!")
            break
        else:
            print("Invalid choice. Please select a valid option.")


if __name__ == "__main__":
    main()
