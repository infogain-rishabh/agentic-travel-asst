#!/usr/bin/env python
import sys
from crew import TravelCrew


def run():
    inputs = {
        # 'user_query': 'plan trip from delhi to goa on 22 jan'
        'user_query': 'I am travelling from bangalore to las vegas on 20th jan 2025. Give me some good hotels where there is casino and suggest me some cheap flights too.'
    }
    # result = TravelCrew().crew().kickoff(inputs=inputs)
    # print(result)
    TravelCrew().crew().kickoff(inputs=inputs)



def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        'user_query': 'plan trip from delhi to dubai from 22-29 jan with return fligts as well',
    }
    try:
        TravelCrew().crew().train(n_iterations=int(sys.argv[1]), inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: main.py <command> [<args>]")
        sys.exit(1)

    command = sys.argv[1]
    if command == "run":
        run()
    elif command == "train":
        train()
    # elif command == "replay":
    #     replay()
    # elif command == "test":
    #     test()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
