import argparse
import os

task_list = {"arc-e", "arc-c", "boolq"}
model_list = {"../mlora2/meta-llama/Llama-2-7b-hf"}

def main(args):
    for model in model_list:
        if args.multi_task:
            command = ""
            if args.run:
                # Run the model.
                command = f"python launch.py run --base_model {model}"
            elif args.evaluate:
                # Evaluate the model.
                command = f"python launch.py evaluate --base_model {model}"
            cuda_config = f"CUDA_VISIBLE_DEVICES={args.cuda} "
            command = cuda_config + command
            os.system(command)

        else:
            for task in task_list:
                # Generate config
                config = f"python launch.py gen --template mixlora --tasks {task}"

                # Generate command.
                command = ""
                if args.run:
                    # Run the model.
                    config += " --file_name run.json"
                    command = f"python launch.py run --base_model {model} --config run.json"
                elif args.evaluate:
                    # Evaluate the model.
                    config += " --file_name evaluate.json"
                    command = f"python launch.py evaluate --base_model {model} --config evaluate.json"
                cuda_config = f"CUDA_VISIBLE_DEVICES={args.cuda} "
                command = cuda_config + command

                # Run the command.
                os.system(config)
                os.system(command)


def parse_args():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description="Batch run tasks")
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device number")
    parser.add_argument("--run", action="store_true", help="Run the tasks")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the tasks")
    parser.add_argument("--multi-task", action="store_true", help="Run multiple tasks at once")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
