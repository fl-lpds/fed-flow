import os
import subprocess
import time

# List your Docker Compose file paths and corresponding config file paths
docker_compose_configs = [
    {
        "compose_file": "test_classicFL_simnet_1_1_5.yaml",
        "compose_name": "ClassifFL",
    },
    {
        "compose_file": "test_only_edge_simnet_1_1_5.yaml",
        "compose_name": "OnlyEdge",
    },
    {
        "compose_file": "test_only_server_simnet_1_1_5.yaml",
        "compose_name": "OnlyServer",
    },
    {
        "compose_file": "test_random_splitting_simnet_1_1_5.yaml",
        "compose_name": "Random",
    },
    {
        "compose_file": "test_heuristic_offloading_simnet_1_1_5.yaml",
        "compose_name": "OurMethod",
    },
]

SRC_CONFIG_FILE = 'config.py'
DST_CONFIG_FILE = './../../../../app/config/config.py'

# Folder to archive images
ARCHIVE_BASE_FOLDER = './../../../../../results/1_1_5/'


def get_running_containers_with_name(substring):
    """
    Get a list of running containers whose names include the given substring.
    """
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name={substring}", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            check=True,
        )
        containers = result.stdout.strip().splitlines()
        return containers
    except subprocess.CalledProcessError as e:
        print(f"Error while checking containers: {e}")
        return []


def down_compose(compose_file):
    """
    Bring down the Docker Compose stack.
    """
    print(f"Stopping containers for {compose_file}")
    subprocess.run(["docker", "compose", "-f", compose_file, "down"], check=True)


def copy_config_file():
    """
    Copy a configuration file to the application's config destination.
    """
    try:
        with open(f"{SRC_CONFIG_FILE}", "r") as f:
            data = f.read()

        with open(f"{DST_CONFIG_FILE}", "w") as f:
            f.write(data)
    except Exception as e:
        print(f"Error copying config file: {e}")
        raise


def up_compose(compose_file):
    """
    Bring up the Docker Compose stack.
    """
    print(f"Starting containers for {compose_file}")
    subprocess.run(["docker", "compose", "-f", compose_file, "up", "-d"], check=True)


def archive_graphs(compose_name):
    """
    Archive the "Graphs" folder from the container whose name includes "server".
    """
    server_containers = get_running_containers_with_name("server")
    if not server_containers:
        print(f"No running container with 'server' in its name found for {compose_name}. Skipping archiving.")
        return

    container_name = server_containers[0]  # Assuming the first match is the intended container
    archive_folder = os.path.join(ARCHIVE_BASE_FOLDER, compose_name)

    # Ensure the archive directory exists
    os.makedirs(archive_folder, exist_ok=True)

    print(f"Archiving Graphs folder from {container_name} to {archive_folder}...")
    try:
        # Copy the Graphs folder from the container to the host
        subprocess.run(
            [
                "docker", "cp",
                f"{container_name}:/fed-flow/Graphs",  # Adjust this path as needed
                archive_folder
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error archiving Graphs folder: {e}")
        raise


def main():
    current_compose_index = 0

    while current_compose_index < len(docker_compose_configs):
        compose_config = docker_compose_configs[current_compose_index]
        compose_file = compose_config["compose_file"]
        compose_name = compose_config["compose_name"]

        print(f"Preparing to start {compose_file}...")

        # Copy the config file
        copy_config_file()

        # Start the current compose file
        up_compose(compose_file)

        while True:
            running_clients = get_running_containers_with_name("client")
            if not running_clients:
                print("All 'client' containers are down. Archiving and switching to next stack.")

                # Archive the Graphs folder
                archive_graphs(compose_name)

                # Stop the current stack
                down_compose(compose_file)

                current_compose_index += 1
                break

            print(f"Running 'client' containers: {running_clients}")
            # Wait and check again
            time.sleep(60)

        # If we've processed all compose files, exit
        if current_compose_index >= len(docker_compose_configs):
            print("All compose files processed. Exiting.")
            break


if __name__ == "__main__":
    main()
