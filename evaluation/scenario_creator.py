import os
import sys

import click
import questionary
from colorama import Fore
from jinja2 import Environment, FileSystemLoader

current_directory = os.path.dirname(os.path.abspath(__file__))
env = Environment(loader=FileSystemLoader(current_directory + '/templates'))
template = env.get_template('docker-compose.yml.j2')


def topology_prompt(ctx, param, value):
    if ctx.params.get('decentralized'):
        return click.prompt('Topology type', type=click.Choice(['ring', 'fully_connected']), default='ring')
    return None


def select_clients_interactive(available_clients, edge_number):
    choices = [questionary.Choice(title=f"Client {c}", value=c) for c in available_clients]
    selected = questionary.checkbox(
        f"Select clients for Edge {edge_number}:",
        choices=choices,
    ).ask()
    return selected or []


def select_clients_non_interactive(available_clients, edge_number):
    print(f"Selecting clients for Edge {edge_number}")
    print("Available clients:", ", ".join(map(str, available_clients)))
    selected = input("Enter client numbers separated by spaces: ").split()
    return [int(c) for c in selected if c.isdigit() and int(c) in available_clients]


def is_interactive():
    return sys.stdin.isatty()


def select_clients(available_clients, edge_number):
    if is_interactive():
        return select_clients_interactive(available_clients, edge_number)
    else:
        return select_clients_non_interactive(available_clients, edge_number)


def create_edge_neighbors_by_topology(num_edges, topology):
    if topology == 'ring':
        if num_edges == 1:
            return {}
        if num_edges == 2:
            return {
                'edge1': ['edge2'],
                'edge2': ['edge1']
            }
        neighbors = {
            f'edge{i}': [f'edge{i - 1}', f'edge{i + 1}'] for i in range(2, num_edges)
        }
        neighbors.update({
            'edge1': ['edge2', f'edge{num_edges}'],
            f'edge{num_edges}': ['edge1', f'edge{num_edges - 1}']
        })
    elif topology == 'fully_connected':
        neighbors = {
            f'edge{i}': [f'edge{j}' for j in range(1, num_edges + 1) if j != i]
            for i in range(1, num_edges + 1)
        }
    else:
        raise ValueError(f"Unsupported topology: {topology}")

    return neighbors


def print_with_color(text, color):
    print(f"{color}{text}{Fore.RESET}")


def render_docker_compose_template(data):
    output = template.render(data)
    with open(current_directory + '/docker-compose.yml', 'w') as f:
        f.write(output)


def create_node(port: int, decentralized: bool, offloading: bool, splitting_method: str, node_index: int,
                node_type: str, round_count: int, device_count: int, neighbors: list[str],
                scenario_description: str, cpu_limit: str = '1', memory_limit: str = '1G') -> dict:
    return {
        'port': port,
        'decentralized': str(decentralized),
        'offloading': str(offloading),
        'splitting_method': splitting_method,
        'node_index': str(node_index - 1),
        'ip': f'{node_type}{node_index}',
        'node_name': f'{node_type}{node_index}',
        'node_type': f'{node_type}',
        'neighbors': ' '.join(neighbors),
        'round_count': round_count,
        'device_count': device_count,
        'scenario_description': scenario_description,
        'cpu_limit': cpu_limit,
        'memory_limit': memory_limit
    }


@click.command()
@click.option('--num-clients', default=1, prompt='Enter number of clients')
@click.option('--num-edges', default=1, prompt='Enter number of edges')
@click.option('--decentralized', is_flag=True, default=False, prompt='Is it decentralized?')
@click.option('--offloading', is_flag=True, default=True, prompt='Is offloading enabled?')
@click.option('--splitting-method', default='fake_decentralized_splitting',
              prompt='Splitting method (find the list of available methods in the splitting.py file)')
@click.option('--topology', default=None, callback=topology_prompt)
@click.option('--round-count', default=2, prompt='Enter number of FL round')
@click.option('--client-cpu-limit', default='0.5', prompt='Enter CPU limit for clients (e.g., 0.5 for half CPU core)')
@click.option('--client-memory-limit', default='512M', prompt='Enter memory limit for clients (e.g., 512M, 1G)')
def create_docker_compose(num_clients, num_edges, decentralized, offloading, splitting_method,
                          topology, round_count, client_cpu_limit, client_memory_limit):
    data = {
        'clients': [],
        'edges': [],
        'servers': []
    }

    scenario_description = f'{"Decentralized" if decentralized else "Centralized"} ' + \
                           f'{topology if decentralized else ""} ' + \
                           f'{num_edges} {num_clients} ' + \
                           f'{"Offloading" if offloading else "No Offloading"}'
    current_port = 8080
    for i in range(1, num_clients + 1):
        client = create_node(current_port, decentralized, offloading, splitting_method, i, 'client', round_count,
                             num_clients, [], scenario_description, client_cpu_limit, client_memory_limit)
        data['clients'].append(client)
        current_port += 1

    edge_clients = {}
    available_clients = list(range(1, num_clients + 1))
    for i in range(1, num_edges + 1):
        edge = create_node(current_port, decentralized, offloading, splitting_method, i, 'edge', round_count,
                           num_clients, [], scenario_description)
        data['edges'].append(edge)
        current_port += 1

        click.echo(f"\nAssigning clients to Edge {i}")
        assigned_clients = select_clients(available_clients, i)
        edge_clients[i] = assigned_clients

        for client in assigned_clients:
            available_clients.remove(client)

    if not decentralized:
        server = create_node(current_port, decentralized, offloading, splitting_method, 1, 'server', round_count,
                             num_clients, [], scenario_description)
        data['servers'].append(server)
        current_port += 1

    if not decentralized:
        server_neighbors = []
        for edge_index in range(1, num_edges + 1):
            data['edges'][edge_index - 1]['neighbors'] = f"server1,{data['servers'][0]['port']}"
            server_neighbors.append(f"edge{edge_index},{data['edges'][edge_index - 1]['port']}")
        data['servers'][0]['neighbors'] = ' '.join(server_neighbors)
    else:
        edge_neighbors = create_edge_neighbors_by_topology(num_edges, topology)
        for edge_id, neighbors in edge_neighbors.items():
            edge_index = int(edge_id[4:])
            data['edges'][edge_index - 1]['neighbors'] = ' '.join(
                [f"{n},{data['edges'][int(n[4:]) - 1]['port']}" for n in neighbors])

    for edge_index, client_list in edge_clients.items():
        edge_clients = [f"client{c},{data['clients'][c - 1]['port']}" for c in client_list]
        data['edges'][edge_index - 1]['neighbors'] += ' ' + ' '.join(edge_clients)

        for client_index in client_list:
            data['clients'][client_index - 1]['neighbors'] = f"edge{edge_index},{data['edges'][edge_index - 1]['port']}"

    render_docker_compose_template(data)
    print_with_color("docker-compose.yml created successfully", Fore.GREEN)


if __name__ == '__main__':
    create_docker_compose()
