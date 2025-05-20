import os
import sys
import click
import questionary
from colorama import Fore
from jinja2 import Environment, FileSystemLoader

current_directory = os.path.dirname(os.path.abspath(__file__))
env = Environment(loader=FileSystemLoader(os.path.join(current_directory, 'templates')))
template = env.get_template('docker-compose.yml.j2')


def print_with_color(text, color):
    print(f"{color}{text}{Fore.RESET}")


def is_interactive():
    return sys.stdin.isatty()


def topology_prompt(ctx, param, value):
    if ctx.params.get('decentralized') and not ctx.params.get('d2d'):
        return click.prompt('Topology type', type=click.Choice(['ring', 'fully_connected']), default='ring')
    return None


def d2d_cluster_topology_prompt(cluster_number):
    return click.prompt(f'Topology for Cluster {cluster_number}', type=click.Choice(['ring', 'star']), default='ring')


def select_clients_interactive(available_clients, edge_or_cluster_number):
    choices = [questionary.Choice(title=f"Client {c}", value=c) for c in available_clients]
    selected = questionary.checkbox(
        f"Select clients for Edge/Cluster {edge_or_cluster_number}:",
        choices=choices,
    ).ask()
    return selected or []


def select_clients_non_interactive(available_clients, edge_or_cluster_number):
    print(f"Selecting clients for Edge/Cluster {edge_or_cluster_number}")
    print("Available clients:", ", ".join(map(str, available_clients)))
    selected = input("Enter client numbers separated by spaces: ").split()
    return [int(c) for c in selected if c.isdigit() and int(c) in available_clients]


def select_clients(available_clients, edge_or_cluster_number):
    if is_interactive():
        return select_clients_interactive(available_clients, edge_or_cluster_number)
    else:
        return select_clients_non_interactive(available_clients, edge_or_cluster_number)


def create_edge_neighbors_by_topology(num_edges, topology):
    if topology == 'ring':
        if num_edges == 1:
            return {}
        if num_edges == 2:
            return {'edge1': ['edge2'], 'edge2': ['edge1']}
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


def create_d2d_cluster_neighbors(cluster_clients, topology, client_ports, server_port):
    neighbors_map = {}
    if topology == 'ring':
        for i, cid in enumerate(cluster_clients):
            neighbors = [f"server1,{server_port}"]
            if i > 0:
                neighbors.append(f"client{cluster_clients[i - 1]},{client_ports[cluster_clients[i - 1]]}")
            if i < len(cluster_clients) - 1:
                neighbors.append(f"client{cluster_clients[i + 1]},{client_ports[cluster_clients[i + 1]]}")
            neighbors_map[cid] = neighbors
    elif topology == 'star':
        center = cluster_clients[0]
        for cid in cluster_clients:
            neighbors = [f"server1,{server_port}"]
            if cid != center:
                neighbors.append(f"client{center},{client_ports[center]}")
            else:
                for other in cluster_clients[1:]:
                    neighbors.append(f"client{other},{client_ports[other]}")
            neighbors_map[cid] = neighbors
    return neighbors_map


def create_node(port: int, decentralized: bool, offloading: bool, splitting_method: str, node_index: int,
                node_type: str, round_count: int, device_count: int, neighbors: list[str],
                scenario_description: str, cpu_limit: str = '1', memory_limit: str = '1G',
                d2d: bool = False) -> dict:
    return {
        'port': port,
        'decentralized': str(decentralized),
        'offloading': str(offloading),
        'd2d': str(d2d),
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
        'memory_limit': memory_limit,
        'd2d_arg': "-d2d True" if d2d else ""
    }


def render_docker_compose_template(data):
    output = template.render(data)
    with open(os.path.join(current_directory, 'docker-compose.yml'), 'w') as f:
        f.write(output)


@click.command()
@click.option('--num-clients', default=1, prompt='Enter number of clients')
@click.option('--num-edges', default=1, prompt='Enter number of edges')
@click.option('--decentralized', is_flag=True, default=False, prompt='Is it decentralized?')
@click.option('--offloading', is_flag=True, default=True, prompt='Is offloading enabled?')
@click.option('--splitting-method', default='fake_decentralized_splitting',
              prompt='Splitting method (check splitting.py)')
@click.option('--topology', default=None, callback=topology_prompt)
@click.option('--round-count', default=2, prompt='Enter number of FL rounds')
@click.option('--client-cpu-limit', default='2', prompt='CPU limit for clients')
@click.option('--client-memory-limit', default='512M', prompt='Memory limit for clients')
@click.option('--d2d', is_flag=True, default=False, prompt='Is D2D enabled?')
@click.option('--num-clusters', default=1, prompt='Number of D2D clusters')
def create_docker_compose(num_clients, num_edges, decentralized, offloading, splitting_method,
                          topology, round_count, client_cpu_limit, client_memory_limit, d2d, num_clusters):
    data = {
        'clients': [],
        'edges': [],
        'servers': [],
        'broker': None,
        'd2d': d2d
    }

    scenario_description = f"{'Decentralized' if decentralized else 'Centralized'} " +                            f"{'D2D' if d2d else topology if decentralized else ''} " +                            f"{num_edges} {num_clients} " +                            f"{'Offloading' if offloading else 'No Offloading'}"
    current_port = 8080

    if d2d:
        data['broker'] = {
            'image': 'rabbitmq:3-management',
            'container_name': 'rabbitmq',
            'ports': ['5672:5672', '15672:15672'],
            'environment': {
                'RABBITMQ_DEFAULT_USER': 'rabbitmq',
                'RABBITMQ_DEFAULT_PASS': 'rabbitmq',
                'RABBITMQ_MAX_MESSAGE_SIZE': '536870912'
            }
        }

        available_clients = list(range(1, num_clients + 1))
        client_ports = {}
        clusters = []

        for cluster_index in range(1, num_clusters + 1):
            cluster_clients = select_clients(available_clients, cluster_index)
            if not cluster_clients:
                print_with_color(f"Warning: Cluster {cluster_index} has no clients!", Fore.YELLOW)
                continue
            for cid in cluster_clients:
                available_clients.remove(cid)
                client_ports[cid] = current_port
                current_port += 1
            clusters.append((cluster_index, cluster_clients))

        server_port = current_port
        current_port += 1

        all_neighbors = {}
        for cluster_index, cluster_clients in clusters:
            cluster_topology = d2d_cluster_topology_prompt(cluster_index)
            cluster_neighbors = create_d2d_cluster_neighbors(cluster_clients, cluster_topology, client_ports, server_port)
            all_neighbors.update(cluster_neighbors)

        for cid, port in client_ports.items():
            neighbors = all_neighbors.get(cid, [])
            client = create_node(port, decentralized, offloading, splitting_method, cid, 'client', round_count,
                                 num_clients, neighbors, scenario_description,
                                 client_cpu_limit, client_memory_limit, d2d)
            data['clients'].append(client)

        server_neighbors = [f"client{i},{client_ports[i]}" for i in sorted(client_ports)]
        server = create_node(server_port, decentralized, offloading, splitting_method, 1, 'server', round_count,
                             num_clients, server_neighbors, scenario_description, '2', '2G', d2d)
        data['servers'].append(server)

    else:
        available_clients = list(range(1, num_clients + 1))
        edge_clients = {}
        client_ports = {}

        for i in range(1, num_clients + 1):
            client = create_node(current_port, decentralized, offloading, splitting_method, i, 'client', round_count,
                                 num_clients, [], scenario_description, client_cpu_limit, client_memory_limit)
            data['clients'].append(client)
            client_ports[i] = current_port
            current_port += 1

        for i in range(1, num_edges + 1):
            edge = create_node(current_port, decentralized, offloading, splitting_method, i, 'edge', round_count,
                               num_clients, [], scenario_description)
            data['edges'].append(edge)
            current_port += 1

            assigned_clients = select_clients(available_clients, i)
            edge_clients[i] = assigned_clients
            for cid in assigned_clients:
                available_clients.remove(cid)

        if not decentralized:
            server = create_node(current_port, decentralized, offloading, splitting_method, 1, 'server', round_count,
                                 num_clients, [], scenario_description, '2', '2G')
            data['servers'].append(server)
            current_port += 1

            server_neighbors = []
            for edge_index in range(1, num_edges + 1):
                edge_port = data['edges'][edge_index - 1]['port']
                data['edges'][edge_index - 1]['neighbors'] = f"server1,{server['port']}"
                server_neighbors.append(f"edge{edge_index},{edge_port}")
            data['servers'][0]['neighbors'] = ' '.join(server_neighbors)
        else:
            edge_neighbors = create_edge_neighbors_by_topology(num_edges, topology)
            for edge_id, neighbors in edge_neighbors.items():
                edge_index = int(edge_id[4:])
                edge_node = data['edges'][edge_index - 1]
                edge_node['neighbors'] = ' '.join(
                    [f"{n},{data['edges'][int(n[4:]) - 1]['port']}" for n in neighbors])

        for edge_index, client_list in edge_clients.items():
            edge_node = data['edges'][edge_index - 1]
            edge_clients_str = [f"client{c},{client_ports[c]}" for c in client_list]
            edge_node['neighbors'] += ' ' + ' '.join(edge_clients_str)
            for client_index in client_list:
                data['clients'][client_index - 1]['neighbors'] = f"edge{edge_index},{edge_node['port']}"

    render_docker_compose_template(data)
    print_with_color("docker-compose.yml created successfully", Fore.GREEN)


if __name__ == '__main__':
    create_docker_compose()
