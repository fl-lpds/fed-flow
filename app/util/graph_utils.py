import csv
import os
import random
import shutil
import time
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import yaml

import app.util.model_utils as model_utils
from app.config import config
from app.config.logger import fed_logger
from app.entity.node import Node


def save_results_to_csv(node: Node, training_times: list[float], client_bandwidths: list[float],
                        accuracy: list[float], save_path: str, neighbor_bandwidths: Optional[list[float]] = None):
    """Save numerical results to a CSV file."""
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    csv_file = os.path.join(save_path, f"results-{str(node)}.csv")
    rounds_count = len(training_times)
    
    # Prepare data for CSV
    headers = ["Round", "Training Time (s)", "Bandwidth (bytes/s)", "Accuracy (%)"]
    if neighbor_bandwidths:
        headers.append("Neighbor Bandwidth (bytes/s)")
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for i in range(rounds_count):
            row = [
                i + 1,  # Round number
                training_times[i],
                client_bandwidths[i],
                accuracy[i]
            ]
            if neighbor_bandwidths:
                row.append(neighbor_bandwidths[i])
            writer.writerow(row)
    
    fed_logger.info(f"CSV results saved to {csv_file}")


def report_results(node: Node, training_times: list[float], client_bandwidths: list[float],
                   accuracy: list[float], neighbor_bandwidths: Optional[list[float]] = None, accuracy_duration: bool = True):
    current_time = time.strftime("%Y-%m-%d %H:%M")
    runtime_config = f'{current_time} {config.SCENARIO_DESCRIPTION}'
    save_path = f"Results/{runtime_config}"
    rounds_count = config.R
    
    # Save numerical data to CSV
    save_results_to_csv(node, training_times, client_bandwidths, accuracy, save_path, neighbor_bandwidths)
    
    # Generate plots
    draw_graph(10, 5, range(1, rounds_count + 1), training_times, str(node), "FL Rounds", "Training Time (s)",
               save_path, f"training-time-{str(node)}")
    draw_graph(10, 5, range(1, rounds_count + 1), client_bandwidths, str(node), "FL Rounds", "Bandwidths (bytes/s)",
               save_path, f"bandwidth-{str(node)}")
    draw_graph(10, 5, range(1, rounds_count + 1), accuracy, str(node), "FL Rounds", "Accuracy (%)",
               save_path, f"accuracy-{str(node)}")
    if neighbor_bandwidths:
        draw_graph(10, 5, range(1, rounds_count + 1), neighbor_bandwidths, str(node), "FL Rounds",
                   "Neighbors Bandwidths (bytes/s)",
                   save_path, f"neighbor-bandwidths-{str(node)}")
    if accuracy_duration:
        timeline = [0]
        for duration in training_times:
            timeline.append(timeline[-1] + duration)
        draw_graph(10, 5, timeline[1:], accuracy, str(node), "Time (s)", "Accuracy (%)",
                   save_path, f"accuracy-duration-{str(node)}")
    copy_compose_file_if_exists(save_path)
    
    # Generate architecture diagram from docker-compose.yml
    compose_file = os.path.join(save_path, 'docker-compose.yml')
    if os.path.exists(compose_file):
        generate_architecture_diagram(compose_file, save_path, 'architecture')
    
    fed_logger.info(f"Results created successfully at {save_path}")


def draw_graph(figSizeX, figSizeY, x, y, title, xlabel, ylabel, savePath, pictureName, saveFig=True):
    # Create a plot
    plt.figure(figsize=(int(figSizeX), int(figSizeY)))  # Set the figure size
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if saveFig:
        if not os.path.exists(savePath):
            os.makedirs(savePath, exist_ok=True)
        plt.savefig(os.path.join(savePath, pictureName))
    plt.close()


def copy_compose_file_if_exists(dest):
    src = 'evaluation/docker-compose.yml'
    dest += '/docker-compose.yml'
    if os.path.isfile(src):
        try:
            shutil.copy(src, dest)
            fed_logger.info(f"File '{src}' copied to '{dest}' successfully.")
        except Exception as e:
            print(f"Failed to copy file: {e}")
    else:
        print(f"File '{src}' does not exist.")


def parse_docker_compose_for_architecture(compose_file_path):
    """Parse docker-compose.yml to extract node information and connections."""
    try:
        with open(compose_file_path, 'r') as f:
            compose_data = yaml.safe_load(f)
    except Exception as e:
        fed_logger.error(f"Failed to parse docker-compose.yml: {e}")
        return None, None, None
    
    nodes = {}
    connections = []
    
    services = compose_data.get('services', {})
    
    # Check if D2D architecture
    is_d2d = False
    clustering_info = None
    
    for service_name, service_config in services.items():
        # Skip broker and node_base
        if service_name in ['broker', 'node_base']:
            continue
        
        env = service_config.get('environment', {})
        node_type = env.get('NODE_TYPE', 'unknown')
        neighbors = env.get('NEIGHBORS', '')
        ip = env.get('IP', service_name)
        port = env.get('PORT', '')
        d2d_flag = env.get('D2D', 'False')
        clustering = env.get('CLUSTERING', 'none_clustering')
        
        # Detect D2D architecture
        if d2d_flag == 'True':
            is_d2d = True
            clustering_info = clustering
        
        # Store node info
        nodes[service_name] = {
            'type': node_type,
            'ip': ip,
            'port': port,
            'neighbors': []
        }
        
        # Parse neighbor connections
        if neighbors:
            neighbor_list = neighbors.strip().split()
            for neighbor in neighbor_list:
                if ',' in neighbor:
                    neighbor_name, neighbor_port = neighbor.split(',')
                    nodes[service_name]['neighbors'].append(neighbor_name)
                    # Create bidirectional connection
                    connections.append((service_name, neighbor_name))
    
    # Build cluster information
    clusters = {}
    if is_d2d:
        # For D2D: detect clusters by analyzing client-to-client connections
        # Use connected components to find which clients gossip together
        clients = [name for name, info in nodes.items() if info['type'] == 'client']
        servers = [name for name, info in nodes.items() if info['type'] == 'server']
        
        # Build a graph of client-to-client connections (excluding server)
        client_graph = nx.Graph()
        for client in clients:
            client_graph.add_node(client)
        
        for client in clients:
            client_neighbors = nodes[client]['neighbors']
            for neighbor in client_neighbors:
                # Only add edges between clients (not to servers)
                if neighbor in clients:
                    client_graph.add_edge(client, neighbor)
        
        # Find connected components (clusters)
        connected_components = list(nx.connected_components(client_graph))
        for idx, component in enumerate(connected_components):
            cluster_name = f'cluster{idx + 1}'
            clusters[cluster_name] = list(component)
    
    elif any(n['type'] == 'edge' for n in nodes.values()):
        # For hierarchical architectures, cluster clients by their connected edge
        edges = [name for name, info in nodes.items() if info['type'] == 'edge']
        clients = [name for name, info in nodes.items() if info['type'] == 'client']
        
        # Assign clients to clusters based on their edge connections
        for client in clients:
            client_neighbors = nodes[client]['neighbors']
            # Find which edge this client is connected to
            cluster_parent = None
            for neighbor in client_neighbors:
                if neighbor in edges:
                    cluster_parent = neighbor
                    break
            
            if cluster_parent:
                if cluster_parent not in clusters:
                    clusters[cluster_parent] = []
                clusters[cluster_parent].append(client)
            else:
                # Orphan client - create its own cluster
                if 'orphan' not in clusters:
                    clusters['orphan'] = []
                clusters['orphan'].append(client)
    
    architecture_info = {
        'is_d2d': is_d2d,
        'clustering': clustering_info,
        'clusters': clusters
    }
    
    return nodes, connections, architecture_info


def get_cluster_colors(num_clusters):
    """Generate distinct colors for clusters, avoiding blue and yellow."""
    # Color palette excluding blue and yellow shades
    base_colors = [
        '#FF6B6B',  # Red
        '#95E1D3',  # Mint
        '#F38181',  # Light coral
        '#AA96DA',  # Purple
        '#FCBAD3',  # Pink
        '#A8D8EA',  # Sky blue (light)
        '#FFD93D',  # Light yellow (acceptable)
        '#6BCB77',  # Green
        '#FD79A8',  # Rose
        '#74B9FF',  # Light blue
        '#FF7675',  # Salmon
        '#A29BFE',  # Lavender
        '#FD79A8',  # Pink rose
        '#6C5CE7',  # Indigo
        '#00B894',  # Teal green
    ]
    
    # If we need more colors, generate them
    if num_clusters > len(base_colors):
        import colorsys
        colors = []
        for i in range(num_clusters):
            hue = i / num_clusters
            # Avoid hues around blue (0.5-0.7) and yellow (0.15-0.17)
            if 0.5 <= hue <= 0.7:
                hue = (hue - 0.5) * 0.5  # Shift to red/orange
            rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
            colors.append('#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)))
        return colors
    
    return base_colors[:num_clusters]


def create_hierarchical_layout(G, nodes, architecture_info):
    """Create hierarchical layout for centralized/edge architectures."""
    pos = {}
    clusters = architecture_info['clusters']
    
    # Separate nodes by type
    servers = [name for name, info in nodes.items() if info['type'] == 'server']
    edges = [name for name, info in nodes.items() if info['type'] == 'edge']
    clients = [name for name, info in nodes.items() if info['type'] == 'client']
    
    # Layer 0: Servers at the top
    y_offset = 2.0
    if servers:
        x_spacing = 3.0 / max(len(servers), 1)
        for i, server in enumerate(servers):
            x = -1.5 + (i + 0.5) * x_spacing
            pos[server] = (x, y_offset)
    
    # Layer 1: Edge servers
    y_offset = 0.0
    if edges:
        x_spacing = 5.0 / max(len(edges), 1)
        for i, edge in enumerate(edges):
            x = -2.5 + (i + 0.5) * x_spacing
            pos[edge] = (x, y_offset)
    
    # Layer 2: Clients grouped by cluster
    y_offset = -2.0
    cluster_items = list(clusters.items())
    
    if cluster_items:
        cluster_width = 6.0 / len(cluster_items)
        for cluster_idx, (cluster_parent, cluster_clients) in enumerate(cluster_items):
            if not cluster_clients:
                continue
            
            # Center x position for this cluster
            cluster_center_x = -3.0 + (cluster_idx + 0.5) * cluster_width
            
            # Arrange clients in this cluster
            num_clients = len(cluster_clients)
            if num_clients == 1:
                pos[cluster_clients[0]] = (cluster_center_x, y_offset)
            else:
                client_spacing = min(cluster_width * 0.8 / num_clients, 0.5)
                for i, client in enumerate(cluster_clients):
                    x = cluster_center_x - (num_clients - 1) * client_spacing / 2 + i * client_spacing
                    pos[client] = (x, y_offset)
    
    # Handle any clients not in clusters
    orphan_clients = [c for c in clients if c not in pos]
    if orphan_clients:
        y_offset = -2.0
        x_spacing = 6.0 / max(len(orphan_clients), 1)
        for i, client in enumerate(orphan_clients):
            x = -3.0 + (i + 0.5) * x_spacing
            pos[client] = (x, y_offset)
    
    return pos


def create_d2d_layout(G, nodes, architecture_info):
    """Create radial layout for D2D architectures with server in center."""
    pos = {}
    clusters = architecture_info['clusters']
    
    servers = [name for name, info in nodes.items() if info['type'] == 'server']
    
    # Place server(s) in center
    if servers:
        for server in servers:
            pos[server] = (0, 0)
    
    # Arrange clusters radially around server
    cluster_items = list(clusters.items())
    num_clusters = len(cluster_items)
    
    if num_clusters > 0:
        import math
        for cluster_idx, (cluster_parent, cluster_clients) in enumerate(cluster_items):
            if not cluster_clients:
                continue
            
            # Calculate angle for this cluster
            angle = 2 * math.pi * cluster_idx / num_clusters
            
            # Radius for cluster
            radius = 2.5
            
            # Center position for this cluster
            cluster_center_x = radius * math.cos(angle)
            cluster_center_y = radius * math.sin(angle)
            
            # Arrange clients within cluster
            num_clients = len(cluster_clients)
            if num_clients == 1:
                pos[cluster_clients[0]] = (cluster_center_x, cluster_center_y)
            else:
                # Arrange in a small circle around cluster center
                sub_radius = 0.3 + 0.05 * num_clients
                for i, client in enumerate(cluster_clients):
                    sub_angle = angle + (2 * math.pi * i / num_clients)
                    x = cluster_center_x + sub_radius * math.cos(sub_angle)
                    y = cluster_center_y + sub_radius * math.sin(sub_angle)
                    pos[client] = (x, y)
    
    return pos


def generate_architecture_diagram(compose_file_path, save_path, diagram_name='architecture'):
    """Generate a visual diagram of the federated learning architecture from docker-compose.yml."""
    nodes, connections, architecture_info = parse_docker_compose_for_architecture(compose_file_path)
    
    if nodes is None:
        fed_logger.error("Failed to generate architecture diagram")
        return
    
    # Create directed graph
    G = nx.Graph()
    
    # Add nodes with their types
    for node_name, node_info in nodes.items():
        G.add_node(node_name, node_type=node_info['type'])
    
    # Add edges (connections)
    for source, target in connections:
        if source in G.nodes and target in G.nodes:
            G.add_edge(source, target)
    
    # Determine node colors based on architecture and clusters
    node_colors = []
    node_sizes = []
    clusters = architecture_info['clusters']
    is_d2d = architecture_info['is_d2d']
    
    # Generate cluster colors
    cluster_color_map = {}
    if clusters:
        cluster_names = list(clusters.keys())
        cluster_palette = get_cluster_colors(len(cluster_names))
        for i, cluster_name in enumerate(cluster_names):
            cluster_color_map[cluster_name] = cluster_palette[i]
    
    # Node to cluster mapping
    node_to_cluster = {}
    for cluster_name, cluster_members in clusters.items():
        for member in cluster_members:
            node_to_cluster[member] = cluster_name
    
    # Base colors for special nodes
    server_color = '#4169E1'  # Royal Blue
    edge_color = '#FFD700'    # Gold
    
    size_map = {
        'server': 3500,
        'edge': 2500,
        'client': 1800,
        'unknown': 1000
    }
    
    # Assign colors and sizes
    legend_clusters = set()
    for node in G.nodes():
        node_type = nodes[node]['type']
        node_sizes.append(size_map.get(node_type, size_map['unknown']))
        
        if node_type == 'server':
            node_colors.append(server_color)
        elif node_type == 'edge':
            node_colors.append(edge_color)
        elif node_type == 'client':
            # Color based on cluster
            if node in node_to_cluster:
                cluster = node_to_cluster[node]
                node_colors.append(cluster_color_map[cluster])
                legend_clusters.add(cluster)
            else:
                node_colors.append('#CCCCCC')  # Gray for orphan clients
        else:
            node_colors.append('#CCCCCC')
    
    # Create the plot
    plt.figure(figsize=(16, 12))
    
    # Choose layout based on architecture type
    if is_d2d:
        pos = create_d2d_layout(G, nodes, architecture_info)
        title = 'D2D Federated Learning Architecture'
    elif any(n['type'] == 'edge' for n in nodes.values()):
        pos = create_hierarchical_layout(G, nodes, architecture_info)
        title = 'Hierarchical Federated Learning Architecture'
    elif any(n['type'] == 'server' for n in nodes.values()):
        pos = create_hierarchical_layout(G, nodes, architecture_info)
        title = 'Centralized Federated Learning Architecture'
    else:
        pos = nx.circular_layout(G)
        title = 'Decentralized Federated Learning Architecture'
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='#888888', width=1.5, alpha=0.5)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, 
                          alpha=0.9, edgecolors='black', linewidths=2.5)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', font_color='black')
    
    # Create legend
    legend_elements = []
    
    # Add server legend
    if any(n['type'] == 'server' for n in nodes.values()):
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=server_color, markersize=14, 
                                        label='Server', 
                                        markeredgecolor='black', markeredgewidth=2))
    
    # Add edge legend
    if any(n['type'] == 'edge' for n in nodes.values()):
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=edge_color, markersize=12, 
                                        label='Edge Server', 
                                        markeredgecolor='black', markeredgewidth=2))
    
    # Add cluster legends
    for cluster_name in sorted(legend_clusters):
        if cluster_name in cluster_color_map:
            label = f'Cluster: {cluster_name}' if cluster_name != 'orphan' else 'Clients'
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=cluster_color_map[cluster_name], 
                                            markersize=10, 
                                            label=label, 
                                            markeredgecolor='black', markeredgewidth=2))
    
    if legend_elements:
        plt.legend(handles=legend_elements, loc='upper right', fontsize=10, 
                  framealpha=0.95, edgecolor='black', fancybox=True, shadow=True)
    
    plt.title(title, fontsize=18, fontweight='bold', pad=25)
    plt.axis('off')
    plt.tight_layout()
    
    # Save the diagram
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    diagram_path = os.path.join(save_path, f"{diagram_name}.png")
    plt.savefig(diagram_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    fed_logger.info(f"Architecture diagram saved to {diagram_path}")
