import streamlit as st
import json
import os
from pathlib import Path
import plotly.graph_objects as go
import numpy as np
import pandas as pd

# Set page config
st.set_page_config(
    page_title="GG24 Alignment Dashboard: Mapping Problems, Projects, and Funding",
    page_icon="📋",
    layout="wide"
)

def parse_config(config_path):
    """Parse config.txt to extract folder names."""
    folders = []
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and ':' in line:
                    folder_name = line.split(':')[0]
                    folders.append(folder_name)
    return folders

def load_problems(folder_path):
    """Load problems from a folder's problems.json file."""
    problems_path = os.path.join(folder_path, 'problems', 'problems.json')
    if os.path.exists(problems_path):
        with open(problems_path, 'r') as f:
            data = json.load(f)
            return data.get('problems', [])
    return []

def load_alignment_results(folder_path):
    """Load alignment results from a folder's alignment_results.json file."""
    results_path = os.path.join(folder_path, 'alignment_results.json')
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            data = json.load(f)
            return data.get('results', [])
    return []

def format_measurement_rubric(rubric):
    """Format measurement rubric as a user-friendly display."""
    if not rubric:
        return "No rubric available"
    
    formatted = []
    for score, description in sorted(rubric.items(), key=lambda x: int(x[0])):
        formatted.append(f"**Score {score}:** {description}")
    return "\n\n".join(formatted)

def format_positive_signals(signals):
    """Format positive signals as a bullet list."""
    if not signals:
        return "No signals specified"
    return "\n".join([f"• {signal}" for signal in signals])

def load_funding_data(folder_path):
    """Load funding data from a folder's funding/funding_data.csv file."""
    funding_path = os.path.join(folder_path, 'funding', 'funding_data.csv')
    if os.path.exists(funding_path):
        try:
            df = pd.read_csv(funding_path)
            # Create a mapping of project name to total funding (Matching + Donations)
            funding_map = {}
            for _, row in df.iterrows():
                project_name = row['Project Name']
                matching = float(row['Matching (USDC)']) if pd.notna(row['Matching (USDC)']) else 0
                donations = float(row['Donations (USD)']) if pd.notna(row['Donations (USD)']) else 0
                funding_map[project_name] = matching + donations
            return funding_map
        except Exception as e:
            return {}
    return {}

def load_round_metadata(folder_path):
    """Load round metadata from a folder's round/round metadata.json file."""
    metadata_path = os.path.join(folder_path, 'round', 'round metadata.json')
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            return None
    return None

def create_alignment_bar_chart(problems, alignment_results):
    """Create a stacked bar chart showing primary and secondary alignment projects per problem."""
    if not problems or not alignment_results:
        return None
    
    # Create mapping of problem_id to name
    problem_map = {p['problem_id']: p['name'] for p in problems}
    
    # Count primary and secondary alignments per problem
    primary_counts = {}
    secondary_counts = {}
    
    for result in alignment_results:
        # Get primary problem ID for this project
        primary_problem_id = None
        if 'primary_problem' in result and result['primary_problem']:
            primary = result['primary_problem']
            primary_problem_id = primary.get('problem_id')
            if primary_problem_id in problem_map:
                score = primary.get('score', 0)
                if score > 0:
                    problem_name = problem_map[primary_problem_id]
                    primary_counts[problem_name] = primary_counts.get(problem_name, 0) + 1
        
        # Process secondary problem
        # Only count if it's a different problem from the primary
        if 'secondary_problem' in result and result['secondary_problem']:
            secondary = result['secondary_problem']
            secondary_problem_id = secondary.get('problem_id')
            if secondary_problem_id in problem_map:
                score = secondary.get('score', 0)
                if score > 0:
                    # Only count secondary if it's a different problem from primary
                    if secondary_problem_id != primary_problem_id:
                        problem_name = problem_map[secondary_problem_id]
                        secondary_counts[problem_name] = secondary_counts.get(problem_name, 0) + 1
    
    # Get all problems that have at least one alignment
    all_problems = set(primary_counts.keys()) | set(secondary_counts.keys())
    
    if not all_problems:
        return None
    
    # Prepare data for sorting
    problem_data = []
    for problem_name in all_problems:
        primary_count = primary_counts.get(problem_name, 0)
        secondary_count = secondary_counts.get(problem_name, 0)
        total = primary_count + secondary_count
        problem_data.append({
            'name': problem_name,
            'primary': primary_count,
            'secondary': secondary_count,
            'total': total
        })
    
    # Sort by total in descending order
    problem_data.sort(key=lambda x: x['total'], reverse=True)
    
    # Extract sorted lists
    problem_names = [p['name'] for p in problem_data]
    primary_values = [p['primary'] for p in problem_data]
    secondary_values = [p['secondary'] for p in problem_data]
    
    # Create stacked bar chart
    fig = go.Figure()
    
    # Add primary bar (bottom)
    fig.add_trace(go.Bar(
        x=problem_names,
        y=primary_values,
        name='Primary Alignment',
        marker_color='#9E93F0',
        hovertemplate='<b>%{x}</b><br>Primary: %{y}<extra></extra>'
    ))
    
    # Add secondary bar (stacked on top)
    fig.add_trace(go.Bar(
        x=problem_names,
        y=secondary_values,
        name='Secondary Alignment',
        marker_color='#B5DAEF',
        hovertemplate='<b>%{x}</b><br>Secondary: %{y}<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        #title="Projects per Problem (Primary and Secondary Alignment)",
        xaxis_title="Problems",
        yaxis_title="Number of Projects",
        barmode='stack',
        height=800,
        xaxis=dict(tickangle=-45),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_alignment_heatmap(problems, alignment_results):
    """Create a heatmap showing project-problem alignment scores."""
    if not problems or not alignment_results:
        return None
    
    # Create mapping of problem_id to index and name
    problem_map = {p['problem_id']: (idx, p['name']) for idx, p in enumerate(problems)}
    
    # Get all unique projects
    projects = sorted(set([r['project'] for r in alignment_results]))
    project_map = {p: idx for idx, p in enumerate(projects)}
    
    # Initialize matrices
    score_matrix = np.zeros((len(projects), len(problems)), dtype=float)
    primary_matrix = np.zeros((len(projects), len(problems)), dtype=bool)
    hover_texts = []
    
    # Fill matrices with alignment data
    for result in alignment_results:
        project = result['project']
        project_idx = project_map[project]
        
        # Process primary problem
        if 'primary_problem' in result and result['primary_problem']:
            primary = result['primary_problem']
            problem_id = primary.get('problem_id')
            if problem_id in problem_map:
                problem_idx, _ = problem_map[problem_id]
                score = primary.get('score', 0)
                if score > 0:
                    score_matrix[project_idx, problem_idx] = score
                    primary_matrix[project_idx, problem_idx] = True
        
        # Process secondary problem (typically a different problem from primary)
        if 'secondary_problem' in result and result['secondary_problem']:
            secondary = result['secondary_problem']
            problem_id = secondary.get('problem_id')
            if problem_id in problem_map:
                problem_idx, _ = problem_map[problem_id]
                score = secondary.get('score', 0)
                if score > 0:
                    # If this problem already has a primary alignment, keep primary
                    # Otherwise, set as secondary
                    if score_matrix[project_idx, problem_idx] == 0:
                        score_matrix[project_idx, problem_idx] = score
                        primary_matrix[project_idx, problem_idx] = False
                    # If primary exists for this problem, don't overwrite
    
    # Calculate total projects per problem (primary + secondary)
    problem_totals = {}
    for problem_idx, problem in enumerate(problems):
        primary_count = np.sum((primary_matrix[:, problem_idx] == True) & (score_matrix[:, problem_idx] > 0))
        secondary_count = np.sum((primary_matrix[:, problem_idx] == False) & (score_matrix[:, problem_idx] > 0))
        total = primary_count + secondary_count
        problem_totals[problem_idx] = total
    
    # Sort problems by total projects (descending)
    problem_sort_indices = sorted(range(len(problems)), key=lambda i: problem_totals[i], reverse=True)
    
    # Sort projects alphabetically
    project_sort_indices = sorted(range(len(projects)), key=lambda i: projects[i])
    
    # Reorder matrices and data structures based on sorting
    score_matrix = score_matrix[np.ix_(project_sort_indices, problem_sort_indices)]
    primary_matrix = primary_matrix[np.ix_(project_sort_indices, problem_sort_indices)]
    sorted_projects = [projects[i] for i in project_sort_indices]
    sorted_problems = [problems[i] for i in problem_sort_indices]
    
    # Recreate hover text matrix with sorted order
    hover_texts = []
    for project_idx, project in enumerate(sorted_projects):
        row_texts = []
        for problem_idx, problem in enumerate(sorted_problems):
            score = score_matrix[project_idx, problem_idx]
            is_primary = primary_matrix[project_idx, problem_idx]
            alignment_type = "Primary" if is_primary else "Secondary" if score > 0 else "None"
            
            if score > 0:
                hover_text = (
                    f"<b>Project:</b> {project}<br>"
                    f"<b>Problem:</b> {problem['name']}<br>"
                    f"<b>Alignment Score:</b> {score}<br>"
                    f"<b>Type:</b> {alignment_type}"
                )
            else:
                hover_text = (
                    f"<b>Project:</b> {project}<br>"
                    f"<b>Problem:</b> {problem['name']}<br>"
                    f"<b>Alignment Score:</b> No alignment"
                )
            row_texts.append(hover_text)
        hover_texts.append(row_texts)
    
    # Define discrete color palette for whole number scores 0-5
    # Colors: 5=#1e504a, 4=#42715c, 3=#8db883, 2=#b4dc96, 1=#ccf3a2, 0=#ccf3a2
    # Scores map to positions: 0->0.0, 1->0.2, 2->0.4, 3->0.6, 4->0.8, 5->1.0
    discrete_colorscale = [
        [0.0, '#e6e6e6'],    # Score 0
        [0.2, '#ccf3a2'],   # Score 1 (up to 0.2)
        [0.21, '#b4dc96'],   # Score 2
        [0.4, '#b4dc96'],   # Score 2 (up to 0.4)
        [0.41, '#8db883'],   # Score 3
        [0.6, '#8db883'],   # Score 3 (up to 0.6)
        [0.61, '#42715c'],   # Score 4
        [0.8, '#42715c'],   # Score 4 (up to 0.8)
        [0.81, '#1e504a'],   # Score 5
        [1.0, '#1e504a']     # Score 5
    ]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=score_matrix,
        x=[p['name'] for p in sorted_problems],
        y=sorted_projects,
        colorscale=discrete_colorscale,
        zmin=0,
        zmax=5,
        text=hover_texts,
        texttemplate='',
        hovertemplate='%{text}<extra></extra>',
        colorbar=dict(
            title="Alignment Score",
            tickmode='array',
            tickvals=[0, 1, 2, 3, 4, 5],
            ticktext=['0', '1', '2', '3', '4', '5'],
            tickformat='d'  # Display as integers
        ),
        showscale=True
    ))
    
    # Add annotations for primary/secondary indicators
    annotations = []
    problem_names = [p['name'] for p in sorted_problems]
    
    for project_idx, project in enumerate(sorted_projects):
        for problem_idx, problem in enumerate(sorted_problems):
            score = score_matrix[project_idx, problem_idx]
            if score > 0:
                is_primary = primary_matrix[project_idx, problem_idx]
                # Add star icon for primary (larger), circle for secondary
                symbol = "★" if is_primary else "●"
                # Use larger font for primary to make it stand out
                font_size = 18 if is_primary else 14
                # Use specific colors for primary and secondary symbols
                text_color = "#9E93F0" if is_primary else "#B5DAEF"
                
                annotations.append(
                    dict(
                        x=problem_names[problem_idx],
                        y=project,
                        text=symbol,
                        showarrow=False,
                        font=dict(
                            size=font_size, 
                            color=text_color,
                            family="Arial Black"
                        ),
                        xref='x',
                        yref='y'
                    )
                )
    
    fig.update_layout(
        #title="Project-Problem Alignment Heatmap",
        xaxis_title="Problems",
        yaxis_title="Projects",
        width=1200,
        height=max(400, len(sorted_projects) * 45),
        annotations=annotations,
        xaxis=dict(tickangle=-45, side='bottom'),
        yaxis=dict(autorange='reversed')
    )
    
    return fig

def create_funding_sankey(problems, alignment_results, funding_data):
    """Create a Sankey diagram showing funding flow from projects to problems."""
    if not problems or not alignment_results or not funding_data:
        return None
    
    # Create mapping of problem_id to name
    problem_map = {p['problem_id']: p['name'] for p in problems}
    
    # Collect all unique problems and projects
    problem_names = list(problem_map.values())
    project_names = []
    
    # Calculate funding flows and totals
    problem_funding = {name: 0 for name in problem_names}
    project_funding = {}
    
    # Process each alignment result
    for result in alignment_results:
        project_name = result['project']
        
        # Try to find funding for this project (handle name variations)
        project_fund = 0
        for funding_project_name, fund_value in funding_data.items():
            # Try exact match first
            if project_name.lower() == funding_project_name.lower():
                project_fund = fund_value
                break
            # Try partial match (project name might be in funding name or vice versa)
            if project_name.lower() in funding_project_name.lower() or funding_project_name.lower() in project_name.lower():
                project_fund = fund_value
                break
        
        if project_fund == 0:
            continue  # Skip projects without funding data
        
        # Add project if not already added
        if project_name not in project_names:
            project_names.append(project_name)
            project_funding[project_name] = project_fund
        
        # Process primary problem (60% of funding)
        if 'primary_problem' in result and result['primary_problem']:
            primary = result['primary_problem']
            problem_id = primary.get('problem_id')
            if problem_id in problem_map:
                problem_name = problem_map[problem_id]
                primary_fund = project_fund * 0.6
                problem_funding[problem_name] += primary_fund
        
        # Process secondary problem (40% of funding)
        if 'secondary_problem' in result and result['secondary_problem']:
            secondary = result['secondary_problem']
            problem_id = secondary.get('problem_id')
            if problem_id in problem_map:
                problem_name = problem_map[problem_id]
                secondary_fund = project_fund * 0.4
                problem_funding[problem_name] += secondary_fund
    
    # Filter out problems and projects with no funding
    problem_names = [p for p in problem_names if problem_funding.get(p, 0) > 0]
    project_names = [p for p in project_names if project_funding.get(p, 0) > 0]
    
    if not problem_names or not project_names:
        return None
    
    # Create index mappings
    # Projects on left (sources), Problems on right (targets)
    # Funding flows from projects to problems: 60% to primary, 40% to secondary
    
    # Put projects first (left side as sources)
    project_idx_map = {name: idx for idx, name in enumerate(project_names)}
    # Put problems second (right side as targets)
    problem_idx_map = {name: idx + len(project_names) for idx, name in enumerate(problem_names)}
    
    # Build flows: from projects (sources/left) to problems (targets/right)
    source_indices = []
    target_indices = []
    values = []
    labels = []
    hover_texts = []
    flow_types = []  # Track if flow is 'primary' or 'secondary' for coloring
    
    # Add project nodes first (will appear on left as sources)
    for project_name in project_names:
        labels.append(project_name)
    
    # Add problem nodes (will appear on right as targets)
    for problem_name in problem_names:
        labels.append(problem_name)
    
    # Create flows from projects (sources/left) to problems (targets/right)
    # The flow value represents funding allocated from project to problem
    for result in alignment_results:
        project_name = result['project']
        
        if project_name not in project_idx_map:
            continue
        
        project_fund = project_funding.get(project_name, 0)
        if project_fund == 0:
            continue
        
        source_idx = project_idx_map[project_name]
        
        # Primary problem flow (60% of project funding)
        if 'primary_problem' in result and result['primary_problem']:
            primary = result['primary_problem']
            problem_id = primary.get('problem_id')
            if problem_id in problem_map:
                problem_name = problem_map[problem_id]
                if problem_name in problem_idx_map:
                    target_idx = problem_idx_map[problem_name]
                    value = project_fund * 0.6
                    source_indices.append(source_idx)
                    target_indices.append(target_idx)
                    values.append(value)
                    flow_types.append('primary')
                    hover_texts.append(
                        f"<b>{project_name}</b><br>" +
                        f"→ <b>{problem_name}</b> (Primary)<br>" +
                        f"Amount: ${value:,.2f}<br>" +
                        f"(60% of ${project_fund:,.2f})"
                    )
        
        # Secondary problem flow (40% of project funding)
        if 'secondary_problem' in result and result['secondary_problem']:
            secondary = result['secondary_problem']
            problem_id = secondary.get('problem_id')
            if problem_id in problem_map:
                problem_name = problem_map[problem_id]
                if problem_name in problem_idx_map:
                    target_idx = problem_idx_map[problem_name]
                    value = project_fund * 0.4
                    source_indices.append(source_idx)
                    target_indices.append(target_idx)
                    values.append(value)
                    flow_types.append('secondary')
                    hover_texts.append(
                        f"<b>{project_name}</b><br>" +
                        f"→ <b>{problem_name}</b> (Secondary)<br>" +
                        f"Amount: ${value:,.2f}<br>" +
                        f"(40% of ${project_fund:,.2f})"
                    )
    
    if not values:
        return None
    
    # Calculate node funding totals for hover display
    # Note: Plotly Sankey automatically sizes nodes based on flow values,
    # so we can't explicitly set node sizes, but we can show funding info in hover
    node_funding_info = []
    # Projects first (left side) - their total funding
    for project_name in project_names:
        node_funding_info.append(project_funding.get(project_name, 0))
    # Problems second (right side) - total funding they receive
    for problem_name in problem_names:
        node_funding_info.append(problem_funding.get(problem_name, 0))
    
    # Create color arrays for links (primary vs secondary)
    link_colors = []
    for flow_type in flow_types:
        if flow_type == 'primary':
            link_colors.append('#9E93F0')  # Dark blue for primary
        else:
            link_colors.append('#B5DAEF')  # Teal for secondary
    
    # Create Sankey diagram
    # Projects on left (sources), Problems on right (targets)
    # Flow represents funding allocation: 60% to primary problem, 40% to secondary problem
    # Node sizes are automatically calculated from flow values
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=30,  # Increased thickness to provide more space for text
            line=dict(color="black", width=0.5),
            label=labels,
            color=['#00b990' if i < len(project_names) else '#004988' 
                   for i in range(len(labels))],
            customdata=node_funding_info,
            hovertemplate='<b>%{label}</b><br>Total Funding: $%{customdata:,.2f}<extra></extra>',
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=values,
            hovertemplate='%{customdata}<extra></extra>',
            customdata=hover_texts,
            color=link_colors
        )
    )])
    
    fig.update_layout(
        #title="Funding Flow: Projects (left) → Problems (right) | 60% Primary, 40% Secondary",
        font=dict(
            size=12,  # Increased font size for better legibility
            color='#000000'  # Black text for better contrast against light backgrounds
        ),
        height=800,
        width=1200
    )
    
    return fig

def display_problem(problem):
    """Display a single problem in a user-friendly format."""
    st.markdown(f"### {problem.get('name', 'Unnamed Problem')}")
    
    # Problem ID and Domain
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Problem ID:** `{problem.get('problem_id', 'N/A')}`")
    with col2:
        st.markdown(f"**Domain:** {problem.get('domain', 'N/A')}")
    
    st.divider()
    
    # Problem Statement
    st.markdown("#### Problem Statement")
    st.markdown(problem.get('problem_statement', 'No statement provided'))
    
    # Why It Matters
    st.markdown("#### Why It Matters")
    st.markdown(problem.get('why_it_matters', 'No information provided'))
    
    # Solution Shape
    st.markdown("#### Solution Shape")
    st.markdown(problem.get('solution_shape', 'No solution shape provided'))
    
    # Positive Signals
    st.markdown("#### Positive Signals")
    signals = problem.get('positive_signals', [])
    st.markdown(format_positive_signals(signals))
    
    # Measurement Rubric
    st.markdown("#### Measurement Rubric")
    rubric = problem.get('measurement_rubric', {})
    st.markdown(format_measurement_rubric(rubric))

def main():
    st.title("📋 GG24 Alignment Dashboard: Mapping Problems, Projects, and Funding")
    st.markdown("---")
    st.markdown("""
    GG24 is Gitcoin’s largest experiment in problem-first public goods funding, \
    where each domain defines the ecosystem-level gaps it exists to solve. \
    This dashboard brings those problems, the projects addressing them, and \
    the distribution of capital into one place. \
    It helps round operators, donors, and ecosystem stewards see how funding decisions \
    align with the strategic priorities of the Ethereum community.
    """)
    
    # Get the base directory (where config.txt is located)
    base_dir = Path(__file__).parent
    config_path = base_dir / 'config.txt'
    
    # Parse config to get folder names
    folders = parse_config(config_path)
    
    if not folders:
        st.warning("No folders found in config.txt")
        return
    
    # Load round metadata for each folder to get tab names
    tab_names = []
    round_metadata_list = []
    for folder in folders:
        folder_path = base_dir / folder
        metadata = load_round_metadata(folder_path)
        if metadata and 'round_name' in metadata:
            tab_names.append(metadata['round_name'])
        else:
            # Fallback to folder name if metadata not found
            tab_names.append(folder.replace('-', ' ').title())
        round_metadata_list.append(metadata)
    
    # Create tabs for each folder
    tabs = st.tabs(tab_names)
    
    for idx, folder in enumerate(folders):
        with tabs[idx]:
            folder_path = base_dir / folder
            
            if not os.path.exists(folder_path):
                st.error(f"Folder '{folder}' does not exist")
                continue
            
            # Load round metadata and display title/description
            round_metadata = round_metadata_list[idx]
            if round_metadata:
                # Display round name as title
                st.title(round_metadata.get('round_name', folder.replace('-', ' ').title()))
                
                # Display description
                if 'description' in round_metadata:
                    st.markdown(round_metadata['description'])
                
                # Display fields using st.metric in one row
                if 'fields' in round_metadata and round_metadata['fields']:
                    fields = round_metadata['fields']
                    col1, col2, col3 = st.columns(3)
                    
                    # Display fields in order: mechanism, funding_pool, number_of_grantees
                    field_keys = ['mechanism', 'funding_pool', 'number_of_grantees']
                    field_labels = ['Mechanism', 'Funding Pool', 'Number of Grantees']
                    
                    for i, (key, label) in enumerate(zip(field_keys, field_labels)):
                        if key in fields:
                            with [col1, col2, col3][i]:
                                st.metric(label=label, value=fields[key], border=True)
                
                
            
            # Load problems, alignment results, and funding data
            problems = load_problems(folder_path)
            alignment_results = load_alignment_results(folder_path)
            funding_data = load_funding_data(folder_path)
            
            if not problems:
                st.warning(f"No problems found in {folder}/problems/problems.json")
                continue
            
            # Display bar chart and heatmap if alignment results exist
            if alignment_results:
                # Display stacked bar chart
                st.header("Problem Coverage by Participating Projects")
                st.markdown("Use this chart to understand project coverage across problems and identify potential blind spots or over-concentration within the round.")
                bar_chart_fig = create_alignment_bar_chart(problems, alignment_results)
                if bar_chart_fig:
                    st.plotly_chart(bar_chart_fig, use_container_width=True)
                
                # Display heatmap
                st.header("Problem Alignment Overview")
                st.markdown("This heatmap provides a granular view of project alignment across the problems. It helps surface clusters, outliers, and potential gaps in coverage across the round. Stars indicate primary alignment, circles indicate secondary alignment, and the color intensity reflects the strength of alignment.")
                st.markdown("**Legend:** ★ = Primary alignment, ● = Secondary alignment")
                
                heatmap_fig = create_alignment_heatmap(problems, alignment_results)
                if heatmap_fig:
                    st.plotly_chart(heatmap_fig, use_container_width=True)
                
                
                # Display Sankey diagram if funding data exists
                if funding_data:
                    st.header("Capital Flow by Problem Alignment")
                    st.markdown("Use this view to assess how the mechanism channels funding toward different problem areas within the round. It reveals concentration, gaps, and the impact of project alignment on the final distribution.")
                    st.markdown("**Funding allocation:** 60% to primary problem, 40% to secondary problem")
                    
                    # Add problem filter with "All" option
                    problem_names = [p['name'] for p in problems]
                    ALL_OPTION = "All"
                    
                    # Create options list with "All" at the beginning
                    filter_options = [ALL_OPTION] + problem_names
                    
                    # Get default selection (all problems + "All")
                    default_selection = [ALL_OPTION] + problem_names
                    
                    # Get current selection from multiselect
                    selected = st.multiselect(
                        "Filter by Problems",
                        options=filter_options,
                        default=default_selection,
                        key=f"problem_filter_{idx}"
                    )
                    
                    # Handle "All" option logic
                    if ALL_OPTION in selected:
                        # If "All" is selected, use all problems
                        selected_problems = problem_names
                    else:
                        # Use only the selected problems (excluding "All" if it was there)
                        selected_problems = [p for p in selected if p != ALL_OPTION]
                        
                        # If all individual problems are selected, treat as "All"
                        if set(selected_problems) == set(problem_names):
                            selected_problems = problem_names
                    
                    # Filter problems based on selection
                    filtered_problems = [p for p in problems if p['name'] in selected_problems] if selected_problems else problems
                    
                    sankey_fig = create_funding_sankey(filtered_problems, alignment_results, funding_data)
                    if sankey_fig:
                        st.plotly_chart(sankey_fig, use_container_width=True)
                    else:
                        st.warning("Unable to create Sankey diagram. Check that project names in alignment results match funding data.")
                    
            
            st.header("Problem Definitions")
            st.markdown("""
            The problem definitions below are grounded in the domain’s sensemaking research and the round’s eligibility guidance. They describe the systemic gaps the domain seeks to address, framed at the right level of abstraction to remain stable across rounds while still allowing clear classification of diverse projects.
            """)
            #st.markdown(f"Found **{len(problems)}** problem(s)")
            #st.markdown("---")
            
            # Create an expander for each problem
            for problem in problems:
                problem_name = problem.get('name', f"Problem {problem.get('problem_id', 'Unknown')}")
                with st.expander(f"🔍 {problem_name}", expanded=False):
                    display_problem(problem)

if __name__ == "__main__":
    main()
