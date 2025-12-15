import streamlit as st
import json
import os
from pathlib import Path
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from functools import lru_cache
from collections import defaultdict

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

@st.cache_data
def load_problems(folder_path):
    """Load problems from a folder's problems.json file."""
    problems_path = os.path.join(folder_path, 'problems', 'problems.json')
    if os.path.exists(problems_path):
        with open(problems_path, 'r') as f:
            data = json.load(f)
            return data.get('problems', [])
    return []

@st.cache_data
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

@st.cache_data
def load_funding_data(folder_path):
    """Load funding data from a folder's funding/funding_data.csv file.
    
    Returns:
        Dictionary mapping project names to dictionaries containing all CSV fields.
        Each value dict includes a 'total_funding' key for backward compatibility.
        Also includes a '_lookup' key with normalized lowercase keys for faster matching.
    """
    funding_path = os.path.join(folder_path, 'funding', 'funding_data.csv')
    if os.path.exists(funding_path):
        try:
            df = pd.read_csv(funding_path)
            # Create a mapping of project name to all funding data
            funding_map = {}
            
            # Vectorized operations for better performance
            project_names = df['Project Name'].values
            
            # Calculate total funding vectorized
            if 'Funding' in df.columns:
                total_funding = pd.to_numeric(df['Funding'], errors='coerce').fillna(0).values
            else:
                # Fallback to Matching + Donations for QF rounds
                matching_col = None
                donations_col = None
                for col in df.columns:
                    if 'Matching' in col and 'USDC' in col:
                        matching_col = col
                    elif 'Donations' in col and 'USD' in col:
                        donations_col = col
                
                matching = pd.to_numeric(df[matching_col], errors='coerce').fillna(0).values if matching_col else np.zeros(len(df))
                donations = pd.to_numeric(df[donations_col], errors='coerce').fillna(0).values if donations_col else np.zeros(len(df))
                total_funding = matching + donations
            
            # Process each row more efficiently
            for idx, project_name in enumerate(project_names):
                project_data = {}
                row = df.iloc[idx]
                
                # Store all available columns with optimized type conversion
                for col in df.columns:
                    if col == 'Project Name':
                        continue
                    value = row[col]
                    
                    if pd.notna(value):
                        # Optimize type conversion based on column name patterns
                        col_lower = col.lower()
                        if 'matching' in col_lower or 'donations' in col_lower or 'funding' in col_lower:
                            try:
                                project_data[col] = float(value)
                            except (ValueError, TypeError):
                                project_data[col] = value
                        elif 'unique donors' in col_lower or ('donors' in col_lower and 'unique' in col_lower):
                            try:
                                project_data[col] = int(value)
                            except (ValueError, TypeError):
                                project_data[col] = value
                        elif 'multiplier' in col_lower or 'per unique donor' in col_lower:
                            try:
                                # Remove $ sign if present
                                clean_value = str(value).replace('$', '').replace(',', '').strip()
                                project_data[col] = float(clean_value)
                            except (ValueError, TypeError):
                                project_data[col] = value
                        else:
                            project_data[col] = value
                    else:
                        project_data[col] = None
                
                # Add total funding
                project_data['total_funding'] = float(total_funding[idx])
                
                funding_map[project_name] = project_data
            
            return funding_map
        except Exception as e:
            return {}
    return {}

def get_funding_amount(fund_value):
    """Extract funding amount from funding data value.
    
    Handles both old format (number) and new format (dict with 'total_funding' key).
    """
    if isinstance(fund_value, dict):
        return fund_value.get('total_funding', 0.0)
    elif isinstance(fund_value, (int, float)):
        return float(fund_value)
    else:
        return 0.0

def find_project_funding(project_name, funding_data):
    """Find funding data for a project using optimized lookup.
    
    Uses multiple matching strategies for robustness:
    1. Exact match (case-sensitive)
    2. Case-insensitive exact match
    3. Partial substring match
    
    Returns tuple of (funding_amount, funding_data_dict).
    """
    if not funding_data:
        return (0.0, None)
    
    project_name_lower = project_name.lower()
    
    # Try exact match first (most common case, O(1))
    if project_name in funding_data:
        fund_value = funding_data[project_name]
        return (get_funding_amount(fund_value), fund_value if isinstance(fund_value, dict) else None)
    
    # Try case-insensitive exact match (O(n) but only if exact match fails)
    for funding_project_name, fund_value in funding_data.items():
        if project_name_lower == funding_project_name.lower():
            return (get_funding_amount(fund_value), fund_value if isinstance(fund_value, dict) else None)
    
    # Try partial match (fallback for name variations, O(n))
    for funding_project_name, fund_value in funding_data.items():
        funding_name_lower = funding_project_name.lower()
        if project_name_lower in funding_name_lower or funding_name_lower in project_name_lower:
            return (get_funding_amount(fund_value), fund_value if isinstance(fund_value, dict) else None)
    
    return (0.0, None)

@st.cache_data
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

def format_currency(value, currency="$"):
    """Format a currency value with the symbol before ($) or after (others like WETH).
    Uses whole numbers for chart text display.
    
    Args:
        value: The numeric value to format
        currency: The currency symbol (default: "$")
    
    Returns:
        Formatted string like "$100" or "7 WETH"
    """
    if currency == "$":
        return f"${value:,.0f}"
    else:
        return f"{value:,.0f} {currency}"

def format_currency_decimal(value, currency="$"):
    """Format a currency value with decimals, with symbol before ($) or after (others).
    
    Args:
        value: The numeric value to format
        currency: The currency symbol (default: "$")
    
    Returns:
        Formatted string like "$100.50" or "1.50 WETH"
    """
    if currency == "$":
        return f"${value:,.2f}"
    else:
        return f"{value:,.2f} {currency}"

def create_problem_scatter_plot(problems, alignment_results, funding_data=None, currency="$"):
    """Create a 2D scatter plot showing problems as bubbles.
    X-axis: Number of projects (Primary + Secondary)
    Y-axis: Total attributed funding
    Bubble size: Average donation per project
    """
    if not problems or not alignment_results:
        return None
    
    # Create mapping of problem_id to name (optimized)
    problem_map = {p['problem_id']: p['name'] for p in problems}
    
    # Count primary and secondary alignments per problem (using defaultdict for efficiency)
    primary_counts = defaultdict(int)
    secondary_counts = defaultdict(int)
    # Calculate attributed funding per problem (60% from primary, 40% from secondary)
    problem_funding = defaultdict(float)
    
    for result in alignment_results:
        project_name = result['project']
        
        # Get project funding if available
        project_fund, _ = find_project_funding(project_name, funding_data)
        
        # Get primary problem ID for this project
        primary_problem_id = None
        primary = result.get('primary_problem')
        if primary:
            primary_problem_id = primary.get('problem_id')
            if primary_problem_id in problem_map:
                score = primary.get('score', 0)
                if score > 0:
                    problem_name = problem_map[primary_problem_id]
                    primary_counts[problem_name] += 1
                    # Add 60% of project funding to primary problem
                    if project_fund > 0:
                        problem_funding[problem_name] += project_fund * 0.6
        
        # Process secondary problem
        # Only count if it's a different problem from the primary
        secondary = result.get('secondary_problem')
        if secondary:
            secondary_problem_id = secondary.get('problem_id')
            if secondary_problem_id in problem_map and secondary_problem_id != primary_problem_id:
                score = secondary.get('score', 0)
                if score > 0:
                    problem_name = problem_map[secondary_problem_id]
                    secondary_counts[problem_name] += 1
                    # Add 40% of project funding to secondary problem
                    if project_fund > 0:
                        problem_funding[problem_name] += project_fund * 0.4
    
    # Get all problems that have at least one alignment
    all_problems = set(primary_counts.keys()) | set(secondary_counts.keys())
    
    if not all_problems:
        return None
    
    # Prepare data for scatter plot
    scatter_data = []
    for problem_name in all_problems:
        primary_count = primary_counts.get(problem_name, 0)
        secondary_count = secondary_counts.get(problem_name, 0)
        total_projects = primary_count + secondary_count
        total_funding = problem_funding.get(problem_name, 0.0)
        
        # Calculate average donation per project
        # Avoid division by zero
        avg_donation = total_funding / total_projects if total_projects > 0 else 0.0
        
        scatter_data.append({
            'name': problem_name,
            'x': total_projects,
            'y': total_funding,  # Y-axis: Total attributed funding
            'size': avg_donation,  # Bubble size: Average donation per project
            'primary_count': primary_count,
            'secondary_count': secondary_count,
            'total_funding': total_funding,
            'avg_donation': avg_donation
        })
    
    # Filter out problems with no funding if funding data exists
    if funding_data:
        scatter_data = [d for d in scatter_data if d['y'] > 0]  # Filter by Y-axis (total funding)
    
    if not scatter_data:
        return None
    
    # Extract data for plotting
    problem_names = [d['name'] for d in scatter_data]
    x_values = [d['x'] for d in scatter_data]
    y_values = [d['y'] for d in scatter_data]  # Total attributed funding
    sizes = [d['size'] for d in scatter_data]  # Average donation per project
    primary_counts_list = [d['primary_count'] for d in scatter_data]
    secondary_counts_list = [d['secondary_count'] for d in scatter_data]
    total_funding_list = [d['total_funding'] for d in scatter_data]
    avg_donation_list = [d['avg_donation'] for d in scatter_data]
    
    # Create hover text
    currency_label = currency if currency != "$" else "$"
    hover_texts = []
    for i, name in enumerate(problem_names):
        hover_text = (
            f"<b>{name}</b><br>"
            f"Projects: {x_values[i]} (Primary: {primary_counts_list[i]}, Secondary: {secondary_counts_list[i]})<br>"
            f"Total Attributed Funding: {format_currency_decimal(y_values[i], currency)}<br>"
            f"Avg Donation/Project: {format_currency_decimal(avg_donation_list[i], currency)}"
        )
        hover_texts.append(hover_text)
    
    # Create color palette for problems
    # Use a distinct color palette that works well for categorical data
    color_palette = [
        '#9E93F0', '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', 
        '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B739',
        '#52BE80', '#E74C3C', '#3498DB', '#F39C12', '#9B59B6',
        '#1ABC9C', '#E67E22', '#34495E', '#16A085', '#27AE60'
    ]
    
    # Create a color mapping for each unique problem
    unique_problems = sorted(set(problem_names))
    problem_color_map = {prob: color_palette[i % len(color_palette)] 
                        for i, prob in enumerate(unique_problems)}
    
    # Calculate bubble sizes with square root scaling to make differences more visible
    # Using square root of average donation values exaggerates differences in bubble sizes
    if sizes and max(sizes) > 0:
        max_size = max(sizes)
        min_size = min(sizes) if min(sizes) > 0 else max_size * 0.01
        
        # Apply square root transformation to make size differences more pronounced
        sqrt_sizes = [np.sqrt(s) for s in sizes]
        sqrt_max = max(sqrt_sizes)
        sqrt_min = min(sqrt_sizes)
        
        # Scale square root values to pixel sizes (15 to 80 pixels)
        # This makes the visual differences much more apparent
        if sqrt_max > sqrt_min:
            scale_factor = (80 - 15) / (sqrt_max - sqrt_min)
            offset = 15
            # Convert to area values for Plotly (since sizemode='area')
            scaled_sizes = [((sqrt_val - sqrt_min) * scale_factor + offset) ** 2 
                           for sqrt_val in sqrt_sizes]
        else:
            # All sizes are the same
            scaled_sizes = [50 ** 2] * len(sizes)
        
        # Calculate sizeref: max_scaled_size / desired_max_area
        # We want max bubble to be around 80 pixels, so desired_max_area = 80^2
        max_scaled = max(scaled_sizes)
        sizeref = max_scaled / (80.0 ** 2)
    else:
        sizeref = 1.0
        scaled_sizes = [50 ** 2] * len(sizes) if sizes else []
    
    # Create scatter plot with separate traces for each problem (for legend)
    fig = go.Figure()
    
    # Group data by problem name and prepare text labels
    problem_data_groups = {}
    for i, prob_name in enumerate(problem_names):
        if prob_name not in problem_data_groups:
            problem_data_groups[prob_name] = {
                'x': [], 'y': [], 'sizes': [], 'hover_texts': [], 'text_labels': []
            }
        problem_data_groups[prob_name]['x'].append(x_values[i])
        problem_data_groups[prob_name]['y'].append(y_values[i])
        problem_data_groups[prob_name]['sizes'].append(scaled_sizes[i])
        problem_data_groups[prob_name]['hover_texts'].append(hover_texts[i])
        # Create text label: problem name and average funding
        avg_funding_text = format_currency_decimal(avg_donation_list[i], currency)
        text_label = f"{prob_name}<br>Avg Donation: {avg_funding_text}"
        problem_data_groups[prob_name]['text_labels'].append(text_label)
    
    # Add a trace for each problem
    for prob_name in unique_problems:
        if prob_name in problem_data_groups:
            data = problem_data_groups[prob_name]
            color = problem_color_map[prob_name]
            # Use a darker shade for the border
            border_color = color  # Keep same color or use a darker variant
            
            fig.add_trace(go.Scatter(
                x=data['x'],
                y=data['y'],
                mode='markers+text',
                marker=dict(
                    size=data['sizes'],
                    sizemode='area',
                    sizeref=sizeref,
                    sizemin=15,
                    color=color,
                    opacity=0.7,
                    line=dict(width=2, color=border_color)
                ),
                text=data['text_labels'],
                textposition='middle center',
                textfont=dict(
                    size=12,
                    color='black',
                    #family='Arial Black'
                ),
                hovertemplate='%{customdata}<extra></extra>',
                customdata=data['hover_texts'],
                name=prob_name,
                showlegend=True
            ))
    
    # Update layout
    fig.update_layout(
        xaxis_title="Number of Projects (Primary + Secondary)",
        yaxis_title=f"Total Attributed Funding ({currency_label})",
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.10,  # Puts legend above the plot
            xanchor="center",
            x=0.5,
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        ),
        hovermode='closest',
        margin=dict(t=120, b=60)  # Add top margin for legend, some bottom margin for labels
    )
    
    return fig

def create_alignment_table(problems, alignment_results, funding_data=None, currency="$"):
    """Create a table showing project alignment with funding data.
    
    Returns:
        pandas.DataFrame with columns: Project, Funding Allocated, Primary Problem Alignment, Secondary Problem Alignment,
        and additional fields from funding data if available (Unique donors, Match per unique donor, Match-to-Donation Multiplier)
    """
    if not problems or not alignment_results:
        return None
    
    # Create mapping of problem_id to name
    problem_map = {p['problem_id']: p['name'] for p in problems}
    
    # Build table data
    table_data = []
    
    for result in alignment_results:
        project_name = result['project']
        
        # Get project funding and additional fields if available
        project_fund, project_funding_data = find_project_funding(project_name, funding_data)
        
        # Get primary problem
        primary_problem_name = None
        primary_score = None
        if 'primary_problem' in result and result['primary_problem']:
            primary = result['primary_problem']
            problem_id = primary.get('problem_id')
            if problem_id in problem_map:
                score = primary.get('score', 0)
                if score > 0:
                    primary_problem_name = problem_map[problem_id]
                    primary_score = score
        
        # Get secondary problem (only if different from primary)
        secondary_problem_name = None
        secondary_score = None
        primary_problem_id = result.get('primary_problem', {}).get('problem_id') if result.get('primary_problem') else None
        if 'secondary_problem' in result and result['secondary_problem']:
            secondary = result['secondary_problem']
            problem_id = secondary.get('problem_id')
            if problem_id in problem_map:
                score = secondary.get('score', 0)
                if score > 0:
                    # Only include if different from primary
                    if problem_id != primary_problem_id:
                        secondary_problem_name = problem_map[problem_id]
                        secondary_score = score
        
        # Format alignment strings
        #primary_alignment = f"{primary_problem_name} (Score: {primary_score})" if primary_problem_name else "None"
        #secondary_alignment = f"{secondary_problem_name} (Score: {secondary_score})" if secondary_problem_name else "None"
        primary_alignment = f"{primary_problem_name}" if primary_problem_name else "None"
        secondary_alignment = f"{secondary_problem_name}" if secondary_problem_name else "None"

        # Build row data in the desired order:
        # Project, Funding Allocated, Matching (USDC), Donations (USD), then other additional fields (if available), then alignments
        row_data = {
            'Project': project_name,
            'Funding Allocated': project_fund
        }
        
        # Add additional fields from funding data if available (in specific order)
        matching_usdc = None
        donations_usd = None
        unique_donors = None
        match_per_donor = None
        multiplier = None
        
        if project_funding_data:
            # Extract additional fields with optimized column name matching
            # Check exact matches first (most common case, O(1) lookup)
            if 'Matching (USDC)' in project_funding_data:
                matching_usdc = project_funding_data['Matching (USDC)']
            if 'Donations (USD)' in project_funding_data:
                donations_usd = project_funding_data['Donations (USD)']
            
            # Then check flexible matches using lowercase lookup (only for fields not yet found)
            for col_name, value in project_funding_data.items():
                if col_name == 'total_funding':
                    continue
                col_lower = col_name.lower()
                
                if matching_usdc is None and 'matching' in col_lower and 'usdc' in col_lower:
                    matching_usdc = value
                elif donations_usd is None and 'donations' in col_lower and 'usd' in col_lower:
                    donations_usd = value
                elif unique_donors is None and 'unique' in col_lower and 'donors' in col_lower:
                    unique_donors = value
                elif match_per_donor is None and 'match' in col_lower and 'per unique donor' in col_lower:
                    match_per_donor = value
                elif multiplier is None and 'multiplier' in col_lower and 'match' in col_lower and 'donation' in col_lower:
                    multiplier = value
        
        # Add additional fields in order (only if they have values)
        # Format dollar amounts with $ prefix
        if matching_usdc is not None:
            # Format as dollar amount
            if isinstance(matching_usdc, (int, float)):
                row_data['Matching (USDC)'] = format_currency_decimal(matching_usdc, "$")
            else:
                row_data['Matching (USDC)'] = matching_usdc
        if donations_usd is not None:
            # Format as dollar amount
            if isinstance(donations_usd, (int, float)):
                row_data['Donations (USD)'] = format_currency_decimal(donations_usd, "$")
            else:
                row_data['Donations (USD)'] = donations_usd
        if unique_donors is not None:
            row_data['Unique donors'] = unique_donors
        if match_per_donor is not None:
            # Format as dollar amount
            if isinstance(match_per_donor, (int, float)):
                row_data['Match per unique donor'] = format_currency_decimal(match_per_donor, "$")
            else:
                row_data['Match per unique donor'] = match_per_donor
        if multiplier is not None:
            row_data['Match-to-Donation Multiplier'] = multiplier
        
        # Add alignment fields at the end
        row_data['Primary Problem Alignment'] = primary_alignment
        row_data['Secondary Problem Alignment'] = secondary_alignment

        table_data.append(row_data)
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Define the desired column order
    base_columns = ['Project', 'Funding Allocated']
    funding_detail_columns = ['Matching (USDC)', 'Donations (USD)']
    additional_columns = ['Unique donors', 'Match per unique donor', 'Match-to-Donation Multiplier']
    alignment_columns = ['Primary Problem Alignment', 'Secondary Problem Alignment']
    
    # Build ordered column list: include only columns that exist in the DataFrame (optimized)
    all_desired_columns = base_columns + funding_detail_columns + additional_columns + alignment_columns
    ordered_columns = [col for col in all_desired_columns if col in df.columns]
    
    # Add any remaining columns that weren't in the desired order
    remaining_cols = [col for col in df.columns if col not in ordered_columns]
    ordered_columns.extend(remaining_cols)
    
    # Reorder DataFrame columns
    df = df[ordered_columns]
    
    # Sort by funding allocated (descending), then by project name (optimized with inplace=False)
    df = df.sort_values(['Funding Allocated', 'Project'], ascending=[False, True], kind='mergesort')
    
    return df

def create_funding_sankey(problems, alignment_results, funding_data, currency="$"):
    """Create a Sankey diagram showing funding flow: Total Funding → Problems → Projects."""
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
    # Track flows from problems to projects: (problem_name, project_name) -> (amount, alignment_type)
    problem_to_project_flows = {}  # {(problem_name, project_name): (amount, 'primary'|'secondary')}
    
    # Process each alignment result
    for result in alignment_results:
        project_name = result['project']
        
        # Try to find funding for this project (handle name variations)
        project_fund, _ = find_project_funding(project_name, funding_data)
        
        if project_fund == 0:
            print(f"[warning] Funding data missing for project: {project_name}")
            continue  # Skip projects without funding data
        
        # Add project if not already added
        if project_name not in project_names:
            project_names.append(project_name)
            project_funding[project_name] = project_fund
        
        # Get primary problem ID to check for duplicates
        primary_problem_id = None
        if 'primary_problem' in result and result['primary_problem']:
            primary = result['primary_problem']
            primary_problem_id = primary.get('problem_id')
        
        # Process primary problem (60% of funding)
        if primary_problem_id and primary_problem_id in problem_map:
            problem_name = problem_map[primary_problem_id]
            primary_fund = project_fund * 0.6
            problem_funding[problem_name] += primary_fund
            # Track flow from problem to project with alignment type
            key = (problem_name, project_name)
            problem_to_project_flows[key] = (primary_fund, 'primary')
        
        # Process secondary problem (40% of funding)
        # Only count if it's a different problem from the primary
        if 'secondary_problem' in result and result['secondary_problem']:
            secondary = result['secondary_problem']
            secondary_problem_id = secondary.get('problem_id')
            if secondary_problem_id in problem_map and secondary_problem_id != primary_problem_id:
                problem_name = problem_map[secondary_problem_id]
                secondary_fund = project_fund * 0.4
                problem_funding[problem_name] += secondary_fund
                # Track flow from problem to project with alignment type
                key = (problem_name, project_name)
                problem_to_project_flows[key] = (secondary_fund, 'secondary')
    
    # Filter out problems and projects with no funding
    problem_names = [p for p in problem_names if problem_funding.get(p, 0) > 0]
    project_names = [p for p in project_names if project_funding.get(p, 0) > 0]
    
    if not problem_names or not project_names:
        return None
    
    # Calculate total funding
    total_funding = sum(project_funding.values())
    
    if total_funding == 0:
        return None
    
    # Create index mappings for three levels:
    # Level 1 (left): Total Funding (index 0)
    # Level 2 (middle): Problems (indices 1 to len(problem_names))
    # Level 3 (right): Projects (indices len(problem_names)+1 to len(problem_names)+len(project_names))
    
    TOTAL_FUNDING_IDX = 0
    problem_idx_map = {name: idx + 1 for idx, name in enumerate(problem_names)}
    project_idx_map = {name: idx + len(problem_names) + 1 for idx, name in enumerate(project_names)}
    
    # Build labels: Total Funding, then Problems, then Projects
    labels = ["Total Funding"]
    labels.extend(problem_names)
    labels.extend(project_names)
    
    # Build flows
    source_indices = []
    target_indices = []
    values = []
    hover_texts = []
    link_colors = []
    
    # Level 1 → Level 2: Total Funding → Problems
    for problem_name in problem_names:
        problem_fund = problem_funding.get(problem_name, 0)
        if problem_fund > 0:
            source_indices.append(TOTAL_FUNDING_IDX)
            target_indices.append(problem_idx_map[problem_name])
            values.append(problem_fund)
            link_colors.append('#E3F6FF')  # Blue for total to problem flows
            percentage = (problem_fund / total_funding * 100) if total_funding > 0 else 0
            hover_texts.append(
                f"<b>Total Funding</b><br>" +
                f"→ <b>{problem_name}</b><br>" +
                f"Amount: {format_currency_decimal(problem_fund, currency)}<br>" +
                f"({percentage:.1f}% of total)"
            )
    
    # Level 2 → Level 3: Problems → Projects
    for (problem_name, project_name), (amount, alignment_type) in problem_to_project_flows.items():
        if problem_name in problem_idx_map and project_name in project_idx_map and amount > 0:
            source_indices.append(problem_idx_map[problem_name])
            target_indices.append(project_idx_map[project_name])
            values.append(amount)
            # Set color based on alignment type
            if alignment_type == 'primary':
                link_colors.append('#E8D9FF')  # Primary alignment color
                alignment_label = "Primary"
            else:
                link_colors.append('#FFF0D3')  # Secondary alignment color
                alignment_label = "Secondary"
            
            hover_texts.append(
                f"<b>{problem_name}</b><br>" +
                f"→ <b>{project_name}</b> ({alignment_label})<br>" +
                f"Amount: {format_currency_decimal(amount, currency)}"
            )
    
    if not values:
        return None
    
    # Calculate node funding totals for hover display
    node_funding_info = []
    # Total Funding (level 1)
    node_funding_info.append(total_funding)
    # Problems (level 2) - total funding they receive
    for problem_name in problem_names:
        node_funding_info.append(problem_funding.get(problem_name, 0))
    # Projects (level 3) - their total funding
    for project_name in project_names:
        node_funding_info.append(project_funding.get(project_name, 0))
    
    # Create Sankey diagram with three levels
    # Level 1 (left): Total Funding
    # Level 2 (middle): Problems
    # Level 3 (right): Projects
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=30,
            line=dict(color="black", width=0.5),
            label=labels,
            # Color: Total Funding (green), Problems (blue), Projects (teal)
            color=['#00b990' if i == 0 else '#004988' if i <= len(problem_names) else '#00b990'
                   for i in range(len(labels))],
            customdata=node_funding_info,
            hovertemplate=f'<b>%{{label}}</b><br>Total Funding: {"$%{customdata:,.2f}" if currency == "$" else f"%{{customdata:,.2f}} {currency}"}<extra></extra>',
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

    # Add "legend" entries for primary and secondary attributions as invisible scatter traces
    # (These don't affect the Sankey visualization but provide a legend)
    fig.add_trace(go.Scatter(
        x=[0], y=[1.1], 
        mode='markers',
        marker=dict(size=15, color='#E8D9FF'),
        showlegend=True,
        name='Primary Alignment',
        legendgroup="attrib"
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=15, color='#FFF0D3'),
        showlegend=True,
        name='Secondary Alignment',
        legendgroup="attrib"
    ))

    
    fig.update_layout(
        font=dict(
            size=14,
            color='#000000'
        ),
        height=1200 if len(project_names) > 50 else 800,
        width=1200,
        xaxis_showticklabels=False,
        yaxis_showticklabels=False,
        legend=dict(
            orientation='h',
            x=0.5,
            y=0,
            xanchor='center',
            yanchor='bottom'
        )
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

@st.cache_data
def load_ethereum_taxonomy(base_dir):
    """Load Ethereum Problem Space Taxonomy CSV.
    
    Returns:
        DataFrame with columns: Ethereum Level 1 Problem, Ethereum Level 2 Problem, etc.
    """
    taxonomy_path = base_dir / 'gg24 problem mapping' / 'Ethereum Problem Space Taxonomy.csv'
    if taxonomy_path.exists():
        try:
            df = pd.read_csv(taxonomy_path)
            return df
        except Exception as e:
            st.error(f"Error loading taxonomy CSV: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

@st.cache_data
def load_ethereum_mapping(base_dir):
    """Load Ethereum x GG24 Mapping CSV.
    
    Returns:
        DataFrame with columns: Round, Problem, Mapped Ethereum Problem (Level 1), etc.
    """
    mapping_path = base_dir / 'gg24 problem mapping' / 'Ethereum x GG24 Mapping.csv'
    if mapping_path.exists():
        try:
            df = pd.read_csv(mapping_path)
            return df
        except Exception as e:
            st.error(f"Error loading mapping CSV: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def create_ethereum_sankey(taxonomy_df, mapping_df, selected_categories=None, selected_rounds=None):
    """Create a 4-step Sankey diagram mapping:
    Category (Level 1) -> Ecosystem Challenge (Level 2) -> Round Problem -> Round
    
    Args:
        taxonomy_df: DataFrame with Ethereum taxonomy data
        mapping_df: DataFrame with Ethereum x GG24 mapping data
        selected_categories: List of category names to filter by (None = show all)
        selected_rounds: List of round names to filter by (None = show all)
    
    Returns:
        Plotly figure or None if data is insufficient
    """
    if taxonomy_df.empty or mapping_df.empty:
        return None
    
    # Get all unique categories (Level 1) and ecosystem challenges (Level 2) from taxonomy
    # This ensures all are shown even if not mapped
    all_categories = sorted(taxonomy_df['Ethereum Level 1 Problem'].dropna().unique())
    all_ecosystem_challenges = sorted(taxonomy_df['Ethereum Level 2 Problem'].dropna().unique())
    
    # Get unique round problems and rounds from mapping
    # Only include rows where Problem and Round are not null
    mapping_df_clean = mapping_df.dropna(subset=['Problem', 'Round'])
    unique_round_problems = sorted(mapping_df_clean['Problem'].unique())
    unique_rounds = sorted(mapping_df_clean['Round'].unique())
    
    if not all_categories or not all_ecosystem_challenges:
        return None
    
    # Track which ecosystem challenges are mapped to round problems
    mapped_challenges = set()
    challenge_to_round_problem = defaultdict(int)
    for _, row in mapping_df_clean.iterrows():
        challenge = row.get('Mapped Ethereum Problem (Level 2)')
        round_problem = row.get('Problem')
        if pd.notna(challenge) and pd.notna(round_problem):
            mapped_challenges.add(challenge)
            challenge_to_round_problem[(challenge, round_problem)] += 1
    
    # Add "Unmapped" node to round problems if there are unmapped ecosystem challenges
    unmapped_challenges = set(all_ecosystem_challenges) - mapped_challenges
    if unmapped_challenges:
        # Add "Unmapped" as a special round problem node
        if "Unmapped" not in unique_round_problems:
            unique_round_problems.append("Unmapped")
            unique_round_problems = sorted(unique_round_problems)
    
    # Apply filters: determine which nodes should be active based on selections
    # Initialize to empty sets when filters are applied, only populate based on actual connections
    active_categories = set()
    active_rounds = set()
    active_ecosystem_challenges = set()
    active_round_problems = set()
    
    # If no filters, all nodes are active
    if not selected_categories and not selected_rounds:
        active_categories = set(all_categories)
        active_rounds = set(unique_rounds)
        active_ecosystem_challenges = set(all_ecosystem_challenges)
        active_round_problems = set(unique_round_problems)
    else:
        # If rounds are filtered, find which round problems and ecosystem challenges connect to them
        if selected_rounds:
            active_rounds = set(selected_rounds)
            for _, row in mapping_df_clean.iterrows():
                round_name = row.get('Round')
                round_problem = row.get('Problem')
                challenge = row.get('Mapped Ethereum Problem (Level 2)')
                category = row.get('Mapped Ethereum Problem (Level 1)')
                
                if round_name in active_rounds:
                    if pd.notna(round_problem):
                        active_round_problems.add(round_problem)
                    if pd.notna(challenge):
                        active_ecosystem_challenges.add(challenge)
                    if pd.notna(category):
                        active_categories.add(category)
        
        # If categories are filtered, find which ecosystem challenges belong to them
        if selected_categories:
            active_categories = set(selected_categories)
            for _, row in taxonomy_df.iterrows():
                category = row.get('Ethereum Level 1 Problem')
                challenge = row.get('Ethereum Level 2 Problem')
                if category in active_categories and pd.notna(challenge):
                    active_ecosystem_challenges.add(challenge)
            
            # Then find which round problems connect to these challenges
            for _, row in mapping_df_clean.iterrows():
                challenge = row.get('Mapped Ethereum Problem (Level 2)')
                round_problem = row.get('Problem')
                round_name = row.get('Round')
                if challenge in active_ecosystem_challenges:
                    if pd.notna(round_problem):
                        active_round_problems.add(round_problem)
                    if pd.notna(round_name):
                        active_rounds.add(round_name)
    
    # Build index mappings for 4 levels:
    # Level 1: Categories (indices 0 to len(all_categories)-1)
    # Level 2: Ecosystem Challenges (indices len(all_categories) to len(all_categories)+len(all_ecosystem_challenges)-1)
    # Level 3: Round Problems (indices len(all_categories)+len(all_ecosystem_challenges) to ...)
    # Level 4: Rounds (indices ... to ...)
    
    category_idx_map = {cat: idx for idx, cat in enumerate(all_categories)}
    eco_challenge_start = len(all_categories)
    eco_challenge_idx_map = {challenge: idx + eco_challenge_start 
                            for idx, challenge in enumerate(all_ecosystem_challenges)}
    round_problem_start = eco_challenge_start + len(all_ecosystem_challenges)
    round_problem_idx_map = {prob: idx + round_problem_start 
                            for idx, prob in enumerate(unique_round_problems)}
    round_start = round_problem_start + len(unique_round_problems)
    round_idx_map = {round_name: idx + round_start 
                    for idx, round_name in enumerate(unique_rounds)}
    
    # Build labels list
    labels = list(all_categories) + list(all_ecosystem_challenges) + list(unique_round_problems) + list(unique_rounds)
    
    # Build flows
    source_indices = []
    target_indices = []
    values = []
    hover_texts = []
    
    # Step 1: Category -> Ecosystem Challenge (from taxonomy)
    # Count occurrences for flow width
    category_to_challenge = defaultdict(int)
    for _, row in taxonomy_df.iterrows():
        category = row.get('Ethereum Level 1 Problem')
        challenge = row.get('Ethereum Level 2 Problem')
        if pd.notna(category) and pd.notna(challenge):
            category_to_challenge[(category, challenge)] += 1
    
    link_colors = []  # Track link colors for highlighting
    for (category, challenge), count in category_to_challenge.items():
        if category in category_idx_map and challenge in eco_challenge_idx_map:
            source_indices.append(category_idx_map[category])
            target_indices.append(eco_challenge_idx_map[challenge])
            values.append(count)
            hover_texts.append(
                f"<b>{category}</b><br>→ <b>{challenge}</b>"
            )
            # Highlight if both nodes are active
            if category in active_categories and challenge in active_ecosystem_challenges:
                link_colors.append('rgba(30, 144, 255, 0.3)')  # Bright blue for active
            else:
                link_colors.append('rgba(200, 200, 200, 0.2)')  # Dim gray for inactive
    
    # Step 2: Ecosystem Challenge -> Round Problem (from mapping)
    # Use the challenge_to_round_problem dictionary already built above
    for (challenge, round_problem), count in challenge_to_round_problem.items():
        if challenge in eco_challenge_idx_map and round_problem in round_problem_idx_map:
            source_indices.append(eco_challenge_idx_map[challenge])
            target_indices.append(round_problem_idx_map[round_problem])
            values.append(count)
            hover_texts.append(
                f"<b>{challenge}</b><br>→ <b>{round_problem}</b>"
            )
            # Highlight if both nodes are active
            if challenge in active_ecosystem_challenges and round_problem in active_round_problems:
                link_colors.append('rgba(30, 144, 255, 0.3)')  # Bright blue for active
            else:
                link_colors.append('rgba(200, 200, 200, 0.2)')  # Dim gray for inactive
    
    # Add connections for unmapped ecosystem challenges to "Unmapped" node
    # This ensures all Level 2 nodes are positioned on the same line
    if unmapped_challenges and "Unmapped" in round_problem_idx_map:
        for challenge in unmapped_challenges:
            if challenge in eco_challenge_idx_map:
                source_indices.append(eco_challenge_idx_map[challenge])
                target_indices.append(round_problem_idx_map["Unmapped"])
                values.append(0.1)  # Small value to make it visible but not prominent
                hover_texts.append(
                    f"<b>{challenge}</b><br>→ <b>Unmapped</b><br>(No Round Problem mapping)"
                )
                # Highlight if challenge is active
                if challenge in active_ecosystem_challenges:
                    link_colors.append('rgba(200, 200, 200, 0.4)')  # Slightly brighter for active unmapped
                else:
                    link_colors.append('rgba(200, 200, 200, 0.1)')  # Very dim for inactive unmapped
    
    # Step 3: Round Problem -> Round (from mapping)
    round_problem_to_round = defaultdict(int)
    for _, row in mapping_df_clean.iterrows():
        round_problem = row.get('Problem')
        round_name = row.get('Round')
        if pd.notna(round_problem) and pd.notna(round_name):
            round_problem_to_round[(round_problem, round_name)] += 1
    
    for (round_problem, round_name), count in round_problem_to_round.items():
        if round_problem in round_problem_idx_map and round_name in round_idx_map:
            source_indices.append(round_problem_idx_map[round_problem])
            target_indices.append(round_idx_map[round_name])
            values.append(count)
            hover_texts.append(
                f"<b>{round_problem}</b><br>→ <b>{round_name}</b>"
            )
            # Highlight if both nodes are active
            if round_problem in active_round_problems and round_name in active_rounds:
                link_colors.append('rgba(30, 144, 255, 0.3)')  # Bright blue for active
            else:
                link_colors.append('rgba(200, 200, 200, 0.2)')  # Dim gray for inactive
    
    if not values:
        return None
    
    # Create color scheme for different levels with opacity based on active status
    # Categories: blue, Ecosystem Challenges: teal, Round Problems: purple, Rounds: red
    node_colors = []
    for i in range(len(labels)):
        if i < len(all_categories):
            category = all_categories[i]
            if category in active_categories:
                node_colors.append('#004988')  # Blue for active categories
            else:
                node_colors.append('rgba(0, 73, 136, 0.3)')  # Dimmed blue for inactive
        elif i < len(all_categories) + len(all_ecosystem_challenges):
            challenge_idx = i - len(all_categories)
            challenge = all_ecosystem_challenges[challenge_idx]
            if challenge in active_ecosystem_challenges:
                node_colors.append('#00b990')  # Teal for active ecosystem challenges
            else:
                node_colors.append('rgba(0, 185, 144, 0.3)')  # Dimmed teal for inactive
        elif i < len(all_categories) + len(all_ecosystem_challenges) + len(unique_round_problems):
            # Check if this is the "Unmapped" node
            node_idx = i - (len(all_categories) + len(all_ecosystem_challenges))
            round_problem = unique_round_problems[node_idx]
            if round_problem == "Unmapped":
                node_colors.append('#CCCCCC')  # Light gray for unmapped
            elif round_problem in active_round_problems:
                node_colors.append('#9E93F0')  # Purple for active round problems
            else:
                node_colors.append('rgba(158, 147, 240, 0.3)')  # Dimmed purple for inactive
        else:
            round_idx = i - (len(all_categories) + len(all_ecosystem_challenges) + len(unique_round_problems))
            round_name = unique_rounds[round_idx]
            if round_name in active_rounds:
                node_colors.append('#FF6B6B')  # Red for active rounds
            else:
                node_colors.append('rgba(255, 107, 107, 0.3)')  # Dimmed red for inactive
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=30,
            line=dict(color="black", width=0.5),
            label=labels,
            color=node_colors,
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=values,
            hovertemplate='%{customdata}<extra></extra>',
            customdata=hover_texts,
            color=link_colors  # Use dynamic colors based on filter state
        )
    )])
    
    fig.update_layout(
        font=dict(size=12, color='#000000'),
        height=max(800, len(labels) * 20),  # Dynamic height based on number of nodes
        width=1400,
        title_text="Ethereum Problem Space Mapping: Category → Ecosystem Challenge → Round Problem → Round",
        title_x=0.5,
        title_font_size=16
    )
    
    return fig

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
    
    # CTA before tabs
    st.info("""
    **📊 Analysis Scope:** This is a WIP analysis covering GG24 rounds that have completed allocations. 
    
    **💬 We Welcome Your Feedback:** If you are a round operator, a domain SME, or a project owner, we welcome your feedback on:
    - What additional questions should we investigate
    - Inputs on how problems have been defined for each round
    - Mapping of projects to problems

    📄 [Feedback Form](https://forms.gle/3kDCzjMrUZA7RDoG6)
    """)
    
    # Add the Ethereum mapping tab name
    tab_names.append("Ethereum Problem Mapping")
    
    # Create tabs for each folder plus the new mapping tab
    tabs = st.tabs(tab_names)
    
    # Display folder tabs
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
            
            # Extract currency from round metadata
            currency = "$"  # Default to dollar
            if round_metadata and 'fields' in round_metadata:
                currency = round_metadata['fields'].get('Funding_Currency', '$')
            
            if not problems:
                st.warning(f"No problems found in {folder}/problems/problems.json")
                continue
            
            # Display bar chart and heatmap if alignment results exist
            if alignment_results:
                # Display alignment table
                st.header("Project Alignment & Funding Summary")
                st.markdown("**Explore how projects align with funding priorities.** This table shows each funded project alongside its allocated amount and problem alignments. Projects are sorted by funding amount (highest first) to help you quickly identify the most heavily funded initiatives. The **Primary Problem Alignment** indicates the main problem area each project addresses, while **Secondary Problem Alignment** shows additional problem areas where the project may contribute.")
                
                alignment_df = create_alignment_table(problems, alignment_results, funding_data, currency)
                if alignment_df is not None and not alignment_df.empty:
                    # Find max funding value for ProgressColumn
                    max_funding = alignment_df['Funding Allocated'].max()
                    
                    # Format string for currency
                    if currency == "$":
                        format_str = "$%d"
                    else:
                        format_str = f"%.2f {currency}"
                    
                    # Configure column display
                    column_config = {
                        "Project": st.column_config.TextColumn(
                            "Project",
                            width="medium"
                        ),
                        "Funding Allocated": st.column_config.ProgressColumn(
                            "Funding Allocated",
                            min_value=0,
                            max_value=max_funding if max_funding > 0 else 1,
                            format=format_str
                        ),
                        "Primary Problem Alignment": st.column_config.TextColumn(
                            "Primary Problem Alignment",
                            width="large"
                        ),
                        "Secondary Problem Alignment": st.column_config.TextColumn(
                            "Secondary Problem Alignment",
                            width="large"
                        )
                    }
                    
                    st.dataframe(
                        alignment_df,
                        column_config=column_config,
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.warning("No alignment data available to display.")

                # Display scatter plot if funding data exists
                if funding_data:
                    st.header("Problem Funding Distribution")
                    st.markdown("**Compare project engagement with funding allocation across problem areas.** Each bubble represents a problem area, positioned by the number of participating projects (horizontal axis) and total attributed funding (vertical axis). The bubble size indicates the average funding per project—larger bubbles mean higher average funding. **Key insights:** Problems in the upper-right have both high engagement and high funding; problems in the upper-left receive significant funding with fewer projects; problems in the lower-right have many projects but lower total funding.")
                    scatter_fig = create_problem_scatter_plot(problems, alignment_results, funding_data, currency)
                    if scatter_fig:
                        st.plotly_chart(scatter_fig, use_container_width=True)
                
                
                
                
                # Display Sankey diagram if funding data exists
                if funding_data:
                    st.header("Funding Flow: From Problems to Projects")
                    st.markdown("**Visualize how funding flows through problem areas to individual projects.** This Sankey diagram shows the complete funding pathway: total funding first distributes across problem areas (middle column), then flows to specific projects (right column) based on their alignments. **How to read:** The width of each flow represents the funding amount. Primary alignments (light purple) receive 60% of a project's funding, while secondary alignments (light yellow) receive 40%. Use this to identify which problem areas attract the most funding and how projects receive funding through their problem alignments.")
                    
                    # Add problem filter with "All" option
                    problem_names = [p['name'] for p in problems]
                    ALL_OPTION = "All"
                    
                    # Create options list with "All" at the beginning
                    filter_options = [ALL_OPTION] + problem_names
                    
                    # Get default selection (all problems + "All")
                    default_selection = [ALL_OPTION] 
                    
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
                    
                    sankey_fig = create_funding_sankey(filtered_problems, alignment_results, funding_data, currency)
                    if sankey_fig:
                        st.plotly_chart(sankey_fig, use_container_width=True)
                    else:
                        st.warning("Unable to create Sankey diagram. Check that project names in alignment results match funding data.")
                    
            
            st.header("Problem Definitions")
            st.markdown("""
            The problem definitions below are grounded in the domain’s sensemaking research and the round’s eligibility guidance. They describe the systemic gaps the domain seeks to address, framed at the right level of abstraction to remain stable across rounds while still allowing clear classification of diverse projects.
            """)
            
            # Create an expander for each problem
            for problem in problems:
                problem_name = problem.get('name', f"Problem {problem.get('problem_id', 'Unknown')}")
                with st.expander(f"🔍 {problem_name}", expanded=False):
                    display_problem(problem)
    
    # Display Ethereum Problem Mapping tab (last tab)
    with tabs[len(folders)]:
        st.title("Ethereum Problem Space Mapping")
        st.markdown("""
        This visualization maps round-specific problems to the broader Ethereum ecosystem challenges. 
        The Sankey diagram shows the flow from:
        - **Category** (Ethereum Level 1 Problem): High-level problem categories
        - **Ecosystem Challenge** (Ethereum Level 2 Problem): Specific challenges within each category
        - **Round Problem**: Problems defined in GG24 rounds
        - **Round**: The specific GG24 round
        
        All Categories and Ecosystem Challenges are shown, even if they don't have mappings to Round Problems.
        
        **💡 Interactive Filters:** Use the filters below to drill down and explore specific categories or rounds. 
        Selected paths will be highlighted in bright blue, while inactive paths are dimmed.
        """)
        
        # Load CSV data
        taxonomy_df = load_ethereum_taxonomy(base_dir)
        mapping_df = load_ethereum_mapping(base_dir)
        
        if taxonomy_df.empty:
            st.warning("Ethereum Problem Space Taxonomy CSV not found or could not be loaded.")
        elif mapping_df.empty:
            st.warning("Ethereum x GG24 Mapping CSV not found or could not be loaded.")
        else:
            # Get unique values for filters
            all_categories = sorted(taxonomy_df['Ethereum Level 1 Problem'].dropna().unique())
            mapping_df_clean = mapping_df.dropna(subset=['Problem', 'Round'])
            all_rounds = sorted(mapping_df_clean['Round'].unique())
            
            # Create filter section
            st.header("🔍 Interactive Filters")
            col1, col2 = st.columns(2)
            
            with col1:
                selected_categories = st.multiselect(
                    "Filter by Category (Ethereum Level 1)",
                    options=all_categories,
                    default=[],
                    help="Select one or more categories to see which rounds address those needs. Leave empty to show all."
                )
            
            with col2:
                selected_rounds = st.multiselect(
                    "Filter by Round",
                    options=all_rounds,
                    default=[],
                    help="Select one or more rounds to see where they fit in the Ethereum ecosystem. Leave empty to show all."
                )
            
            # Show summary of what's selected
            if selected_categories or selected_rounds:
                st.info(f"""
                **Active Filters:**
                - **Categories:** {', '.join(selected_categories) if selected_categories else 'All'}
                - **Rounds:** {', '.join(selected_rounds) if selected_rounds else 'All'}
                
                *Highlighted paths show connections for your selections. Dimmed paths show other connections.*
                """)
            
            # Create and display Sankey diagram with filters
            sankey_fig = create_ethereum_sankey(
                taxonomy_df, 
                mapping_df, 
                selected_categories=selected_categories if selected_categories else None,
                selected_rounds=selected_rounds if selected_rounds else None
            )
            if sankey_fig:
                st.plotly_chart(sankey_fig, use_container_width=True)
            else:
                st.warning("Unable to create Sankey diagram. Please check the CSV data format.")

if __name__ == "__main__":
    main()
