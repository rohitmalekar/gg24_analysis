#!/usr/bin/env python3
"""
Alignment Evaluation Script

This script:
1. Reads config.txt to find folders with flag set to TRUE
2. For each folder, matches projects with problems using LLM
3. Assigns scores based on measurement rubrics
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from openai import OpenAI
import streamlit as st

# Initialize OpenAI client (set OPENAI_API_KEY environment variable)
#if not os.getenv('OPENAI_API_KEY'):
#    print("Warning: OPENAI_API_KEY environment variable not set. LLM calls will fail.", file=sys.stderr)

#OPENAI_API_KEY = st.secrets["api"]["OPENAI_API_KEY"]

def get_api_key():
    # Prefer env var in production, fall back to st.secrets locally
    if "OPENAI_API_KEY" in os.environ:
        return os.environ["OPENAI_API_KEY"]
    return st.secrets["api"]["OPENAI_API_KEY"]

OPENAI_API_KEY = get_api_key()    

client = OpenAI(api_key=OPENAI_API_KEY)


def read_config(config_path: str) -> Dict[str, bool]:
    """Read config.txt and return dict of folder_name -> enabled flag."""
    config = {}
    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            folder_name, flag = line.split(':', 1)
            config[folder_name.strip()] = flag.strip().upper() == 'TRUE'
    return config


def load_problems(problems_path: str) -> List[Dict]:
    """Load problems from problems.json."""
    with open(problems_path, 'r') as f:
        data = json.load(f)
    return data.get('problems', [])


def read_project_file(project_path: str) -> str:
    """Read project description from file."""
    with open(project_path, 'r', encoding='utf-8') as f:
        return f.read()


def get_project_files(projects_dir: str) -> List[str]:
    """Get all .txt project files from projects directory."""
    projects_path = Path(projects_dir)
    if not projects_path.exists():
        return []
    return [str(p) for p in projects_path.glob('*.txt') if p.name != 'round info.txt']


def match_problems_with_llm(project_content: str, problems: List[Dict]) -> Tuple[Optional[str], Optional[str]]:
    """
    Use LLM to find the most aligned primary and secondary problem for a project.
    Returns (primary_problem_id, secondary_problem_id)
    """
    # Format problems for the prompt
    problems_summary = []
    for prob in problems:
        problems_summary.append({
            'problem_id': prob['problem_id'],
            'name': prob['name'],
            'problem_statement': prob['problem_statement'],
            'why_it_matters': prob['why_it_matters'],
            'solution_shape': prob['solution_shape'],
            'positive_signals': prob['positive_signals']
        })
    
    prompt = f"""You are an expert evaluator in the Ethereum ecosystem evaluating project alignment with problems.

Given a project description and a list of problems, identify:
1. The PRIMARY problem (most aligned) - the problem this project most directly addresses
2. The SECONDARY problem (second most aligned) - another problem this project meaningfully addresses

Project Description:
{project_content}

Available Problems:
{json.dumps(problems_summary, indent=2)}

Respond with ONLY a JSON object in this exact format:
{{
    "primary_problem_id": "problem_id_here",
    "secondary_problem_id": "problem_id_here",
    "reasoning": "Brief explanation of why these problems were chosen"
}}

If no problem is clearly aligned, use null for that field."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a precise evaluator that outputs only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        primary_id = result.get('primary_problem_id')
        secondary_id = result.get('secondary_problem_id')
        
        return (primary_id, secondary_id)
    except Exception as e:
        print(f"Error in LLM matching: {e}", file=sys.stderr)
        return (None, None)


def score_project_with_rubric(project_content: str, problem: Dict) -> Optional[int]:
    """
    Use LLM to assign a score (1-5) to a project based on the problem's measurement rubric.
    """
    rubric = problem.get('measurement_rubric', {})
    
    prompt = f"""You are an expert evaluator in the Ethereum ecosystem scoring projects against a measurement rubric.

Project Description:
{project_content}

Problem Context:
- Name: {problem['name']}
- Problem Statement: {problem['problem_statement']}
- Solution Shape: {problem['solution_shape']}
- Positive Signals: {', '.join(problem['positive_signals'])}

Measurement Rubric (1-5 scale):
{json.dumps(rubric, indent=2)}

Assign a score from 1 to 5 based on how well this project addresses the problem according to the rubric.
Respond with ONLY a JSON object:
{{
    "score": <integer 1-5>,
    "reasoning": "Brief explanation of the score"
}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a precise evaluator that outputs only valid JSON with integer scores."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        score = result.get('score')
        
        # Validate score is in range
        if isinstance(score, int) and 1 <= score <= 5:
            return score
        else:
            print(f"Warning: Invalid score {score} returned, using None", file=sys.stderr)
            return None
    except Exception as e:
        print(f"Error in scoring: {e}", file=sys.stderr)
        return None


def evaluate_folder(folder_name: str, base_path: str) -> List[Dict]:
    """
    Evaluate all projects in a folder.
    Returns list of evaluation results.
    """
    folder_path = Path(base_path) / folder_name
    projects_dir = folder_path / 'projects'
    problems_path = folder_path / 'problems' / 'problems.json'
    
    if not problems_path.exists():
        print(f"Warning: {problems_path} not found, skipping folder {folder_name}", file=sys.stderr)
        return []
    
    # Load problems
    problems = load_problems(str(problems_path))
    problems_dict = {p['problem_id']: p for p in problems}
    
    # Get project files
    project_files = get_project_files(str(projects_dir))
    
    if not project_files:
        print(f"Warning: No project files found in {projects_dir}", file=sys.stderr)
        return []
    
    results = []
    
    for project_file in project_files:
        project_name = Path(project_file).stem
        print(f"Processing: {folder_name}/{project_name}...", file=sys.stderr)
        
        # Read project content
        project_content = read_project_file(project_file)
        
        # Match with problems
        primary_id, secondary_id = match_problems_with_llm(project_content, problems)
        
        result = {
            'folder': folder_name,
            'project': project_name,
            'project_file': Path(project_file).name,
            'primary_problem': None,
            'secondary_problem': None,
            'primary_score': None,
            'secondary_score': None
        }
        
        # Score primary problem
        if primary_id and primary_id in problems_dict:
            problem = problems_dict[primary_id]
            score = score_project_with_rubric(project_content, problem)
            result['primary_problem'] = {
                'problem_id': primary_id,
                'name': problem['name'],
                'score': score
            }
            result['primary_score'] = score
        
        # Score secondary problem
        if secondary_id and secondary_id in problems_dict:
            problem = problems_dict[secondary_id]
            score = score_project_with_rubric(project_content, problem)
            result['secondary_problem'] = {
                'problem_id': secondary_id,
                'name': problem['name'],
                'score': score
            }
            result['secondary_score'] = score
        
        results.append(result)
    
    return results


def main():
    """Main execution function."""
    base_path = Path(__file__).parent
    config_path = base_path / 'config.txt'
    
    if not config_path.exists():
        print(f"Error: {config_path} not found", file=sys.stderr)
        sys.exit(1)
    
    # Read config
    config = read_config(str(config_path))
    
    # Get enabled folders
    enabled_folders = [folder for folder, enabled in config.items() if enabled]
    
    if not enabled_folders:
        print("No folders enabled in config.txt", file=sys.stderr)
        sys.exit(0)
    
    print(f"Processing {len(enabled_folders)} enabled folder(s): {', '.join(enabled_folders)}", file=sys.stderr)
    
    all_results = []
    
    # Process each enabled folder
    for folder_name in enabled_folders:
        print(f"\n=== Processing folder: {folder_name} ===", file=sys.stderr)
        results = evaluate_folder(folder_name, str(base_path))
        all_results.extend(results)
        
        # Save results to each folder
        folder_path = base_path / folder_name
        output_file = folder_path / 'alignment_results.json'
        folder_output = {
            'summary': {
                'total_projects': len(results),
                'folder': folder_name
            },
            'results': results
        }
        with open(output_file, 'w') as f:
            json.dump(folder_output, f, indent=2)
        print(f"Results saved to {output_file}", file=sys.stderr)
    
    # Output combined results as JSON to stdout
    output = {
        'summary': {
            'total_projects': len(all_results),
            'folders_processed': enabled_folders
        },
        'results': all_results
    }
    
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()

