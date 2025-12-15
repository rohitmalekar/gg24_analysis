# GG24 Alignment Analysis

A comprehensive analysis tool for Gitcoin Grants Round 24 (GG24) that evaluates project alignment with ecosystem problems, visualizes funding distributions, and maps problems across the Ethereum ecosystem.

## Overview

GG24 is Gitcoin's largest experiment in problem-first public goods funding, where each domain defines the ecosystem-level gaps it exists to solve. This tool brings together:

- **Problem definitions** - Ecosystem-level gaps identified by each domain
- **Project alignments** - How projects address these problems (evaluated using LLM)
- **Funding data** - Capital distribution across projects and problems
- **Visualizations** - Interactive dashboards showing alignment and funding flows

## Features

### 1. Project-Problem Alignment Evaluation
- Uses LLM (GPT-4o-mini) to match projects with primary and secondary problems
- Scores projects (1-5 scale) based on measurement rubrics
- Handles multiple rounds/domains simultaneously

### 2. Interactive Dashboard
- **Project Alignment Table** - Shows all projects with their funding and problem alignments
- **Problem Funding Distribution** - Scatter plot comparing project engagement vs. funding allocation
- **Funding Flow Sankey Diagram** - Visualizes how funding flows from problems to projects
- **Problem Definitions** - Detailed view of each problem's statement, solution shape, and measurement rubric
- **Ethereum Problem Space Mapping** - Maps round-specific problems to broader Ethereum ecosystem challenges

### 3. Multi-Round Support
- Configure which rounds to analyze via `config.txt`
- Each round can have its own problems, projects, and funding data
- Round metadata includes mechanism type, funding pool, and number of grantees

## Project Structure

```
gg24_analysis/
├── alignment_eval.py          # LLM-based alignment evaluation script
├── display_rounds.py           # Streamlit dashboard application
├── config.txt                  # Configuration for enabled rounds
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container configuration
├── fly.toml                    # Fly.io deployment config
│
├── [round-name]/               # Each round/domain folder
│   ├── problems/
│   │   └── problems.json      # Problem definitions
│   ├── projects/               # Project description files (.txt)
│   ├── funding/
│   │   └── funding_data.csv   # Funding allocation data
│   ├── round/
│   │   ├── round metadata.json # Round metadata (name, description, fields)
│   │   └── round info.txt     # Additional round information
│   └── alignment_results.json # Generated alignment evaluation results
│
└── gg24 problem mapping/       # Ethereum ecosystem mapping
    ├── Ethereum Problem Space Taxonomy.csv
    └── Ethereum x GG24 Mapping.csv
```

## Setup

### Prerequisites

- Python 3.12+
- OpenAI API key
- (Optional) Docker for containerized deployment

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd gg24_analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up OpenAI API key:
   - Option 1: Environment variable
     ```bash
     export OPENAI_API_KEY="your-api-key-here"
     ```
   - Option 2: Streamlit secrets (for local Streamlit deployment)
     Create `.streamlit/secrets.toml`:
     ```toml
     [api]
     OPENAI_API_KEY = "your-api-key-here"
     ```

4. Configure rounds in `config.txt`:
```
dev-tooling-qf:TRUE
interop:TRUE
privacy:FALSE
```

## Usage

### Running the Alignment Evaluation

Evaluate project alignments for all enabled rounds:

```bash
python alignment_eval.py
```

This script:
1. Reads `config.txt` to find enabled rounds
2. For each round, matches projects with problems using LLM
3. Assigns scores based on measurement rubrics
4. Saves results to `[round-name]/alignment_results.json`

### Running the Dashboard

Start the Streamlit dashboard:

```bash
streamlit run display_rounds.py
```

The dashboard will be available at `http://localhost:8501`

### Dashboard Features

1. **Round Tabs** - Each enabled round gets its own tab
2. **Project Alignment Table** - Sortable table with funding and alignments
3. **Problem Scatter Plot** - Compare project count vs. funding by problem
4. **Funding Sankey Diagram** - Interactive flow visualization with problem filtering
5. **Problem Definitions** - Expandable cards with full problem details
6. **Ethereum Mapping** - Cross-round problem taxonomy visualization

## Data Format

### Problems JSON (`problems/problems.json`)

```json
{
  "problems": [
    {
      "problem_id": "devtool_01",
      "domain": "Developer Tooling",
      "name": "Underfunded Core Infrastructure",
      "level": "problem_type",
      "problem_statement": "...",
      "why_it_matters": "...",
      "solution_shape": "...",
      "positive_signals": ["signal1", "signal2", ...],
      "measurement_rubric": {
        "1": "Weak alignment description",
        "2": "...",
        "3": "...",
        "4": "...",
        "5": "Strong alignment description"
      }
    }
  ]
}
```

### Round Metadata (`round/round metadata.json`)

```json
{
  "round_name": "Developer Tooling QF",
  "description": "Round description...",
  "fields": {
    "mechanism": "Quadratic Funding",
    "funding_pool": "$500,000",
    "number_of_grantees": 48,
    "Funding_Currency": "$"
  }
}
```

### Funding Data (`funding/funding_data.csv`)

CSV with columns:
- `Project Name` (required)
- `Funding` or `Matching (USDC)` + `Donations (USD)`
- Optional: `Unique donors`, `Match per unique donor`, `Match-to-Donation Multiplier`

## Deployment

### Docker

Build and run:
```bash
docker build -t gg24-analysis .
docker run -p 8501:8501 -e OPENAI_API_KEY=your-key gg24-analysis
```

### Fly.io

Deploy using the included `fly.toml`:
```bash
fly deploy
```

## Development

### Adding a New Round

1. Create a new folder (e.g., `my-round/`)
2. Add structure:
   - `problems/problems.json`
   - `projects/*.txt` (project descriptions)
   - `funding/funding_data.csv`
   - `round/round metadata.json`
3. Add to `config.txt`: `my-round:TRUE`
4. Run `alignment_eval.py` to generate results
5. View in dashboard

### Customizing Problem Identification

The problem identification prompt template is in `problem identification prompt.txt`. Modify this to adjust how problems are identified from domain materials.

## Dependencies

- `openai>=1.0.0` - LLM API client
- `streamlit>=1.28.0` - Dashboard framework
- `plotly>=5.17.0` - Interactive visualizations
- `pandas>=2.0.0` - Data processing
- `numpy>=1.24.0` - Numerical operations
- `altair>=5.0.0` - Additional visualization support

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

## Feedback

This is a work-in-progress analysis. We welcome feedback from:
- Round operators
- Domain subject matter experts
- Project owners

[Feedback Form](https://forms.gle/3kDCzjMrUZA7RDoG6)
