# RedDB Query Scripts

This directory contains comprehensive analysis scripts for exploring your RedDB data landscape. These scripts help you understand what data you have, how it's flowing through your system, and how your RL training is performing.

## 🚀 Quick Start

Run all analysis scripts at once:
```bash
python queries/run_all_queries.py
```

Or run individual scripts:
```bash
python queries/data_overview.py
python queries/streaming_data_analysis.py
python queries/rl_training_analysis.py
```

## 📊 Available Scripts

### 1. `data_overview.py` - Complete Data Landscape
**Purpose**: Shows you the complete overview of all data types in your RedDB system.

**What it shows**:
- **Data Types**: Counts of embeddings, signatures, verifiers, actions, logs, contexts, patches, retrieval events, training sessions
- **Streams**: Time-series data streams and their activity levels
- **Collections**: Registries and collections of related data
- **Performance Metrics**: System performance and health indicators
- **System Health**: Current system status and configuration

**Example Output**:
```
📋 REDDB DATA LANDSCAPE SUMMARY
============================================================
🎯 TOTAL DATA ENTITIES: 1,247
📊 BREAKDOWN:
   • 156 agent actions
   • 89 log entries  
   • 23 DSPy signatures
   • 12 code verifiers
   • 45 retrieval events
   • 8 training sessions
   • 1,234 stream entries
   • 45 registry items

📈 DATA TYPE SUMMARY:
   • 23 DSPy Signatures (18 active)
   • 12 Code Verifiers
   • 156 Agent Actions
   • 89 Log Entries
   • 12 Code Patches
   • 45 Retrieval Events
   • 8 Training Sessions
```

### 2. `streaming_data_analysis.py` - Streaming & Real-time Data
**Purpose**: Analyzes your streaming data pipeline and real-time data ingestion.

**What it shows**:
- **Streaming Sources**: Different data streams and their ingestion rates
- **Data Ingestion Rates**: How much data is being ingested per hour/day
- **Vectorization Status**: Quality of vector embeddings and similarity search
- **RL Training Data**: Quality and trends of training data
- **Real-time Metrics**: Current throughput and system performance
- **Data Flow Analysis**: Pipeline stages and potential bottlenecks

**Example Output**:
```
🌊 STREAMING DATA LANDSCAPE SUMMARY
============================================================
📊 STREAMING OVERVIEW:
   • 1,234 total stream entries
   • 2,456 data points ingested in 24h
   • 8 active streams

🔥 TOP STREAMING SOURCES:
   • action_stream: 456 entries
   • log_stream: 234 entries
   • retrieval_stream: 123 entries
   • training_stream: 89 entries
   • patch_stream: 45 entries

🤖 RL TRAINING STATUS:
   • 156 training actions
   • 0.742 average reward
   • 23 high-reward actions

⚡ REAL-TIME PERFORMANCE:
   • 45 events/hour current throughput
   • 12 actions/hour
   • 8 retrievals/hour
```

### 3. `rl_training_analysis.py` - RL Training Deep Dive
**Purpose**: Deep analysis of your reinforcement learning training data and effectiveness.

**What it shows**:
- **Action Analysis**: Action patterns, quality metrics, and effectiveness by type
- **Reward Analysis**: Reward distribution, trends, and high-reward patterns
- **Learning Progress**: Training session analysis and convergence status
- **Training Effectiveness**: Improvement over time and efficiency metrics
- **Policy Quality**: Policy consistency and decision-making patterns
- **Exploration vs Exploitation**: Balance analysis and efficiency
- **Training Recommendations**: Actionable recommendations for improvement

**Example Output**:
```
🤖 RL TRAINING ANALYSIS SUMMARY
============================================================
📊 OVERALL METRICS:
   • 156 total training actions
   • 0.742 average reward
   • 8 training sessions

📈 LEARNING PROGRESS:
   • Training accuracy trend: improving
   • Validation accuracy trend: improving
   • Loss trend: improving

⚡ TRAINING EFFECTIVENESS:
   • Reward improvement: +0.123
   • Action diversity: medium
   • Exploration balance: balanced

💡 IMMEDIATE RECOMMENDATIONS:
   1. Continue current training - system appears stable
   2. Monitor performance metrics and adjust hyperparameters as needed
   3. Plan for scaling and additional complexity as performance improves
```

### 4. `run_all_queries.py` - Master Runner
**Purpose**: Runs all analysis scripts and provides a comprehensive summary.

**What it does**:
- Executes all analysis scripts in sequence
- Provides a summary of results and execution status
- Shows generated output files
- Gives a quick system overview
- Saves results to a summary file

## 📁 Output Files

Each script generates JSON output files with detailed data:

- `data_overview.json` - Complete data landscape
- `streaming_analysis.json` - Streaming data analysis
- `rl_training_analysis.json` - RL training analysis
- `query_results_summary.json` - Summary of all query results

## 🔧 Understanding Your Data Landscape

### Data Types in Your System

1. **Embeddings** - Vector representations of code/text
2. **Signatures** - DSPy signature performance metrics
3. **Verifiers** - Code verification and quality metrics
4. **Actions** - Agent actions for RL training
5. **Logs** - Structured system logs
6. **Contexts** - Agent state and context information
7. **Patches** - Code patch generation and application records
8. **Retrieval Events** - Knowledge retrieval and search events
9. **Training Sessions** - RL training progress and metrics

### Streaming Data Sources

Your system streams data through multiple channels:

- **Code Analysis Stream** - DSPy signature performance
- **Action Stream** - Agent actions for RL training
- **Log Stream** - System logs and events
- **Retrieval Stream** - Knowledge retrieval events
- **Training Stream** - Training session data
- **Patch Stream** - Code patch history
- **Context Stream** - Agent context changes
- **Health Stream** - System health metrics

### RL Training Data

Your RL system tracks:

- **Action Quality** - Reward, confidence, execution time
- **Learning Progress** - Training/validation accuracy, loss trends
- **Policy Quality** - Consistency and decision-making patterns
- **Exploration Balance** - Exploration vs exploitation ratio
- **Training Effectiveness** - Improvement rates and efficiency

## 🎯 Use Cases

### For System Monitoring
- Run `data_overview.py` to see overall system health
- Check streaming rates and data flow
- Monitor system performance metrics

### For RL Training Optimization
- Run `rl_training_analysis.py` to understand training effectiveness
- Check reward trends and learning progress
- Get recommendations for improvement

### For Data Pipeline Analysis
- Run `streaming_data_analysis.py` to analyze data flow
- Identify bottlenecks and optimization opportunities
- Monitor real-time performance

### For Development & Debugging
- Use individual scripts to focus on specific areas
- Check data quality and consistency
- Monitor system behavior over time

## 🚨 Troubleshooting

### Common Issues

1. **No data found**: Your RedDB might be empty or not properly initialized
2. **Import errors**: Make sure you're running from the project root
3. **Permission errors**: Check file permissions for output directories

### Getting Help

If you encounter issues:

1. Check that RedDB is properly initialized
2. Verify your data manager is working: `from dspy_agent.db import get_enhanced_data_manager; dm = get_enhanced_data_manager()`
3. Run individual scripts to isolate issues
4. Check the generated JSON files for detailed error information

## 🔄 Regular Monitoring

For ongoing monitoring, consider:

- Running `run_all_queries.py` daily to track system health
- Setting up alerts based on key metrics
- Monitoring trends over time using the generated JSON files
- Using the recommendations from RL training analysis to optimize performance

---

**Remember**: These scripts are designed to give you a comprehensive view of your RedDB data landscape. Use them regularly to understand your system's behavior and optimize your RL training pipeline! 🚀
