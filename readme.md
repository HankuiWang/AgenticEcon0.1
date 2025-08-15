# Agentic Econ Workflow - Replication Codebase

This repository contains the replication code for the "Agentic Econ Workflow" research project, implementing an AI agent-based system for automated economic research workflows. The system consists of four specialized teams that work together to conduct comprehensive economic research.

## 🏗️ Architecture OverviewThe system is built around four specialized AI agent teams, each handling different aspects of economic research:

1. **DataTeam** - Data discovery, retrieval, cleaning, and preprocessing
2. **IdeationTeam** - Research idea generation, refinement, and contextualization
3. **LiteratureTeam** - Literature review and academic source analysis
4. **ModelTeam** - Economic model development, calibration, and validation

## 📋 Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Google Custom Search API key (optional, for enhanced functionality)
- FRED API key (for economic data access)

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd AgenticWorkflow/docs/publication/IAAI-26/Suppl_Material/code

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the code directory:

```bash
# Copy example environment file
copy example.env .env

# Edit .env with your API keys
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_SEARCH_ENGINE_ID=your_cse_id_here
FRED_API_KEY=your_fred_api_key_here
```

### 3. Run Individual Teams

#### Data Team

```bash
python DataTeam.py --indicators "gdp,inflation,unemployment"
```

#### Ideation Team

```bash
# Human-guided mode
python IdeationTeam.py --idea "How does AI affect economic growth?"

# Automatic mode
python IdeationTeam.py --mode automatic

# Debug mode
python IdeationTeam.py --mode automatic --debug
```

#### Literature Team

```bash
python literatureTeam.py --topic "artificial intelligence economics"
```

#### Model Team

```bash
python ModelTeam.py --model_type "DSGE" --focus "monetary policy transmission"
```

## 🔧 Team Descriptions

### DataTeam.py

**Purpose**: Automated data pipeline for economic indicators

**Features**:

- FRED API integration for economic data
- Automated data discovery and validation
- Data cleaning and preprocessing
- Feature engineering capabilities
- Human-in-the-loop checkpoints

**Key Functions**:

- `search_indicator()` - Discover available data series
- `retrieve_data()` - Fetch data from sources
- `clean_preprocess()` - Data cleaning pipeline
- `feature_engineering()` - Create derived variables
- `validate_processed_data()` - Quality assurance

### IdeationTeam.py

**Purpose**: Generate and refine economic research questions

**Features**:

- Trend analysis from news sources
- Academic literature integration
- Policy paper discovery
- Multi-stage idea refinement
- Human expert oversight

**Key Functions**:

- `scrape_economic_news()` - Google Search API integration
- `search_policy_literature()` - Grey literature discovery
- `crawl_academic_sources()` - arXiv, SSRN, NBER access
- `generate_ideas()` - LLM-powered idea generation
- `refine_ideas()` - Idea quality improvement

### LiteratureTeam.py

**Purpose**: Comprehensive literature review automation

**Features**:

- Google Custom Search integration
- arXiv academic paper search
- Multi-source content analysis
- Gap identification
- Reference management

**Key Functions**:

- `google_search()` - Web content discovery
- `arxiv_search()` - Academic paper search
- Automated insight extraction
- Research gap identification
- Reference compilation

### ModelTeam.py

**Purpose**: Economic model development and calibration

**Features**:

- Theoretical framework selection
- Mathematical model specification
- Calibration and estimation
- Sensitivity analysis
- Human expert validation

**Key Functions**:

- Theoretical framework development
- Mathematical model translation
- Parameter calibration
- Robustness testing

## 🔄 Workflow Process

### 1. Data Pipeline (DataTeam)

```
Discovery → Retrieval → Cleaning → Feature Engineering → Validation → Documentation
```

### 2. Ideation Process (IdeationTeam)

```
Trend Analysis → Literature Review → Idea Generation → Refinement → Contextualization → Finalization
```

### 3. Literature Review (LiteratureTeam)

```
Search → Analysis → Gap Identification → Reference Compilation → Report Generation
```

### 4. Model Development (ModelTeam)

```
Theory → Mathematical Formulation → Calibration → Validation
```

## 📊 Output Files

Each team generates timestamped output files:

- `data_pipeline_YYYYMMDDHHMM.md` - Data pipeline reports
- `ideation_log_YYYYMMDD_HHMMSS.md` - Ideation process logs
- `literature_review_YYYYMMDDHHMM.md` - Literature review reports
- `modeling_log_YYYYMMDDHHMM.md` - Model development logs

## 🛠️ Customization

### Adding New Data Sources

Modify the `AVAILABLE_SOURCES` list in `DataTeam.py` and implement corresponding search/retrieval functions.

### Extending Agent Capabilities

Add new tools using the `FunctionTool` decorator and integrate them into existing agents.

### Modifying Checkpoints

Customize human-in-the-loop checkpoints by modifying the `checkpoint*_input_func` functions.

### Adding interection turns

By changing the `max_turns` parameter in the `RoundRobinGroupChat` class, you can add more interaction turns between the agents.

## 🧪 Testing

Run tests to verify system functionality:

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-mock

# Run tests
pytest tests/
```

## 📚 Dependencies

Key dependencies include:

- **AutoGen**: AI agent framework (v0.6.1)
- **OpenAI**: LLM integration
- **Pandas/NumPy**: Data processing
- **Requests/BeautifulSoup**: Web scraping
- **ArXiv**: Academic paper access
- **FRED API**: Economic data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This code is provided for research replication purposes. Please refer to the main paper for licensing information.

## 📞 Support

For questions or issues:

1. Check the existing documentation
2. Review the code comments
3. Open an issue in the repository
4. Contact the research team

## 🔗 Related Resources

- **Paper**: [Agentic Econ Workflow: AI Agent-Based Economic Research Automation]
- **Dataset**: [Economic indicators and research data]
- **Documentation**: [Extended methodology and technical details]

## 📈 Performance Notes

- **API Rate Limits**: Respect API rate limits for external services
- **Memory Usage**: Large datasets may require memory optimization
- **Execution Time**: Full workflows typically take 5-15 minutes depending on complexity
- **Scalability**: System designed for research-scale workloads

## 🚨 Troubleshooting

### Common Issues

1. **API Key Errors**: Verify all API keys are correctly set in `.env`
2. **Import Errors**: Ensure all dependencies are installed with correct versions
3. **Memory Issues**: Reduce batch sizes for large datasets
4. **Network Timeouts**: Check internet connection and API service status

### Debug Mode

Enable debug logging for detailed execution information:

```bash
python IdeationTeam.py --debug --mode automatic
```

---

*This replication codebase supports the research presented in "Agentic Econ Workflow: AI Agent-Based Economic Research Automation" and provides a foundation for extending automated economic research capabilities.*
