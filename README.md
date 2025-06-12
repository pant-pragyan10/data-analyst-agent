# Data Analyst Agent

This is an intelligent data analyst agent that can analyze various types of documents and provide insights through natural language interaction.

## Features
- Support for multiple file formats (.doc, .txt, .xlsx, .csv, .pdf, images)
- Data analysis and visualization capabilities
- Natural language Q&A interface
- Powered by Together.ai's Llama-4-Maverick-17B model

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the root directory and add your Together.ai API key:
```
TOGETHER_API_KEY=your_api_key_here
```

3. Run the Jupyter notebook:
```bash
jupyter notebook data_analyst_agent.ipynb
```

## Usage
1. Upload your document using the provided interface
2. Ask questions about your data in natural language
3. The agent will analyze the data and provide insights
4. Request visualizations as needed

## Supported File Types
- Text files (.txt)
- Word documents (.doc, .docx)
- Excel files (.xlsx, .xls)
- CSV files (.csv)
- PDF files (.pdf)
- Image files (various formats) 