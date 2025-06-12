# Import required libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from docx import Document
import PyPDF2
from PIL import Image
import together
from dotenv import load_dotenv
import json
from typing import Union, List, Dict, Any
import io
import base64
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load environment variables
load_dotenv()

# Initialize Together.ai client
together.api_key = os.getenv('TOGETHER_API_KEY')
MODEL_NAME = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"

class DataAnalystAgent:
    def __init__(self):
        self.data = None
        self.data_type = None
        self.file_path = None
        
    def load_file(self, file_path: str) -> bool:
        """Load and process different types of files"""
        self.file_path = file_path
        file_extension = file_path.split('.')[-1].lower()
        
        try:
            if file_extension in ['csv', 'xlsx', 'xls']:
                self.data = pd.read_excel(file_path) if file_extension in ['xlsx', 'xls'] else pd.read_csv(file_path)
                self.data_type = 'tabular'
            elif file_extension in ['doc', 'docx']:
                doc = Document(file_path)
                self.data = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
                self.data_type = 'text'
            elif file_extension == 'pdf':
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    self.data = '\n'.join([page.extract_text() for page in pdf_reader.pages])
                self.data_type = 'text'
            elif file_extension in ['txt']:
                with open(file_path, 'r', encoding='utf-8') as file:
                    self.data = file.read()
                self.data_type = 'text'
            elif file_extension in ['png', 'jpg', 'jpeg', 'gif']:
                self.data = Image.open(file_path)
                self.data_type = 'image'
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            return True
        except Exception as e:
            print(f"Error loading file: {str(e)}")
            return False
    
    def get_data_summary(self) -> str:
        """Generate a comprehensive data summary"""
        if self.data_type != 'tabular':
            return "Data summary only available for tabular data"
        
        summary = []
        
        # Basic dataset information
        summary.append(f"Dataset Overview:\n- Rows: {len(self.data)}\n- Columns: {len(self.data.columns)}")
        
        # Column types and basic stats
        summary.append("\nColumn Information:")
        for col in self.data.columns:
            dtype = self.data[col].dtype
            unique_count = self.data[col].nunique()
            if pd.api.types.is_numeric_dtype(dtype):
                mean = self.data[col].mean()
                std = self.data[col].std()
                summary.append(f"- {col}: {dtype} (mean={mean:.2f}, std={std:.2f})")
            else:
                summary.append(f"- {col}: {dtype} ({unique_count} unique values)")
        
        
        # Data quality checks
        summary.append("\nData Quality:")
        missing_values = self.data.isnull().sum()
        if missing_values.any():
            summary.append("Missing Values:")
            for col in missing_values[missing_values > 0].index:
                summary.append(f"- {col}: {missing_values[col]} missing values")
        

        # Outlier detection for numeric columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary.append("\nPotential Outliers (using IQR method):")
            for col in numeric_cols:
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = self.data[(self.data[col] < Q1 - 1.5 * IQR) | (self.data[col] > Q3 + 1.5 * IQR)][col]
                if len(outliers) > 0:
                    summary.append(f"- {col}: {len(outliers)} potential outliers")
        
        return "\n".join(summary)
    
    def get_image_analysis(self) -> str:
        """Analyze image data and return basic information"""
        if self.data_type != 'image':
            return "Image analysis only available for image files"
        
        try:
            # Get basic image information
            width, height = self.data.size
            mode = self.data.mode
            format = self.data.format
            
            # Convert image to numpy array for analysis
            img_array = np.array(self.data)
            
            # Calculate basic statistics
            if len(img_array.shape) == 3:  # Color image
                channels = img_array.shape[2]
                mean_values = [np.mean(img_array[:,:,i]) for i in range(channels)]
                std_values = [np.std(img_array[:,:,i]) for i in range(channels)]
                stats = f"Channel Statistics:\n"
                for i in range(channels):
                    stats += f"Channel {i+1}: Mean={mean_values[i]:.2f}, Std={std_values[i]:.2f}\n"
            else:  # Grayscale image
                mean_value = np.mean(img_array)
                std_value = np.std(img_array)
                stats = f"Image Statistics:\nMean={mean_value:.2f}, Std={std_value:.2f}\n"
            
            return f"""Image Analysis:
Format: {format}
Dimensions: {width}x{height} pixels
Color Mode: {mode}
{stats}"""
        except Exception as e:
            return f"Error analyzing image: {str(e)}"
    
    def analyze_data(self, query: str) -> Dict[str, Any]:
        """Analyze data based on the query using the Llama model"""
        if self.data is None:
            return {"error": "No data loaded"}
        
        # Prepare context based on data type
        if self.data_type == 'tabular':
            # Get comprehensive data summary
            data_summary = self.get_data_summary()
            
            # Prepare additional context for ML analysis
            ml_context = ""
            if len(self.data.select_dtypes(include=[np.number]).columns) > 0:
                # Perform PCA
                numeric_data = self.data.select_dtypes(include=[np.number])
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(numeric_data)
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(scaled_data)
                
                # Perform clustering
                kmeans = KMeans(n_clusters=3, random_state=42)
                clusters = kmeans.fit_predict(scaled_data)
                
                ml_context = f"""
                Advanced Analysis:
                - PCA shows {pca.explained_variance_ratio_[0]:.2%} variance explained by first component
                - K-means clustering identified {len(np.unique(clusters))} distinct groups
                """
            
            context = f"""
            Data Summary:
            {data_summary}
            
            {ml_context}
            
            Feature Engineering Suggestions:
            - Create interaction terms between related features
            - Bin continuous variables into categories
            - Create ratio features for related measurements
            - Normalize/standardize numeric features
            
            Business Applications:
            - Predictive modeling for crop recommendations
            - Soil quality assessment and improvement
            - Resource optimization for agriculture
            - Climate impact analysis on crop yields
            """
        elif self.data_type == 'text':
            context = f"Text content:\n{self.data[:1000]}..."  # First 1000 characters
        elif self.data_type == 'image':
            # Get image analysis
            image_analysis = self.get_image_analysis()
            context = f"""
            Image Analysis:
            {image_analysis}
            
            Note: This is a basic image analysis. For detailed image understanding, 
            you might want to use specialized computer vision models or tools.
            """
        
        # Prepare prompt for the model
        prompt = f"""You are a data analyst assistant. Analyze the following data and answer the question.
        Focus on providing actionable insights and clear explanations.

        Context:
        {context}

        Question: {query}

        Please structure your response to include:
        1. Brief summary of the data
        2. Key findings and patterns
        3. Potential data issues or limitations
        4. Business implications and applications
        5. Next steps or recommendations
        6. Visualizations if needed (specify type)

        Keep the tone professional but engaging."""
        
        try:
            response = together.Complete.create(
                prompt=prompt,
                model=MODEL_NAME,
                max_tokens=1000,
                temperature=0.7,
                top_p=0.7,
                top_k=50,
                repetition_penalty=1.1
            )
            
            # Extract the response text from the Together.ai API response
            if isinstance(response, dict) and 'choices' in response:
                analysis = response['choices'][0]['text']
            else:
                analysis = str(response)
            
            return {
                "analysis": analysis,
                "data_type": self.data_type
            }
        except Exception as e:
            return {"error": f"Error in analysis: {str(e)}"}
    
    def create_visualization(self, plot_type: str, **kwargs) -> Union[str, None]:
        """Create visualizations for tabular data"""
        if self.data_type != 'tabular':
            return "Visualization only available for tabular data"
        
        plt.figure(figsize=(10, 6))
        
        try:
            if plot_type == 'histogram':
                plt.hist(self.data[kwargs.get('column')], bins=kwargs.get('bins', 10))
                plt.title(f'Histogram of {kwargs.get("column")}')
                plt.xlabel(kwargs.get('column'))
                plt.ylabel('Frequency')
            elif plot_type == 'scatter':
                plt.scatter(self.data[kwargs.get('x')], self.data[kwargs.get('y')])
                plt.title(f'Scatter Plot: {kwargs.get("x")} vs {kwargs.get("y")}')
                plt.xlabel(kwargs.get('x'))
                plt.ylabel(kwargs.get('y'))
            elif plot_type == 'box':
                self.data.boxplot(column=kwargs.get('column'))
                plt.title(f'Box Plot of {kwargs.get("column")}')
                plt.ylabel(kwargs.get('column'))
            elif plot_type == 'correlation':
                sns.heatmap(self.data.corr(), annot=True, cmap='coolwarm')
                plt.title('Correlation Matrix')
            elif plot_type == 'pca':
                # Perform PCA
                numeric_data = self.data.select_dtypes(include=[np.number])
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(numeric_data)
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(scaled_data)
                
                plt.scatter(pca_result[:, 0], pca_result[:, 1])
                plt.title('PCA Visualization')
                plt.xlabel('First Principal Component')
                plt.ylabel('Second Principal Component')
            
            plt.tight_layout()
            
            # Save plot to base64 string
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode()
            plt.close()
            
            return img_str
        except Exception as e:
            plt.close()
            return f"Error creating visualization: {str(e)}"

# Example usage
if __name__ == "__main__":
    # Initialize the agent
    agent = DataAnalystAgent()
    
    # Example: Load a CSV file
    # agent.load_file('your_data.csv')
    
    # Example: Analyze data
    # result = agent.analyze_data("What are the main trends in this data?")
    # print(result['analysis'])
    
    # Example: Create visualization
    # if result['data_type'] == 'tabular':
    #     plot = agent.create_visualization('histogram', column='column_name')
    #     if isinstance(plot, str) and not plot.startswith('Error'):
    #         from IPython.display import Image, display
    #         display(Image(base64.b64decode(plot)))







