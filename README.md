# Rent Affordability Analysis Project

**Group 10**  
**Created by**: May Sabai (017390438), Jerry Nguyen (016437330), Anusri Nagarajan (017743700)
## Project Overview

This project analyzes rent affordability across 369 U.S. cities in 50 states, examining the relationship between renter income, median income, population, minimum wage, property taxes, and rent burden. The analysis includes exploratory data analysis, visualizations, and machine learning classification models to predict whether cities have affordable rent (≤30% of income) or unaffordable rent (>30% of income).

## Repository Structure

```
term-project-group10/
├── Project Assignments/
│   ├── data/                                    # Dataset files
│   │   ├── renter_affordability_data - 2023.csv
│   │   └── 2024_renter_affordability_data.csv
│   ├── CS133_Final_Data_Analysis.ipynb          # Main exploratory data analysis
│   ├── CS133_Final_ML.ipynb                     # Machine learning models
│   ├── affordability_plot.html                  # Interactive map visualization for v2
│   ├── CS133_F25_P2_categorical_interactive_transform.v1.ipynb
│   ├── CS133_F25_P2_categorical_interactive_transform.v2.ipynb
│   └── P1_scatter_plots.ipynb
├── requirements.txt                             # Python package dependencies
└── README.md
```

## Dataset

- **Source**: Renter affordability data for U.S. cities (2023)
- **Size**: 369 cities across 50 states
- **Key Variables**:
  - `Average Affordability`: Rent as a percentage of income (target variable)
  - `Population`: City population
  - `Renter Income`: Average income of renters
  - `Median Income`: Median household income
  - `Minimum Wage`: State minimum wage
  - `State property tax`: State property tax rate
  - `CityName`, `StateName`, `abb_StateName`: Location identifiers
  - `RegionID`, `SizeRank`: Additional identifiers

## Environment Setup

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or Google Colab
- pip package manager

### Installing Required Packages

All required packages are listed in the `requirements.txt` file. Install them using:

```bash
pip install -r requirements.txt
```

### Package List

The following packages will be installed:

**Core Data Analysis & Machine Learning:**
- `pandas>=2.0.0` - Data manipulation and analysis
- `numpy>=1.24.0` - Numerical computing
- `scikit-learn>=1.3.0` - Machine learning algorithms

**Visualization:**
- `matplotlib>=3.7.0` - Basic plotting and visualization
- `seaborn>=0.12.0` - Statistical data visualization
- `plotly>=5.14.0` - Interactive visualizations

**Jupyter Environment:**
- `jupyter>=1.0.0` - Jupyter notebook interface
- `ipykernel>=6.25.0` - IPython kernel for Jupyter

**Note**: `google.colab` is pre-installed in Google Colab and does not need separate installation.

## Running Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd term-project-group10
```

### 2. Set Up Your Environment

**Option A: Local Jupyter Notebook**

```bash
# Install required packages
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook
```

Navigate to `Project Assignments/` and open the desired notebook.

**Option B: Google Colab**

**Important**: To run the notebooks on Google Colab without errors, you must set up the correct folder structure in your Google Drive.

1. **Download all project files** from the repository to your local machine
2. **Create the folder structure** in your Google Drive:
   - Navigate to "My Drive"
   - Create a folder named `project` (lowercase)
   - Inside the `project` folder, create a subfolder named `data`
3. **Upload files to Google Drive**:
   - Upload the notebooks (`CS133_Final_Data_Analysis.ipynb`, `CS133_Final_ML.ipynb`) to any location in your Google Drive
   - Upload the CSV data files to `MyDrive/project/data/`:
     - `renter_affordability_data - 2023.csv`
     - `2024_renter_affordability_data.csv`
4. **Open the notebook** with Google Colab (right-click the .ipynb file → Open with → Google Colaboratory)
5. **Mount your Google Drive** by running the first cell:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
6. The notebooks will now be able to access the data at the correct path: `/content/drive/MyDrive/project/data/`

**Expected Google Drive Structure:**
```
Google Drive/
└── My Drive/
    └── project/              ← Must be lowercase "project"
        └── data/
            ├── renter_affordability_data - 2023.csv
            └── 2024_renter_affordability_data.csv
```

### 3. Run the Analysis

#### Data Analysis Notebook (`CS133_Final_Data_Analysis.ipynb`)

This notebook performs exploratory data analysis and answers five key research questions:

1. **Why do people spend more than 30% of their income on rent?**
2. **What key factors influence rent change?**
3. **Are rent increases consistent across state or concentrated in cities?**
4. **Why is the required income for affordable housing in some cities higher than the median income?**
5. **How can data-driven insights help improve financial decision-making and housing policy?**

**To run**: Open the notebook and execute all cells sequentially (Cell > Run All)

#### Machine Learning Notebook (`CS133_Final_ML.ipynb`)

This notebook builds and evaluates classification models to predict rent affordability:

**Data Preparation:**
- Feature engineering (income ratios, tax burden, income gaps)
- Categorical encoding (population categories, income levels)
- Feature scaling with StandardScaler
- Train-test split with stratified sampling (80/20)

**Models Implemented:**
1. **Logistic Regression**
   - Accuracy: 97%
   - Best precision for minority class (unaffordable cities)
   
2. **Random Forest Classifier**
   - Accuracy: 96%
   - Strong overall performance with feature importance analysis
   
3. **K-Nearest Neighbors (KNN)**
   - Accuracy: 92%
   - Lower recall for minority class

**Evaluation Metrics:**
- 5-fold stratified cross-validation
- Confusion matrices
- Precision, recall, F1-score
- ROC curves and AUC scores
- Precision-Recall curves
- Feature importance analysis

**To run**: Open the notebook and execute all cells sequentially (Cell > Run All)

### 4. Interactive Visualizations

The project includes an interactive choropleth map showing average rent affordability by state:

- **File**: `average_rent_affordability_map.html`
- **To View**: Download and open the HTML file in any web browser
- **Features**: 
  - Hover over states to see average rent burden
  - Color-coded by affordability level
  - State-by-state comparison

## Key Findings

### Affordability Crisis
- **12.7%** of cities have rent burden >30%
- **84 cities (22.8%)** require income higher than median income for affordable housing
- **Top burden cities**: New York (41.3%), Miami (41.0%), Port St. Lucie (40.1%)

### Correlation Analysis
- **Strongest positive correlation**: Renter Income (r = 0.758)
- **Moderate positive**: Minimum Wage (r = 0.333)
- **Moderate negative**: State Property Tax (r = -0.327)

### Machine Learning Results
- **Best Model**: Logistic Regression (97% accuracy)
- **Most Important Features**: 
  - Renter Income
  - Renter income gap
  - Property tax burden

## Code Documentation

### Feature Engineering Function

The `affordability_feature_engineering()` function creates derived features:

```python
def affordability_feature_engineering(df):
    """
    Create new features from existing rent affordability data
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with rent affordability data
        
    Returns:
    --------
    df_fe : pandas.DataFrame
        Enhanced dataframe with engineered features:
        - min_wage_to_median_ratio: Annual minimum wage / median income
        - property_tax_burden: Property tax rate × median income
        - renter_income_gap: Median income - renter income
        - renter_income_percentage: Renter income / median income
        - population_category: Categorical population bins
        - income_level: Categorical income bins
    """
```

### Data Preprocessing Pipeline

1. **Missing Value Handling**: Rows with NA values in key columns are dropped
2. **Feature Engineering**: Creates income ratios, gaps, and categorical variables
3. **Encoding**: One-hot encoding for categorical features
4. **Scaling**: StandardScaler for numerical features
5. **Train-Test Split**: 80/20 split with stratified sampling

## Policy Recommendations

1. **Individual Financial Decisions**
   - Target cities with rent <30% of income
   - Use 30% rule for housing budgeting
   
2. **Housing Policy**
   - Increase housing supply in high-demand markets
   - Implement zoning reform
   - Provide rental assistance programs
   
3. **Employer Strategies**
   - Adjust salaries based on local rent burden
   - Offer remote work options
   
4. **Investment Opportunities**
   - Focus on supply-constrained, high-demand markets
   - Target cities with >30% rent burden and large populations

## Authors

- **May Sabai** (017390438)
- **Jerry Nguyen** (016437330)
- **Anusri Nagarajan** (017743700)

## Last Updated

November 29, 2025

## License

This project is for educational purposes as part of CS133 coursework.

