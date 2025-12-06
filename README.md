## User Guide
### ⚙️ Prerequisites & Setup
These are some pre-requisites that needs to be run or ensured before starting.

#### Python Installation
- Python **3.10 or higher** is recommended.
- Verify installation:

```bash
  python --version
```

#### Project Dependencies
Install required packages:

```bash
pip install -r requirements.txt
```

#### Tools - Synthetic Data Generation

The script `simulate_licensing_data.py` generates a synthetic dataset representing media licensing deals. The configuration file `tools\config\sim_config.toml` can be used to alter parameters that generate the data. The output file is generated at the location `data/source/licensing_deals.csv`. 
Use this file to upload to the application interface.

**Usage**

```bash
python ${PROJECT_ROOT}/tools/simulate_licensing_data.py
```

### ▶️ Running the App
From the project root directory, run Streamlit:

```bash
python -m streamlit run src/app.py
```

This will automatically start a local web server and open the app in your default browser.
If it doesn’t open automatically, navigate to:

http://localhost:8501

### 💡Navigating the App - User Guide
1. **Trigger New Pipeline**: From the sidebar, click the "New Pipeline" button to trigger a new pipeline process. Alternatively, you can also click on an existing pipeline run from the list above the button, which will load details regarding the same, and allow you to continue based on the stage the run currently is.
2. **Data Staging**: Select either of the options (file upload or URL) to load the file of your choosing. Refer above for synthetic dataset generator tool. Only IMDB urls are supported for feature master creation. Each data file loaded is staged, and preview is available to see data characteristics, along with schema validation output. 
4. **Feature Master**: Once files are staged, based on selection available, feature master needs to be built. Please note currently, only synthetic dataset serves as the base and IMDb data files are aggregated on top of that to build the final feature master. Therefore, the size of the synthetic dataset is what would be the eventual feature master size.
5. **Data Exploration Panels**: Display Data, EDA, Preprocessing (Cleaning) provides manipulation capabilities on the feature master file. Allows to inspect missing values, outliers, and variable distributions. Expanders reveal options for imputation, normalization, and encoding, with summary comparisons shown before and after cleaning. Once satisfied, save a cleaned `parquet` version of the feature master. Saved paths are displayed on the UI. It uses the `data` directory as the root under your project, make sure to check out the `local.env` file for the same.
5. **Analytical Tools (Modeling)**:  Enables users to select and execute predictive models (Linear Regression, Ridge, Gradient Boosting). Model metrics such as R-sqaured, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) are automatically computed and displayed via interactive charts.
6. **Visual Tools**: Includes visualization and exploration utilities. Users can view correlation heatmaps, distribution charts, and engagement-level analyses across various features.
7. **Reporting**: After modeling, users can generate visual performance reports, download summaries in PDF or CSV format, and review metadata logs stored under `data/reports/<run_id>`.

This below table provides contextual guidance for using each pipeline expander panel responsibly within the user interface.

| **Pipeline Sections**            | **Help & Compliance Guidance**                                                                                                                                                             |
|----------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Data Staging**                 | Supported file formats (CSV, Parquet, TZ) and schema validation rules enforced by the `validator/schema_validator.py` module. Avoid uploading personally identifiable or proprietary data. |
| **Feature Master**               | Features are generated and consolidated from staged datasets.                                                                                                                              |
| **Display Data**                 | Provides guidance on interpreting descriptive statistics and column summaries. Exploratory summaries are diagnostic and not final decision outputs.                                        |
| **Exploration (EDA)**            | Offers recommendations for safe data visualization, including use of filters and scaling. Avoid drawing conclusions without statistical validation.                                        |
| **Preprocessing (and Cleaning)** | Outlines how normalization, encoding, and outlier handling modify data distributions. Save final output as a cleaned feature master, ready for modeling                                    |
| **Analytical Tools – Model**     | Lists supported models (Linear Regression, Ridge, Gradient Boosting) and defines key metrics such as R², MAE, and RMSE.                                                                    |
| **Visual Tools**                 | Details interactive chart behavior (hover, filter, zoom) and cautions against overfitting when visually comparing model outcomes.                                                          |
| **Reporting**                    | Current report generated summaries the model outputs, along with a data preview.                                                                                                           |


_**Note**: The app is designed to run locally for now, with plans to deploy on a cloud platform like AWS in future.

### Limitations and Future Work
While the Pricing & Licensing Insights application provides a robust and modular framework for licensing price estimation, several limitations remain that offer opportunities for future enhancement. The current implementation relies on a streamlined dataset and controlled modeling assumptions, which may limit generalizability when applied to larger, heterogeneous licensing catalogs. Additionally, the pricing model is trained using a single algorithmic approach, and expanding this to include model comparison, hyperparameter tuning, or ensemble strategies could improve predictive accuracy. System monitoring is functional but could be extended with automated drift detection, retraining triggers, and enhanced test coverage. Finally, while the user interface supports all major workflows, additional usability improvements, such as, scenario-based wizards, guided tooltips, or role-specific dashboards, could enhance accessibility for non-technical stakeholders.