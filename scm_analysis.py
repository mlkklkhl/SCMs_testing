"""
Structural Causal Model using Structural Equation Modeling Analysis
-----------------------------------------
This script performs structural equation modeling analysis on environmental data
to understand relationships between various environmental factors according to SCMs.

Requirements:
- Python 3.7+
- semopy
- pandas
- numpy

Usage:
python scm_analysis.py --data_path <path_to_csv> --model <model_type> --sample_frac <fraction>
"""

import argparse
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import semopy as sem
from semopy.means import estimate_means

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model definitions
MODELS = {
    'pc': """
    WS, WD, PCPT, PM2.5, TP, HM ~ Season
    PM2.5 ~  TP
    TP ~  HM
    WD, HM ~  WS
    PM2.5, TP, HM ~  PCPT
    """,

    'chow_liu': """
    PCPT, PM2.5, TP ~ Season
    HM ~ TP
    WS ~ HM
    WD ~ WS
    """,

    'score_based': """
    WS, WD, PCPT, PM2.5, TP, HM ~ Season
    TP ~  HM
    WD, HM ~  WS
    """,

    'clustering': """
    l1 =~ WS + WD + PCPT + PM2.5 + TP + HM + Season
    """,

    'hierarchical': """
    l1 =~ HM + TP
    l2 =~ WD + WS
    l3 =~ Season + PM2.5
    l4 =~ l3 + PCPT
    """
}


class SCMAnalysis:
    """Class for performing Structural Equation Modeling analysis."""

    def __init__(self, data_path: str, model_type: str = 'pc'):
        """
        Initialize SCM analysis.

        Args:
            data_path (str): Path to the CSV data file
            model_type (str): Type of model to use (default: 'pc')
        """
        self.data_path = Path(data_path)
        self.model_type = model_type
        self.required_columns = ['TP', 'WS', 'AP', 'HM', 'WD', 'PCPT', 'PM2.5', 'Season']

    def load_data(self) -> pd.DataFrame:
        """Load and preprocess the data."""
        try:
            data = pd.read_csv(self.data_path, encoding='utf8')
            logger.info(f"Loaded data with {len(data)} rows")

            # Convert int64 columns to float64
            for col in data.columns:
                if data[col].dtype == 'int64':
                    data[col] = data[col].astype('float64')

            # Select required columns and drop NA values
            data = data[self.required_columns].dropna()
            logger.info(f"Preprocessed data has {len(data)} rows")

            return data

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def fit_model(self, data: pd.DataFrame, sample_frac: float = 1.0):
        """
        Fit the SCM model to the data.

        Args:
            data (pd.DataFrame): Input data
            sample_frac (float): Fraction of data to sample (default: 1.0)
        """
        try:
            if sample_frac < 1.0:
                data = data.sample(frac=sample_frac)
                logger.info(f"Sampled {len(data)} rows ({sample_frac * 100}% of data)")

            model = sem.Model(MODELS[self.model_type])
            model.fit(data, obj="MLW", solver="SLSQP")

            return model

        except Exception as e:
            logger.error(f"Error fitting model: {str(e)}")
            raise

    def analyze(self, model, output_path: str = None):
        """
        Analyze the fitted model and generate results.

        Args:
            model: Fitted SCM model
            output_path (str): Path to save the plot (default: None)
        """
        try:
            # Get model inspection results
            result = model.inspect(mode='list', what="names", std_est=True)
            logger.info("\nModel Inspection Results:")
            print(result.to_string())

            # Calculate and display fit statistics
            stats = sem.calc_stats(model)
            logger.info("\nModel Fit Statistics:")
            print(stats.to_string())

            # Generate plot if output path is provided
            if output_path:
                sem.semplot(model, output_path, plot_covs=True)
                logger.info(f"Plot saved to {output_path}")

        except Exception as e:
            logger.error(f"Error analyzing model: {str(e)}")
            raise


def main():
    """Main function to run the SCM analysis."""
    parser = argparse.ArgumentParser(description='Run SCM analysis on environmental data')

    # Define arguments only once with their defaults
    parser.add_argument('--data_path',
                        default='data/partition_combined_data_upsampled_pm_3H_spline_1degreeM.csv',
                        type=str,
                        help='Path to input CSV file')
    parser.add_argument('--model_type',
                        default='pc',
                        type=str,
                        choices=MODELS.keys(),
                        help='Type of model to use')
    parser.add_argument('--sample_frac',
                        type=float,
                        default=1.0,
                        help='Fraction of data to sample')
    parser.add_argument('--output_plot',
                        type=str,
                        default=None,
                        help='Path to save the output plot')

    # Parse arguments once
    args = parser.parse_args()

    # Set default output plot name if not provided
    if args.output_plot is None:
        args.output_plot = f'test_{args.model_type}.png'

    try:
        # Initialize and run analysis
        analysis = SCMAnalysis(args.data_path, args.model_type)
        data = analysis.load_data()
        model = analysis.fit_model(data, args.sample_frac)
        analysis.analyze(model, args.output_plot)

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()