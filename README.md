# ML-AI

Utilizing Random Forest algorithms enhanced by Synthetic Minority Over-sampling Technique (SMOTE), this tool provides reliable predictions on whether market trends will be uptrending, downtrending, or moving sideways.

Data Source:

Optiver - Trading At Close Competition

Key Features:

Advanced Modeling: Leverages a Random Forest classifier trained on NASDAQ closing auction data to predict market movements.
Feature Engineering: Utilizes engineered features such as percentage change in weighted average price (WAP), spread, mid price, and volume imbalances to enhance model predictions.
Class Imbalance Handling: Incorporates SMOTE to effectively balance class distributions, improving the model's performance across diverse market scenarios.
Back-testing Rigor: Includes comprehensive back-testing results that demonstrate the model's efficacy and stability under various historical conditions.
Performance Metrics: Achieves a 99.5% accuracy rate, with detailed metrics provided for precision, recall, and f1-score across all classes, ensuring robustness and reliability for quantitative trading strategies.
Python and Scikit-Learn Integration: Developed using Python and the scikit-learn library, ensuring easy integration and scalability within existing trading systems.
