# Machine Learning (ESS 469) Project

## Contributors
- Benz Poobua (spoobu@uw.edu)
- Jake Ward (jakobtw@uw.edu)

## Introduction
Earthquakes, known for their unpredictable nature, challenge our current understanding of seismic behavior. The Tohoku Earthquake, for instance, demonstrated the unpredictability of seismic events by striking an area considered low risk at the time (Seth Stein et al., 2011). Tragically, nearly a million lives have been lost since 1990 due to such events (Lara et al., 2023).

In response to the unpredictable nature of earthquakes, seismologists have developed Early Warning Systems (EWS). These systems involve complex models incorporating various factors, and the advancements in Machine Learning, as exemplified by the work of Mousavi and his team, have led to the continuous development of new models. Notably, a recent paper by Pablo Lara and his team introduces E3WS, a novel single-station Early Warning System characterized by cost-effectiveness and remarkably accurate predictions.

Motivated by Lara's innovative model, our objective is to construct a layered XGBoost Regression model. This model will be trained on offshore low-magnitude earthquakes in the Pacific Northwest, offering insights into its performance and how it compares to existing models.

## Methodology
### Workflow Overview
In this project, we used the stacked model (ensemble), the same as Lara et al. (2023). A stacked model combines all results from all base models and provides the prediction using the meta-model. From the mentioned literature, the authors used stacking for the source characterization. For this project, we will follow the steps provided by Lara et al. (2023) to see the difference in performance when using different datasets. We will use K-fold (K = 10) cross-validation and XGBoost (XGBRegressor) as the base model. XGBoost does a great job predicting magnitude based on the previously obtained residuals scaled by the learning rate. Lasso is used to regularize the model. 

We will obtain the out-of-fold (OOF) predictions during the iterations and use those values as inputs to the meta-model instead of performance evaluation by mean score. Then, we will construct a new meta-input using the feature vectors and the predicted magnitude during training. The stacked model consists of XGBoost as the base model, Lasso as the meta-model, and a test set. The final results are the predicted magnitude we can use to evaluate the model’s performance using Mean Absolute Error (MAE). This project will compare the model's performance with and without Lasso (i.e., only XGBoost) to the actual magnitudes. 

### Dataset
We use the IRIS database (PNSN cataloged). We applied the filter to get the raw data: location constraints, magnitude range, time frame from 2015 to 2023, and Mount St. Helens events excluded. After obtaining the raw data, we must drop NaN values before model training and fitting. As input to the models, the feature vectors serve as a numerical representation of each seismic wave, capturing information from both temporal and spectral dimensions. Notable characteristics included in this representation encompass energy levels, amplitude values, and maximum amplitude measurements (More details on feature vectors here: https://github.com/UW-ESS-DS/mlgeo-2023-PNSN-E3WS/blob/feature-stevens-examples/PNSN_src/util/display_featurevector.py). We acquired feature vectors (f000-f139) and magnitude data frames. We converted those data frames to numpy arrays for later use. Namely, the model prediction functions expect the arrays as input. We performed data splitting using an 80% train set and a 20% test set, according to Lara et al. (2023). The histogram shows the nearly normal distribution of the dataset.

## Result
The MAE of trained data during K-fold cross-validation iteration is 0.1923, displaying a good fit for the data. The MAE of XGBoost alone is 0.1960, slightly higher than the trained values. However,  the insignificant difference in the errors indicates that XGBoost did a great job and was sufficient to predict the earthquake's magnitude. Combining the meta-model or Lasso, the MAE is larger than XGBoost's (0.2003 for stacking). Overall, employing an ensemble (i.e., Lasso) is potentially unnecessary if we have only one base model, which is XGBoost. ![Alt text](https://github.com/Benz-Poobua/ESS-469-Project/blob/main/EQ_Result)

## Acknowledgement
I sincerely thank Napat Srichan (nsrichan@ucsc.edu) for his invaluable assistance in the knowledge domain of coding and insightful suggestions on coding methodologies. His expertise and guidance have significantly contributed to the success of this project. Also, I would like to express my gratitude to Nate Stevens (ntsteven@uw.edu) for his essential role in facilitating the construction of the dataset and feature vectors. His efforts have significantly contributed to the foundational aspects of this academic endeavor.

## References

Lara, Pablo, Quentin Bletery, Jean‐paul Ampuero, Adolfo Inza, and Hernando Tavera. "Earthquake Early Warning starting from 3 s of records on a single station with machine learning." Journal of Geophysical Research: Solid Earth 128, no. 11 (2023): e2023JB026575. https://doi.org/10.1029/2023JB026575
![image](https://github.com/Benz-Poobua/ESS-469-Project/assets/146503034/5f404cac-cd05-47e6-8c41-ed10a949c516)
Seth Stein, Robert Geller, Mian Liu; Bad Assumptions or Bad Luck: Why Earthquake Hazard Maps Need Objective Testing. Seismological Research Letters 2011; 82 (5): 623–626. doi: https://doi.org/10.1785/gssrl.82.5.623
