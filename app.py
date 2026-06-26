import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42

# title
st.title('Crop type recommendation and health advisory app')
df=pd.read_csv('Crop_recommendation_augmented.csv')

# feature target splitting
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
X = df[features].copy()
y = df['label'].copy()

# Encode crop names as integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split with stratification (keeps class balance the same in both)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_encoded
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)  # using unscaled data for Random Forest as it's less sensitive to scaling

# Calculate optimal ranges for advisory system (using the full dataset)
optimal_ranges = {}
for crop in df['label'].unique():
    crop_df = df[df['label'] == crop]
    optimal_ranges[crop] = {
        feature: (round(crop_df[feature].quantile(0.10), 2),
                  round(crop_df[feature].quantile(0.90), 2))
        for feature in features
    }

# Lookup table: human-readable advice for each off-range feature
ADVISORY_TEXT = {
    'N':           {'low': 'Nitrogen is low. Apply urea or organic compost (~40-60 kg N/ha).',
                    'high': 'Nitrogen is excessive. Stop nitrogenous fertilizer; risk of lodging and disease.'},
    'P':           {'low': 'Phosphorus is low. Apply DAP or single super-phosphate.',
                    'high': 'Phosphorus is excessive. Skip phosphorus fertilizer for this cycle.'},
    'K':           {'low': 'Potassium is low. Apply muriate of potash; supports water regulation.',
                    'high': 'Potassium is excessive. Avoid further potash application.'},
    'temperature': {'low': 'Temperature is below the comfort range. Consider row covers or delayed sowing.',
                    'high': 'Temperature is above the comfort range. Increase mulching and irrigation.'},
    'humidity':    {'low': 'Humidity is low. Increase irrigation frequency; mulch to retain moisture.',
                    'high': 'Humidity is high - fungal disease risk. Improve ventilation and avoid evening irrigation.'},
    'ph':          {'low': 'Soil is too acidic. Apply agricultural lime to raise pH.',
                    'high': 'Soil is too alkaline. Apply elemental sulfur or organic matter to lower pH.'},
    'rainfall':    {'low': 'Rainfall is below the typical range. Supplement with irrigation.',
                    'high': 'Rainfall is above the typical range. Ensure drainage; risk of root rot.'}
}

def get_advisory(current_crop, conditions, model, scaler, label_encoder, optimal_ranges):
    """
    Generate a crop-health advisory.

    Parameters
    ----------
    current_crop : str           - the crop the farmer is growing (e.g. 'rice')
    conditions   : dict          - keys: N, P, K, temperature, humidity, ph, rainfall
    model        : trained classifier
    scaler       : fitted StandardScaler (unused for tree models but kept for interface)
    label_encoder: fitted LabelEncoder
    optimal_ranges: dict of {crop: {feature: (low, high)}}

    Returns
    -------
    dict with predicted crop, health status, and list of advisories
    """
    # 1. Model prediction: what crop is optimal for these conditions?
    feature_order = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    X_input = pd.DataFrame([[conditions[f] for f in feature_order]], columns=feature_order)
    pred_idx = model.predict(X_input)[0]
    recommended_crop = label_encoder.inverse_transform([pred_idx])[0]

    # 2. Compare current conditions to optimal range for the FARMER'S crop
    advisories = []
    if current_crop not in optimal_ranges:
        return {
            'recommended_crop': recommended_crop,
            'health_status': 'UNKNOWN',
            'advisories': [f"No reference data for '{current_crop}'."],
            'reason': 'Current crop not in training data.'
        }

    crop_ranges = optimal_ranges[current_crop]
    out_of_range = []
    for feature in feature_order:
        low, high = crop_ranges[feature]
        val = conditions[feature]
        if val < low:
            advisories.append(f"[{feature}] Current = {val}, optimal range = {low}-{high}. "
                              + ADVISORY_TEXT[feature]['low'])
            out_of_range.append(feature)
        elif val > high:
            advisories.append(f"[{feature}] Current = {val}, optimal range = {low}-{high}. "
                              + ADVISORY_TEXT[feature]['high'])
            out_of_range.append(feature)

    # 3. Health verdict
    if len(out_of_range) == 0:
        health = 'HEALTHY'
        verdict = f'All conditions are within the optimal range for {current_crop}.'
    elif len(out_of_range) <= 2:
        health = 'MODERATE STRESS'
        verdict = f'{len(out_of_range)} condition(s) outside optimal range for {current_crop}.'
    else:
        health = 'HIGH STRESS'
        verdict = f'{len(out_of_range)} conditions outside optimal range for {current_crop} - serious stress risk.'

    # Cross-check vs model recommendation
    if recommended_crop != current_crop and health != 'HEALTHY':
        verdict += f" The model suggests '{recommended_crop}' would be better suited to these conditions."

    return {
        'recommended_crop': recommended_crop,
        'current_crop': current_crop, # In this app, current_crop will be the predicted crop for advisory
        'health_status': health,
        'reason': verdict,
        'advisories': advisories if advisories else ['No corrective action needed.']
    }

# Streamlit Frontend
nitrogen=st.number_input("Nitrogen content in soil: ",min_value=0.0,step=0.1)
phosphorus=st.number_input("Phosphorus content in soil:",min_value=0.0,step=1.0)
potassium=st.number_input("Potassium content in soil:",min_value=0.0,step=1.0)
temperature=st.number_input("Ambient temperature (in C):",min_value=0.0,step=1.0)
humidity=st.number_input("Humidity %:",min_value=0.0,step=1.0)
rainfall=st.number_input("Rainfall in mm:",min_value=0.0,step=1.0)
ph=st.number_input("pH value of soil:",min_value=0.0,step=1.0)

if st.button("Predict Crop Type and Get Advisory"):
  input_data=np.array([[nitrogen,phosphorus,potassium,temperature,humidity,ph,rainfall]])
  prediction_encoded=rf_model.predict(input_data)
  predicted_crop = label_encoder.inverse_transform(prediction_encoded)[0]

  st.success(f'Suited crop type for the conditions is: **{predicted_crop}**')

  # Get detailed advisory
  conditions = {
      'N': nitrogen,
      'P': phosphorus,
      'K': potassium,
      'temperature': temperature,
      'humidity': humidity,
      'ph': ph,
      'rainfall': rainfall
  }
  advisory_result = get_advisory(predicted_crop, conditions, rf_model, scaler, label_encoder, optimal_ranges)

  st.subheader('Crop Health Advisory:')
  st.info(f"**Health Status for {advisory_result['current_crop']}:** {advisory_result['health_status']}")
  st.write(f"**Reason:** {advisory_result['reason']}")
  st.write("**Specific Advisories:**")
  for adv in advisory_result['advisories']:
      st.markdown(f"- {adv}")
