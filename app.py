import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv("dataset.csv")  # Replace with the actual file name

# Fill missing values
df.fillna('', inplace=True)

# Get all unique symptoms from the dataset
symptom_columns = df.columns[1:]  # All columns except 'Disease'
unique_symptoms = set()
for column in symptom_columns:
    symptoms = df[column].dropna().unique()
    unique_symptoms.update([s.strip().lower() for s in symptoms if s != ''])
unique_symptoms = sorted(list(unique_symptoms))

# Combine all symptoms into a single list per row
df['Symptoms'] = df[df.columns[1:]].values.tolist()
df['Symptoms'] = df['Symptoms'].apply(lambda x: [i.strip().lower() for i in x if i != ''])

# Encode symptoms using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df['Symptoms'])

# Encode diseases (target)
y = df['Disease']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

def display_symptoms(symptoms_list, page_size=10):
    total_pages = (len(symptoms_list) + page_size - 1) // page_size
    current_page = 0
    
    while True:
        start_idx = current_page * page_size
        end_idx = min(start_idx + page_size, len(symptoms_list))
        
        print("\nAvailable Symptoms (Page {} of {}):".format(current_page + 1, total_pages))
        for idx, symptom in enumerate(symptoms_list[start_idx:end_idx], start=1):
            print(f"{idx}. {symptom}")
        
        print("\nNavigation:")
        print("n - Next page")
        print("p - Previous page")
        print("s - Select symptom")
        print("d - Done selecting")
        choice = input("Enter your choice: ").lower()
        
        if choice == 'n' and current_page < total_pages - 1:
            current_page += 1
        elif choice == 'p' and current_page > 0:
            current_page -= 1
        elif choice == 's':
            try:
                num = int(input(f"Enter symptom number (1-{page_size}): "))
                if 1 <= num <= page_size:
                    return symptoms_list[start_idx + num - 1]
            except ValueError:
                print("Invalid input. Please enter a number.")
        elif choice == 'd':
            return None
        
        print("\n" + "="*50)

def predict_disease(symptoms):
    # Convert symptoms to lowercase and strip whitespace
    symptoms = [s.strip().lower() for s in symptoms]
    
    # Transform symptoms using the same MultiLabelBinarizer
    symptoms_encoded = mlb.transform([symptoms])
    
    # Make prediction
    prediction = model.predict(symptoms_encoded)
    probabilities = model.predict_proba(symptoms_encoded)
    max_prob = max(probabilities[0]) * 100
    
    return prediction[0], max_prob

def main():
    print("\n=== Medical Diagnosis System ===")
    print("Select your symptoms from the available list.")
    print("You can navigate through pages and select multiple symptoms.")
    
    selected_symptoms = []
    while True:
        print("\nCurrent selected symptoms:", ", ".join(selected_symptoms) if selected_symptoms else "None")
        symptom = display_symptoms(unique_symptoms)
        
        if symptom is None:
            break
            
        if symptom not in selected_symptoms:
            selected_symptoms.append(symptom)
            print(f"\nAdded symptom: {symptom}")
        else:
            print("\nThis symptom is already selected.")
    
    if not selected_symptoms:
        print("\nNo symptoms selected.")
        return
    
    disease, confidence = predict_disease(selected_symptoms)
    print("\nBased on your symptoms:")
    print(f"Selected Symptoms: {', '.join(selected_symptoms)}")
    print(f"Predicted Disease: {disease}")
    print(f"Confidence: {confidence:.2f}%")

if __name__ == "__main__":
    main()