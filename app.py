import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import json
from ttkthemes import ThemedTk

class MedicalDiagnosis:
    def __init__(self, root):
        self.root = root
        self.root.title("Medical Diagnosis System")
        self.root.geometry("800x600")
        
        self.initialize_model()
        self.load_treatments()
        self.create_frames()
        self.create_widgets()
        self.selected_symptoms = []
        
    def initialize_model(self):
        # Load the dataset
        df = pd.read_csv("dataset.csv")
        
        # Combine symptoms into a list per record
        df['Symptoms'] = df[df.columns[1:]].values.tolist()
        
        # Clean and process symptoms
        def clean_symptoms(symptoms_list):
            cleaned = []
            for symptom in symptoms_list:
                # Convert to string if not NaN, strip, and convert to lowercase
                if pd.notna(symptom) and str(symptom).strip():
                    cleaned.append(str(symptom).strip().lower())
            return cleaned
            
        df['Symptoms'] = df['Symptoms'].apply(clean_symptoms)
        
        # Get unique symptoms for the GUI
        self.unique_symptoms = set()
        for symptoms in df['Symptoms']:
            self.unique_symptoms.update(symptoms)
        self.unique_symptoms = sorted(list(self.unique_symptoms))
        
        # Encode symptoms into binary format
        self.mlb = MultiLabelBinarizer()
        X = self.mlb.fit_transform(df['Symptoms'])
        
        # Target labels (Diseases)
        y = df['Disease']
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Naive Bayes model
        self.model = MultinomialNB()
        self.model.fit(X_train, y_train)
        
        # Store disease classes
        self.disease_classes = self.model.classes_
        
        # Print accuracy for monitoring
        y_pred = self.model.predict(X_test)
        print("Model Accuracy:", accuracy_score(y_test, y_pred))

    def load_treatments(self):
        try:
            with open('treatments.json', 'r') as f:
                self.treatments = json.load(f)
        except FileNotFoundError:
            self.treatments = {}
        # Load precautions from CSV
        try:
            precaution_df = pd.read_csv('symptom_precaution.csv')
            self.precautions = {}
            for _, row in precaution_df.iterrows():
                disease = str(row['Disease']).strip().lower()
                precautions = [str(row[p]).strip() for p in precaution_df.columns[1:] if pd.notna(row[p]) and str(row[p]).strip()]
                self.precautions[disease] = precautions
        except Exception as e:
            print(f"Could not load precautions: {e}")
            self.precautions = {}
        # Load descriptions from CSV
        try:
            desc_df = pd.read_csv('symptom_Description.csv')
            self.descriptions = {str(row['Disease']).strip().lower(): str(row['Description']).strip() for _, row in desc_df.iterrows()}
        except Exception as e:
            print(f"Could not load descriptions: {e}")
            self.descriptions = {}
    
    def create_frames(self):
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        
        self.search_frame = ttk.LabelFrame(self.main_frame, text="Search Symptoms", padding="5")
        self.search_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.lists_frame = ttk.Frame(self.main_frame)
        self.lists_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.result_frame = ttk.LabelFrame(self.main_frame, text="Diagnosis Results", padding="5")
        self.result_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.treatment_frame = ttk.LabelFrame(self.main_frame, text="Treatment Information", padding="5")
        self.treatment_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
    def create_widgets(self):
        self.search_var = tk.StringVar()
        self.search_var.trace('w', self.update_symptom_list)
        self.search_entry = ttk.Entry(self.search_frame, textvariable=self.search_var)
        self.search_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=5)
        self.search_frame.columnconfigure(0, weight=1)
        
        available_frame = ttk.LabelFrame(self.lists_frame, text="Available Symptoms")
        available_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        self.available_symptoms = tk.StringVar(value=self.unique_symptoms)
        self.symptoms_listbox = tk.Listbox(available_frame, listvariable=self.available_symptoms,
                                         selectmode=tk.SINGLE, height=15)
        self.symptoms_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        scrollbar = ttk.Scrollbar(available_frame, orient=tk.VERTICAL, 
                                command=self.symptoms_listbox.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.symptoms_listbox.configure(yscrollcommand=scrollbar.set)
        
        selected_frame = ttk.LabelFrame(self.lists_frame, text="Selected Symptoms")
        selected_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        self.selected_listbox = tk.Listbox(selected_frame, height=15)
        self.selected_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        selected_scrollbar = ttk.Scrollbar(selected_frame, orient=tk.VERTICAL,
                                         command=self.selected_listbox.yview)
        selected_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.selected_listbox.configure(yscrollcommand=selected_scrollbar.set)
        
        buttons_frame = ttk.Frame(self.lists_frame)
        buttons_frame.grid(row=0, column=2, sticky=(tk.N, tk.S), padx=5)
        
        ttk.Button(buttons_frame, text="Add >", command=self.add_symptom).grid(row=0, column=0, pady=5)
        ttk.Button(buttons_frame, text="< Remove", command=self.remove_symptom).grid(row=1, column=0, pady=5)
        ttk.Button(buttons_frame, text="Clear All", command=self.clear_symptoms).grid(row=2, column=0, pady=5)
        
        self.lists_frame.columnconfigure(0, weight=1)
        self.lists_frame.columnconfigure(1, weight=1)
        self.lists_frame.rowconfigure(0, weight=1)
        
        self.diagnose_button = ttk.Button(self.result_frame, text="Diagnose",
                                        command=self.diagnose)
        self.diagnose_button.grid(row=0, column=0, pady=5)
        
        # Create a Text widget for displaying multiple disease probabilities
        self.result_text = tk.Text(self.result_frame, height=12, width=90, wrap=tk.WORD)
        self.result_text.grid(row=1, column=0, pady=5, padx=5, sticky=(tk.W, tk.E))
        self.result_text.config(state=tk.DISABLED)
        
        self.treatment_text = tk.Text(self.treatment_frame, height=16, width=90, wrap=tk.WORD)
        self.treatment_text.grid(row=0, column=0, pady=5, padx=5, sticky=(tk.W, tk.E))
        self.treatment_text.config(state=tk.DISABLED)
        
    def update_symptom_list(self, *args):
        search_term = self.search_var.get().lower()
        filtered_symptoms = [s for s in self.unique_symptoms if search_term in s.lower()]
        self.available_symptoms.set(filtered_symptoms)
    
    def add_symptom(self):
        selection = self.symptoms_listbox.curselection()
        if not selection:
            return
            
        symptom = self.symptoms_listbox.get(selection[0])
        if symptom not in self.selected_symptoms:
            self.selected_symptoms.append(symptom)
            self.selected_listbox.insert(tk.END, symptom)
    
    def remove_symptom(self):
        selection = self.selected_listbox.curselection()
        if not selection:
            return
            
        symptom = self.selected_listbox.get(selection[0])
        self.selected_symptoms.remove(symptom)
        self.selected_listbox.delete(selection[0])
    
    def clear_symptoms(self):
        self.selected_symptoms.clear()
        self.selected_listbox.delete(0, tk.END)
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.config(state=tk.DISABLED)
        self.treatment_text.config(state=tk.NORMAL)
        self.treatment_text.delete(1.0, tk.END)
        self.treatment_text.config(state=tk.DISABLED)
    
    def show_treatments(self, disease):
        self.treatment_text.config(state=tk.NORMAL)
        self.treatment_text.delete(1.0, tk.END)
        
        treatment_text = ""
        # Add disease description from CSV
        desc = self.descriptions.get(disease.strip().lower())
        if desc:
            treatment_text += f"Description: {desc}\n\n"
        precautions = self.precautions.get(disease.strip().lower())
        if precautions:
            treatment_text += "\nPrecautions:\n"
            for p in precautions:
                treatment_text += f"• {p}\n"
        else:
            treatment_text += "\nNo additional precautions found for this disease.\n"
        self.treatment_text.insert(tk.END, treatment_text)
        self.treatment_text.config(state=tk.DISABLED)
    
    def diagnose(self):
        if not self.selected_symptoms:
            messagebox.showwarning("Warning", "Please select at least one symptom.")
            return
        
        # Encode input symptoms
        input_vector = self.mlb.transform([self.selected_symptoms])
        
        # Get probabilities for all diseases
        probabilities = self.model.predict_proba(input_vector)[0]
        
        # Sort diseases by probability
        disease_probs = list(zip(self.disease_classes, probabilities))
        disease_probs.sort(key=lambda x: x[1], reverse=True)
        
        # Find top prediction
        top_disease, top_prob = disease_probs[0]
        
        # Display results
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        
        # Show top 5 most likely diseases
        self.result_text.insert(tk.END, "Diagnosis Results:\n\n")
        for disease, prob in disease_probs[:5]:
            probability = prob * 100
            if disease == top_disease:
                self.result_text.insert(tk.END, f"► {disease}: {probability:.2f}% (Most Likely)\n")
            else:
                self.result_text.insert(tk.END, f"  {disease}: {probability:.2f}%\n")
        
        self.result_text.config(state=tk.DISABLED)
        
        # Show treatment for top disease
        self.show_treatments(top_disease)

def main():
    root = ThemedTk(theme="arc")
    app = MedicalDiagnosis(root)
    root.mainloop()

if __name__ == "__main__":
    main() 