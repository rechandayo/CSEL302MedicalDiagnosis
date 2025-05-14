import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import json
from ttkthemes import ThemedTk

class MedicalDiagnosis:
    def __init__(self, root):
        self.root = root
        self.root.title("Medical Diagnosis System")
        self.root.geometry("800x600")
        
        self.initialize_model()
        self.create_frames()
        self.create_widgets()
        self.selected_symptoms = []
        
    def initialize_model(self):
        df = pd.read_csv("dataset.csv")
        df.fillna('', inplace=True)
        
        symptom_columns = df.columns[1:]
        self.unique_symptoms = set()
        for column in symptom_columns:
            symptoms = df[column].dropna().unique()
            self.unique_symptoms.update([s.strip().lower() for s in symptoms if s != ''])
        self.unique_symptoms = sorted(list(self.unique_symptoms))
        
        df['Symptoms'] = df[df.columns[1:]].values.tolist()
        df['Symptoms'] = df['Symptoms'].apply(lambda x: [i.strip().lower() for i in x if i != ''])
        
        self.mlb = MultiLabelBinarizer()
        X = self.mlb.fit_transform(df['Symptoms'])
        y = df['Disease']
        
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = MultinomialNB()
        self.model.fit(X_train, y_train)
    
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
        
        self.result_frame = ttk.LabelFrame(self.main_frame, text="Diagnosis Result", padding="5")
        self.result_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
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
                                         selectmode=tk.SINGLE, height=25, width=40)
        self.symptoms_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        scrollbar = ttk.Scrollbar(available_frame, orient=tk.VERTICAL, 
                                command=self.symptoms_listbox.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.symptoms_listbox.configure(yscrollcommand=scrollbar.set)
        
        selected_frame = ttk.LabelFrame(self.lists_frame, text="Selected Symptoms")
        selected_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        self.selected_listbox = tk.Listbox(selected_frame, height=25, width=40)
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
        
        self.result_label = ttk.Label(self.result_frame, text="")
        self.result_label.grid(row=1, column=0, pady=5)
        
        self.confidence_label = ttk.Label(self.result_frame, text="")
        self.confidence_label.grid(row=2, column=0, pady=5)
        
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
        self.result_label.config(text="")
        self.confidence_label.config(text="")
    
    def diagnose(self):
        if not self.selected_symptoms:
            messagebox.showwarning("Warning", "Please select at least one symptom.")
            return
            
        symptoms_encoded = self.mlb.transform([self.selected_symptoms])
        
        prediction = self.model.predict(symptoms_encoded)
        probabilities = self.model.predict_proba(symptoms_encoded)
        max_prob = max(probabilities[0]) * 100
        
        self.result_label.config(text=f"Predicted Disease: {prediction[0]}")
        self.confidence_label.config(text=f"Confidence: {max_prob:.2f}%")

def main():
    root = ThemedTk(theme="arc")
    app = MedicalDiagnosis(root)
    root.mainloop()

if __name__ == "__main__":
    main() 